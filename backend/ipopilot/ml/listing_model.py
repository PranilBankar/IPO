"""Model 1 — Deep probabilistic listing-return model.

Architecture (PyTorch):
    numeric features (standardised, NaN -> 0 + explicit missing flags)
    + learned embeddings for board and sector
    -> 3 residual MLP blocks (LayerNorm, GELU, dropout)
    -> Mixture Density Network head: K Gaussian components over y = log(1+return)

A *deep ensemble* of independently-seeded networks is trained; the predictive distribution
is the equal-weight mixture of all members' mixtures (captures both aleatoric and epistemic
uncertainty). A scalar variance temperature is fitted on the validation period so the
predicted intervals are calibrated.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import CATEGORICAL, NUMERIC_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    vocab: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        for c in NUMERIC_FEATURES:
            v = pd.to_numeric(df[c], errors="coerce")
            self.means[c] = float(np.nanmean(v)) if v.notna().any() else 0.0
            sd = float(np.nanstd(v)) if v.notna().sum() > 1 else 1.0
            self.stds[c] = sd if sd > 1e-6 else 1.0
        for c in CATEGORICAL:
            self.vocab[c] = ["<unk>"] + sorted(df[c].fillna("unknown").astype(str).unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        X = np.zeros((len(df), len(NUMERIC_FEATURES)), dtype=np.float32)
        for j, c in enumerate(NUMERIC_FEATURES):
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            z = (v - self.means[c]) / self.stds[c]
            X[:, j] = np.nan_to_num(np.clip(z, -6, 6), nan=0.0)
        C = np.zeros((len(df), len(CATEGORICAL)), dtype=np.int64)
        for j, c in enumerate(CATEGORICAL):
            idx = {v: i for i, v in enumerate(self.vocab[c])}
            C[:, j] = [idx.get(str(v), 0) for v in df[c].fillna("unknown").astype(str)]
        return torch.from_numpy(X), torch.from_numpy(C)

    def to_json(self) -> dict:
        return {"means": self.means, "stds": self.stds, "vocab": self.vocab}

    @classmethod
    def from_json(cls, d: dict) -> "Preprocessor":
        return cls(d["means"], d["stds"], d["vocab"])


# --------------------------------------------------------------------------- network
class ResBlock(nn.Module):
    def __init__(self, d: int, p: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d * 2, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.norm(x))))))


class ListingMDN(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], d: int = 96, k: int = 4, blocks: int = 3, p: float = 0.15):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, min(8, (n + 1) // 2 + 1)) for n in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        self.inp = nn.Linear(n_num + emb_dim, d)
        self.blocks = nn.Sequential(*[ResBlock(d, p) for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3 * k)
        # "wide" path: a linear map from the raw (standardised) inputs to the component means
        # makes simple monotone relations (listing ≈ GMP) trivially learnable on small data
        self.wide = nn.Linear(n_num, k)
        nn.init.zeros_(self.wide.weight)
        self.k = k

    def forward(self, x_num, x_cat):
        e = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        h = self.inp(torch.cat([x_num] + e, dim=1))
        h = self.norm(self.blocks(F.gelu(h)))
        out = self.head(h)
        logit_pi, mu, log_sigma = out.split(self.k, dim=1)
        mu = mu + self.wide(x_num)
        log_sigma = torch.clamp(log_sigma, -4.5, 1.5)
        return logit_pi, mu, log_sigma


def mdn_nll(logit_pi, mu, log_sigma, y, temp: float = 1.0):
    sigma = torch.exp(log_sigma) * temp
    log_pi = F.log_softmax(logit_pi, dim=1)
    z = (y.unsqueeze(1) - mu) / sigma
    log_prob = -0.5 * z ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(log_pi + log_prob, dim=1)


# --------------------------------------------------------------------------- distribution helpers
@dataclass
class MixtureDist:
    """Predictive distribution for ONE issue over y = log(1+r): weights/mus/sigmas (M,)."""
    w: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def sample_y(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(len(self.w), size=n, p=self.w)
        return rng.normal(self.mu[comp], self.sigma[comp])

    def sample_return(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.expm1(self.sample_y(n, rng))

    def cdf_y(self, y: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        y = np.atleast_1d(y)[:, None]
        return (self.w[None, :] * norm.cdf((y - self.mu[None, :]) / self.sigma[None, :])).sum(1)

    def quantile_return(self, q: float) -> float:
        lo, hi = float((self.mu - 6 * self.sigma).min()), float((self.mu + 6 * self.sigma).max())
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.cdf_y(np.array([mid]))[0] < q:
                lo = mid
            else:
                hi = mid
        return float(np.expm1((lo + hi) / 2))

    def mean_return(self) -> float:
        # E[e^y]-1 for a Gaussian mixture
        return float((self.w * np.exp(self.mu + 0.5 * self.sigma ** 2)).sum() - 1)

    def prob_loss(self) -> float:
        return float(self.cdf_y(np.array([0.0]))[0])

    def summary(self) -> dict:
        return {
            "mean_pct": 100 * self.mean_return(), "median_pct": 100 * self.quantile_return(0.5),
            "q05_pct": 100 * self.quantile_return(0.05), "q10_pct": 100 * self.quantile_return(0.10),
            "q25_pct": 100 * self.quantile_return(0.25), "q75_pct": 100 * self.quantile_return(0.75),
            "q90_pct": 100 * self.quantile_return(0.90), "q95_pct": 100 * self.quantile_return(0.95),
            "p_loss": self.prob_loss(),
        }

    def to_json(self) -> dict:
        return {"w": self.w.tolist(), "mu": self.mu.tolist(), "sigma": self.sigma.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "MixtureDist":
        return cls(np.array(d["w"]), np.array(d["mu"]), np.array(d["sigma"]))


# --------------------------------------------------------------------------- ensemble wrapper
class ListingEnsemble:
    def __init__(self, pre: Preprocessor, members: list[ListingMDN], temp: float = 1.0, meta: dict | None = None):
        self.pre, self.members, self.temp, self.meta = pre, members, temp, meta or {}

    @torch.no_grad()
    def raw(self, df: pd.DataFrame):
        x, c = self.pre.transform(df)
        x, c = x.to(DEVICE), c.to(DEVICE)
        ws, mus, sigs = [], [], []
        for m in self.members:
            m.eval()
            lp, mu, ls = m(x, c)
            ws.append(torch.softmax(lp, 1) / len(self.members))
            mus.append(mu)
            sigs.append(torch.exp(ls) * self.temp)
        return (torch.cat(ws, 1).cpu().numpy(), torch.cat(mus, 1).cpu().numpy(), torch.cat(sigs, 1).cpu().numpy())

    def predict(self, df: pd.DataFrame) -> list[MixtureDist]:
        w, mu, s = self.raw(df)
        return [MixtureDist(w[i], mu[i], s[i]) for i in range(len(df))]

    def nll(self, df: pd.DataFrame, y: np.ndarray, temp: float | None = None) -> float:
        w, mu, s = self.raw(df)
        if temp is not None:
            s = s / self.temp * temp
        z = (y[:, None] - mu) / s
        lp = np.log(w + 1e-12) - 0.5 * z ** 2 - np.log(s) - 0.5 * math.log(2 * math.pi)
        m = lp.max(1, keepdims=True)
        return float(-(m[:, 0] + np.log(np.exp(lp - m).sum(1))).mean())

    def explain(self, df_row: pd.DataFrame, top: int = 8) -> list[dict]:
        """Occlusion attribution: change in predicted mean return when a feature group is
        reset to its training-population average (i.e. 'what this information contributed')."""
        groups = {
            "GMP (grey market)": ["gmp_last_pct", "gmp_max_pct", "gmp_trend_pct", "gmp_n_obs", "gmp_missing"],
            "Overall subscription": ["log_sub_total", "sub_missing"],
            "QIB demand": ["log_sub_qib"], "NII demand": ["log_sub_nii"], "Retail demand": ["log_sub_rii", "cat_missing"],
            "Issue size": ["log_size", "size_missing"], "Issue price": ["log_price"],
            "Anchor book": ["anchor_pct", "anchor_missing"],
            "Market regime (NIFTY/VIX)": ["nifty_ret20", "nifty_ret60", "nifty_vol20", "vix", "smallcap_ret20", "market_missing"],
            "IPO market heat": ["heat_board10", "heat_all20", "heat_std10", "n_ipos_30d", "heat_missing"],
            "News sentiment (FinBERT)": ["sent_mean", "sent_neg_share", "news_count", "sent_missing"],
        }
        base = self.predict(df_row)[0].mean_return()
        out = []
        for gname, cols in groups.items():
            d2 = df_row.copy()
            for c in cols:
                d2[c] = self.pre.means.get(c, 0.0)
            alt = self.predict(d2)[0].mean_return()
            out.append({"feature": gname, "contribution_pct": 100 * (base - alt)})
        out.sort(key=lambda r: -abs(r["contribution_pct"]))
        return out[:top]

    # ---- persistence
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.members], path / "members.pt")
        cfg = {"pre": self.pre.to_json(), "temp": self.temp, "meta": self.meta,
               "arch": {"n_num": len(NUMERIC_FEATURES), "cat_sizes": [len(self.pre.vocab[c]) for c in CATEGORICAL],
                        **self.meta.get("arch", {})}}
        (path / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ListingEnsemble":
        cfg = json.loads((path / "config.json").read_text())
        pre = Preprocessor.from_json(cfg["pre"])
        arch = cfg["arch"]
        states = torch.load(path / "members.pt", map_location=DEVICE, weights_only=True)
        members = []
        for st in states:
            m = ListingMDN(arch["n_num"], arch["cat_sizes"], d=arch.get("d", 96), k=arch.get("k", 4),
                           blocks=arch.get("blocks", 3), p=arch.get("p", 0.15)).to(DEVICE)
            m.load_state_dict(st)
            m.eval()
            members.append(m)
        return cls(pre, members, cfg["temp"], cfg["meta"])


GMP_COLS = [c for c in NUMERIC_FEATURES if c.startswith("gmp_") and c != "gmp_missing"]


def train_member(pre: Preprocessor, tr: pd.DataFrame, va: pd.DataFrame | None, seed: int, epochs: int = 200,
                 lr: float = 1.5e-3, wd: float = 1e-3, batch: int = 128, arch: dict | None = None,
                 patience: int = 40, fixed_epochs: int | None = None, weights: np.ndarray | None = None,
                 gmp_mask_p: float = 0.0) -> tuple[ListingMDN, int, float]:
    """Train one ensemble member.

    * cosine LR schedule over `epochs` — identical for the validation run and the production
      refit (which simply stops at the validated best epoch via `fixed_epochs`)
    * optional per-sample weights (recency weighting for regime shift)
    * GMP-masking augmentation: with prob. `gmp_mask_p` a row's GMP features are replaced by the
      'missing' encoding, so the network learns both regimes (with and without GMP)
    """
    arch = arch or {}
    torch.manual_seed(seed)
    np.random.seed(seed)
    xtr, ctr = pre.transform(tr)
    ytr = torch.tensor(tr["y"].to_numpy(), dtype=torch.float32)
    wtr = torch.tensor(weights if weights is not None else np.ones(len(tr)), dtype=torch.float32)
    wtr = wtr / wtr.mean()
    model = ListingMDN(xtr.shape[1], [len(pre.vocab[c]) for c in CATEGORICAL], d=arch.get("d", 96),
                       k=arch.get("k", 4), blocks=arch.get("blocks", 3), p=arch.get("p", 0.15)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
    n_ep = fixed_epochs or epochs
    gidx = torch.tensor([NUMERIC_FEATURES.index(c) for c in GMP_COLS], device=DEVICE)
    miss_j = NUMERIC_FEATURES.index("gmp_missing")
    miss_z = float(np.clip((1.0 - pre.means["gmp_missing"]) / pre.stds["gmp_missing"], -6, 6))
    has_gmp = (pd.to_numeric(tr["gmp_missing"]).to_numpy() == 0)
    has_gmp = torch.tensor(has_gmp, device=DEVICE)
    xtr, ctr, ytr, wtr = xtr.to(DEVICE), ctr.to(DEVICE), ytr.to(DEVICE), wtr.to(DEVICE)
    if va is not None:
        xva, cva = (t.to(DEVICE) for t in pre.transform(va))
        yva = torch.tensor(va["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
    best, best_state, best_ep, bad = float("inf"), None, n_ep, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(ytr), generator=g).to(DEVICE)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb = xtr[idx] + 0.03 * torch.randn_like(xtr[idx])
            if gmp_mask_p > 0:
                m = has_gmp[idx] & (torch.rand(len(idx), device=DEVICE) < gmp_mask_p)
                if m.any():
                    xb[m.nonzero(as_tuple=True)[0].unsqueeze(1), gidx.unsqueeze(0)] = 0.0
                    xb[m, miss_j] = miss_z
            loss = (mdn_nll(*model(xb, ctr[idx]), ytr[idx]) * wtr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if va is not None and fixed_epochs is None:
            model.eval()
            with torch.no_grad():
                v = mdn_nll(*model(xva, cva), yva).mean().item()
            if v < best - 1e-4:
                best, best_ep, bad = v, ep + 1, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best
