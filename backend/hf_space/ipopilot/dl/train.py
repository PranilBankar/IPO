"""Train and evaluate listing-return models with a strict walk-forward time split.

    train : close_date <  2024-07-01
    valid : 2024-07-01 .. 2025-03-31   (hyper-parameter selection, variance calibration)
    test  : close_date >= 2025-04-01   (never touched during model selection)

Primary test protocol = WALK-FORWARD: for each test quarter every model is retrained on all
issues that closed before that quarter (exactly how the production system is retrained), with
an inner time-ordered validation split for early stopping. The frozen static split is also reported.

Models compared (all produce a predictive distribution):
    climatology   board-wise historical distribution (no issue information)
    gmp_only      GMP% as the point forecast + empirical residuals (the popular heuristic)
    ridge         linear model on the same features + Gaussian residual
    lightgbm      gradient boosted trees: mean + quantile models
    deep_mdn      deep ensemble of mixture density networks (ours)

Metrics (in listing-return % space): MAE of the median, RMSE of the mean, directional
accuracy, 80% interval coverage, CRPS (via pinball loss on a quantile grid), AUC of P(loss).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import settings
from ..db import session_scope
from .features import CATEGORICAL, NUMERIC_FEATURES, FeatureBuilder, load_tables
from .listing_model import ListingEnsemble, Preprocessor, train_member

log = logging.getLogger(__name__)

QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
SPLIT_VALID = dt.date(2024, 7, 1)
SPLIT_TEST = dt.date(2025, 4, 1)
ARCH = {"d": 96, "k": 4, "blocks": 3, "p": 0.15}


# --------------------------------------------------------------------------- helpers
def mixture_quantiles(w, mu, s, qs=QS, iters=50):
    lo = (mu - 6 * s).min(1)
    hi = (mu + 6 * s).max(1)
    out = np.zeros((len(w), len(qs)))
    for j, q in enumerate(qs):
        a, b = lo.copy(), hi.copy()
        for _ in range(iters):
            m = (a + b) / 2
            c = (w * norm.cdf((m[:, None] - mu) / s)).sum(1)
            a = np.where(c < q, m, a)
            b = np.where(c < q, b, m)
        out[:, j] = (a + b) / 2
    return out  # in y space


def evaluate(name: str, y_true_ret: np.ndarray, q_ret: np.ndarray, mean_ret: np.ndarray, p_loss: np.ndarray,
             boards: np.ndarray | None = None) -> dict:
    """All inputs in return space (fraction, e.g. 0.12 = +12%)."""
    from sklearn.metrics import roc_auc_score

    med = q_ret[:, QS.tolist().index(0.5)]
    err = y_true_ret[:, None] - q_ret
    pinball = np.maximum(QS * err, (QS - 1) * err).mean(1)
    loss = (y_true_ret < 0).astype(int)
    res = {
        "model": name, "n": int(len(y_true_ret)),
        "mae_pct": float(100 * np.abs(med - y_true_ret).mean()),
        "rmse_pct": float(100 * np.sqrt(((mean_ret - y_true_ret) ** 2).mean())),
        "directional_acc": float(((med > 0) == (y_true_ret > 0)).mean()),
        "coverage80": float(((y_true_ret >= q_ret[:, 0]) & (y_true_ret <= q_ret[:, -1])).mean()),
        "crps_pct": float(100 * 2 * pinball.mean()),
        "auc_loss": float(roc_auc_score(loss, p_loss)) if 0 < loss.sum() < len(loss) else None,
        "brier_loss": float(((p_loss - loss) ** 2).mean()),
    }
    if boards is not None:
        res["by_board"] = {}
        for b in np.unique(boards):
            m = boards == b
            res["by_board"][str(b)] = {
                "n": int(m.sum()), "mae_pct": float(100 * np.abs(med[m] - y_true_ret[m]).mean()),
                "directional_acc": float(((med[m] > 0) == (y_true_ret[m] > 0)).mean()),
                "coverage80": float(((y_true_ret[m] >= q_ret[m, 0]) & (y_true_ret[m] <= q_ret[m, -1])).mean()),
                "crps_pct": float(100 * 2 * pinball[m].mean()),
            }
    return res


def _design(df: pd.DataFrame, pre: Preprocessor) -> np.ndarray:
    x, c = pre.transform(df)
    oh = [np.eye(len(pre.vocab[col]))[c[:, j].numpy()] for j, col in enumerate(CATEGORICAL)]
    return np.hstack([x.numpy()] + oh)


# --------------------------------------------------------------------------- baselines
def baseline_climatology(tr, te):
    qs, means, pl = [], [], []
    for b in te["board"]:
        y = tr.loc[tr["board"] == b, "y"].to_numpy() if (tr["board"] == b).sum() > 30 else tr["y"].to_numpy()
        r = np.expm1(y)
        qs.append(np.quantile(r, QS))
        means.append(r.mean())
        pl.append((r < 0).mean())
    return np.array(qs), np.array(means), np.array(pl)


def baseline_gmp(tr, te):
    clim_q, clim_m, clim_p = baseline_climatology(tr, te)
    trg = tr[tr["gmp_missing"] == 0]
    resid = (trg["y"] - np.log1p(trg["gmp_last_pct"].clip(-0.6, 3))).to_numpy() if len(trg) > 20 else np.zeros(1)
    qs, means, pl = clim_q.copy(), clim_m.copy(), clim_p.copy()
    for i, (miss, g) in enumerate(zip(te["gmp_missing"].to_numpy(), te["gmp_last_pct"].to_numpy())):
        if miss == 0 and not np.isnan(g):
            r = np.expm1(np.log1p(np.clip(g, -0.6, 3)) + resid)
            qs[i], means[i], pl[i] = np.quantile(r, QS), r.mean(), (r < 0).mean()
    return qs, means, pl


def baseline_ridge(tr, te, pre):
    from sklearn.linear_model import RidgeCV
    Xtr, Xte = _design(tr, pre), _design(te, pre)
    m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr, tr["y"])
    sd = float(np.std(tr["y"] - m.predict(Xtr)))
    mu = m.predict(Xte)
    qy = mu[:, None] + sd * norm.ppf(QS)[None, :]
    return np.expm1(qy), np.expm1(mu + 0.5 * sd ** 2), norm.cdf((0 - mu) / sd)


def baseline_lgbm(tr, te, va):
    import lightgbm as lgb
    feats = NUMERIC_FEATURES

    def X(d):
        out = d[feats].astype(float).copy()
        out["board_code"] = (d["board"] == "sme").astype(int)
        return out

    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    mean_m = lgb.LGBMRegressor(**params).fit(X(tr), tr["y"], eval_set=[(X(va), va["y"])],
                                             callbacks=[lgb.early_stopping(50, verbose=False)])
    qy = np.zeros((len(te), len(QS)))
    for j, q in enumerate(QS):
        mq = lgb.LGBMRegressor(objective="quantile", alpha=float(q), **params).fit(
            X(tr), tr["y"], eval_set=[(X(va), va["y"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        qy[:, j] = mq.predict(X(te))
    qy = np.sort(qy, axis=1)  # fix quantile crossing
    mu = mean_m.predict(X(te))
    # P(loss) by interpolating the quantile function at y=0
    pl = np.array([np.interp(0.0, row, QS, left=0.02, right=0.98) for row in qy])
    resid_sd = float(np.std(va["y"] - mean_m.predict(X(va))))
    return np.expm1(qy), np.expm1(mu + 0.5 * resid_sd ** 2), pl, mean_m


# --------------------------------------------------------------------------- deep model
# hyper-parameters selected on the VALIDATION period only (see docs/PROJECT_GUIDE.md §5.4)
RECENCY_HALF_LIFE_Y = 4.0
GMP_MASK_P = 0.0
WEIGHT_DECAY = 1e-3


def recency_weights(df: pd.DataFrame, ref: dt.date | None = None) -> np.ndarray:
    ref = ref or max(df["close_date"])
    age = np.array([(ref - d).days / 365.25 for d in df["close_date"]])
    return 0.5 ** (np.clip(age, 0, None) / RECENCY_HALF_LIFE_Y)


def fit_ensemble(tr, va, n: int, epochs: int, fixed_epochs: list[int] | int | None = None,
                 pre: Preprocessor | None = None):
    pre = pre or Preprocessor().fit(tr)
    w = recency_weights(tr)
    members, eps = [], []
    for seed in range(n):
        fe = fixed_epochs[seed] if isinstance(fixed_epochs, list) else fixed_epochs
        m, ep, best = train_member(pre, tr, va, seed=seed, epochs=epochs, arch=ARCH, fixed_epochs=fe,
                                   weights=w, gmp_mask_p=GMP_MASK_P, wd=WEIGHT_DECAY)
        members.append(m)
        eps.append(ep)
        log.info("member %d: best_epoch=%d%s", seed, ep, f" val_nll={best:.4f}" if va is not None else " (refit)")
    return ListingEnsemble(pre, members, 1.0, {"arch": ARCH}), eps


def calibrate_temperature(ens: ListingEnsemble, va: pd.DataFrame) -> float:
    y = va["y"].to_numpy()
    grid = np.linspace(0.6, 2.0, 57)
    nlls = [ens.nll(va, y, temp=t) for t in grid]
    return float(grid[int(np.argmin(nlls))])


def ensemble_outputs(ens: ListingEnsemble, df: pd.DataFrame):
    w, mu, s = ens.raw(df)
    qy = mixture_quantiles(w, mu, s)
    mean = (w * np.exp(mu + 0.5 * s ** 2)).sum(1) - 1
    pl = (w * norm.cdf((0 - mu) / s)).sum(1)
    return np.expm1(qy), mean, pl


# --------------------------------------------------------------------------- main
def build_frame() -> pd.DataFrame:
    with session_scope() as s:
        tables = load_tables(s)
    fb = FeatureBuilder(tables)
    return fb.training_frame()


MODELS = ["climatology", "gmp_only", "ridge", "lightgbm", "deep_mdn"]


def fit_predict_all(hist: pd.DataFrame, target: pd.DataFrame, ensemble: int, epochs: int,
                    inner_valid_frac: float = 0.15) -> tuple[dict, dict]:
    """Train every model on `hist` only (inner time-ordered validation split for early stopping
    and calibration) and predict `target`. Returns ({model: (q, mean, p_loss)}, info)."""
    hist = hist.sort_values("close_date").reset_index(drop=True)
    cut = int(len(hist) * (1 - inner_valid_frac))
    itr, iva = hist.iloc[:cut].reset_index(drop=True), hist.iloc[cut:].reset_index(drop=True)
    out = {
        "climatology": baseline_climatology(hist, target),
        "gmp_only": baseline_gmp(hist, target),
        "ridge": baseline_ridge(hist, target, Preprocessor().fit(hist)),
    }
    q, m, p, lgbm_mean = baseline_lgbm(itr, target, iva)
    out["lightgbm"] = (q, m, p)
    ens, eps = fit_ensemble(itr, iva, ensemble, epochs, pre=Preprocessor().fit(itr))
    ens.temp = calibrate_temperature(ens, iva)
    out["deep_mdn"] = ensemble_outputs(ens, target)
    return out, {"ens": ens, "epochs": eps, "temp": ens.temp, "lgbm": lgbm_mean}


def train_all(epochs: int = 200, ensemble: int = 5) -> dict:
    df = build_frame()
    df["close_date"] = pd.to_datetime(df["close_date"]).dt.date
    tr = df[df["close_date"] < SPLIT_VALID].reset_index(drop=True)
    va = df[(df["close_date"] >= SPLIT_VALID) & (df["close_date"] < SPLIT_TEST)].reset_index(drop=True)
    te = df[df["close_date"] >= SPLIT_TEST].reset_index(drop=True)
    log.info("rows: train=%d valid=%d test=%d (gmp coverage test=%.0f%%)", len(tr), len(va), len(te),
             100 * (te["gmp_missing"] == 0).mean())

    # ---- (1) walk-forward evaluation: every model retrained each quarter on all data before it
    quarters = sorted({(d.year, (d.month - 1) // 3) for d in te["close_date"]})
    wf_rows, wf_preds = [], {k: [] for k in MODELS}
    for (y, qn) in quarters:
        q_start = dt.date(y, 3 * qn + 1, 1)
        q_end = dt.date(y + 1, 1, 1) if qn == 3 else dt.date(y, 3 * qn + 4, 1)
        target = te[(te["close_date"] >= q_start) & (te["close_date"] < q_end)].reset_index(drop=True)
        if target.empty:
            continue
        hist = df[df["close_date"] < q_start]
        log.info("walk-forward %d-Q%d: train on %d issues, predict %d", y, qn + 1, len(hist), len(target))
        preds, _ = fit_predict_all(hist, target, ensemble, epochs)
        for k in MODELS:
            wf_preds[k].append(preds[k])
        wf_rows.append(target)
    wf = pd.concat(wf_rows, ignore_index=True)
    cat = {k: tuple(np.concatenate([p[j] for p in wf_preds[k]]) for j in range(3)) for k in MODELS}
    yte = np.expm1(wf["y"].to_numpy())
    boards = wf["board"].to_numpy()
    results = [evaluate(k, yte, *cat[k], boards) for k in MODELS]
    sub = wf["gmp_missing"].to_numpy() == 0
    subset = ([evaluate(k, yte[sub], cat[k][0][sub], cat[k][1][sub], cat[k][2][sub]) for k in MODELS]
              if sub.sum() > 20 else [])

    # ---- (2) static split (train+valid -> frozen -> whole test), reported for transparency
    static_preds, info = fit_predict_all(pd.concat([tr, va]), te, ensemble, epochs)
    static = [evaluate(k, np.expm1(te["y"].to_numpy()), *static_preds[k], te["board"].to_numpy()) for k in MODELS]

    # ---- save walk-forward predictions for the strategy backtest
    out_dir = settings.models_dir
    rows = wf[["slug", "board", "close_date", "gain_pct", "log_sub_rii", "log_sub_total", "log_price",
               "gmp_last_pct", "gmp_missing"]].copy()
    for name, (q, m, p) in cat.items():
        rows[f"{name}_mean"] = m
        rows[f"{name}_q10"] = q[:, 0]
        rows[f"{name}_q50"] = q[:, 4]
        rows[f"{name}_q90"] = q[:, -1]
        rows[f"{name}_ploss"] = p
    rows.to_pickle(out_dir / "test_predictions.pkl")

    # ---- (3) production model on ALL data; each member stops at its validated best epoch
    full = df.sort_values("close_date").reset_index(drop=True)
    cut = int(len(full) * 0.85)
    f_tr, f_va = full.iloc[:cut].reset_index(drop=True), full.iloc[cut:].reset_index(drop=True)
    ens_v, eps = fit_ensemble(f_tr, f_va, ensemble, epochs)
    temp = calibrate_temperature(ens_v, f_va)
    prod, _ = fit_ensemble(full, None, ensemble, epochs, fixed_epochs=[max(e, 5) for e in eps])
    prod.temp = temp
    version = f"deep_mdn_{dt.date.today():%Y%m%d}"
    prod.meta.update({"version": version, "trained_rows": len(full), "epochs": eps, "temp": temp,
                      "recency_half_life_years": RECENCY_HALF_LIFE_Y,
                      "split": {"valid": str(SPLIT_VALID), "test": str(SPLIT_TEST)}, "features": NUMERIC_FEATURES})
    prod.save(out_dir / "listing_current")

    imp = sorted(zip(NUMERIC_FEATURES + ["board_code"], info["lgbm"].booster_.feature_importance("gain").tolist()),
                 key=lambda x: -x[1])
    report = {
        "version": version, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "protocol": "walk-forward: every model retrained each quarter on all issues that closed before it",
        "rows": {"train": len(tr), "valid": len(va), "test": len(te)},
        "split": {"valid": str(SPLIT_VALID), "test": str(SPLIT_TEST)},
        "gmp_coverage": {k: float((d["gmp_missing"] == 0).mean()) for k, d in (("train", tr), ("valid", va), ("test", te))},
        "temperature": temp, "ensemble_size": ensemble, "best_epochs": eps,
        "hyperparameters": {"recency_half_life_years": RECENCY_HALF_LIFE_Y, "gmp_mask_p": GMP_MASK_P,
                            "weight_decay": WEIGHT_DECAY, **ARCH},
        "test_metrics": results, "test_metrics_gmp_subset": subset, "test_metrics_static": static,
        "lightgbm_feature_gain": [{"feature": f, "gain": float(g)} for f, g in imp[:20]],
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    log.info("saved model %s", version)
    return report
