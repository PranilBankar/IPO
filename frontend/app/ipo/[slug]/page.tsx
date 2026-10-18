"use client";
import Link from "next/link";
import { use, useEffect, useState } from "react";
import useSWR from "swr";
import { ArrowLeft, ExternalLink, FlaskConical, RotateCcw } from "lucide-react";
import { DistributionChart, ExplainBars, GmpChart, SubscriptionBars } from "@/components/charts";
import { Badge, Card, ErrorBox, Field, Notice, Skeleton, Stat, StatusPill } from "@/components/ui";
import { api, fdate, fdatetime, fetcher, IpoDetail, pct, prob, rupee, STAGE_LABEL, times } from "@/lib/api";

type WhatIf = { gmp_pct?: number; sub_total?: number; sub_qib?: number; sub_nii?: number; sub_rii?: number; final_subscription?: boolean };

export default function IpoPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = use(params);
  const { data, error, isLoading } = useSWR<IpoDetail>(`/api/ipos/${slug}`, fetcher);
  const [wi, setWi] = useState<WhatIf>({});
  const [sim, setSim] = useState<IpoDetail | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const active = Object.values(wi).some((v) => v !== undefined && v !== false);
    if (!active) { setSim(null); return; }
    const t = setTimeout(async () => {
      setBusy(true);
      try { setSim(await api.post<IpoDetail>(`/api/ipos/${slug}/whatif`, wi)); } finally { setBusy(false); }
    }, 350);
    return () => clearTimeout(t);
  }, [wi, slug]);

  if (error) return <ErrorBox error={error} />;
  if (isLoading || !data) return <div className="space-y-4"><Skeleton h={90} /><Skeleton h={300} /></div>;
  const d = sim ?? data;
  const p = d.prediction;
  const base = data.prediction;

  return (
    <div className="space-y-5">
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-ink-2 hover:text-ink"><ArrowLeft size={14} /> All IPOs</Link>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{data.name}</h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-sm text-ink-2">
            <Badge kind={data.board === "sme" ? "sme" : "main"}>{data.board === "sme" ? "SME" : "Mainboard"}</Badge>
            <Badge>{STAGE_LABEL[data.stage] ?? data.stage}</Badge>
            {data.is_demo && <Badge kind="demo">fictional demo</Badge>}
            <span className="tnum">Bid {fdate(data.open_date)} – {fdate(data.close_date)}</span>
            {data.listing_date && <span className="tnum">· Listing {fdate(data.listing_date)}</span>}
            {data.symbol && <span>· {data.symbol}</span>}
            {data.registrar && <span>· Registrar: {data.registrar}</span>}
          </div>
        </div>
        {sim && <Badge kind="accent">what-if scenario active</Badge>}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Price band" value={data.price_low && data.price_high && data.price_low !== data.price_high ? `₹${data.price_low}–${data.price_high}` : data.price ? `₹${data.price}` : "—"}
          hint={data.lot_size ? `${data.lot_size} shares / lot` : "lot size TBA"} />
        <Stat label="Minimum application" value={rupee(data.min_investment)} hint={data.allotment.unit_lots > 1 ? `${data.allotment.unit_lots} lots (SME rule)` : "1 lot"} />
        <Stat label="Expected listing return" value={pct(p.mean_pct, 1, true)} tone={p.mean_pct >= 0 ? "pos" : "neg"}
          hint={`80%: ${pct(p.q10_pct, 0)} to ${pct(p.q90_pct, 0)}`} />
        <Stat label="P(listing below issue)" value={prob(p.p_loss, 0)} hint={sim ? `baseline ${prob(base.p_loss, 0)}` : "model probability"} />
        <Stat label="Allotment odds (min app)" value={prob(d.allotment.p_mean)} hint={`${prob(d.allotment.p_q10, 0)}–${prob(d.allotment.p_q90, 0)} given uncertainty`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="Predicted listing-day return distribution"
          subtitle={`Deep MDN ensemble (${p.model_version}), integrated over final-subscription uncertainty. Shaded = 80% interval; dashed = median.`}>
          <DistributionChart data={d.distribution} q10={p.q10_pct} q90={p.q90_pct} median={p.median_pct} actual={data.actual_listing_gain_pct} />
          <div className="mt-2 grid grid-cols-4 gap-2 text-center text-xs text-ink-3">
            {(["q05_pct", "q25_pct", "q75_pct", "q95_pct"] as const).map((k) => (
              <div key={k}><div className="tnum text-sm font-semibold text-ink">{pct(p[k], 1, true)}</div>{k.slice(1, 3)}th percentile</div>
            ))}
          </div>
        </Card>

        <Card title={<span className="flex items-center gap-1.5"><FlaskConical size={16} /> What-if simulator</span>}
          subtitle="Change inputs and the model re-predicts instantly."
          right={<button className="btn btn-ghost !px-2 !py-1 text-xs" onClick={() => setWi({})}><RotateCcw size={12} /> reset</button>}>
          <div className="space-y-3">
            <Slider label="GMP (% of issue price)" min={-30} max={150} step={1} value={wi.gmp_pct ?? d.gmp_consensus_pct ?? 0}
              base={data.gmp_consensus_pct} fmt={(v) => pct(v, 0, true)} onChange={(v) => setWi({ ...wi, gmp_pct: v })} />
            <Slider label="Retail subscription (×)" min={0} max={300} step={0.5} value={wi.sub_rii ?? data.subscription?.rii ?? 5}
              base={data.subscription?.rii} fmt={times} onChange={(v) => setWi({ ...wi, sub_rii: v })} />
            <Slider label="QIB subscription (×)" min={0} max={400} step={0.5} value={wi.sub_qib ?? data.subscription?.qib ?? 5}
              base={data.subscription?.qib} fmt={times} onChange={(v) => setWi({ ...wi, sub_qib: v })} />
            <Slider label="NII subscription (×)" min={0} max={400} step={0.5} value={wi.sub_nii ?? data.subscription?.nii ?? 5}
              base={data.subscription?.nii} fmt={times} onChange={(v) => setWi({ ...wi, sub_nii: v })} />
            <Slider label="Total subscription (×)" min={0} max={400} step={0.5} value={wi.sub_total ?? data.subscription?.total ?? 5}
              base={data.subscription?.total} fmt={times} onChange={(v) => setWi({ ...wi, sub_total: v })} />
            <label className="flex items-center gap-2 text-xs text-ink-2">
              <input type="checkbox" checked={!!wi.final_subscription} onChange={(e) => setWi({ ...wi, final_subscription: e.target.checked })} />
              treat these as FINAL figures (no further demand growth)
            </label>
            {busy && <div className="text-xs text-ink-3">re-running model…</div>}
            {sim && (
              <div className="rounded-lg bg-surface-2 p-2.5 text-xs text-ink-2">
                Mean return <b className="tnum">{pct(base.mean_pct, 1, true)} → {pct(p.mean_pct, 1, true)}</b>; allotment odds{" "}
                <b className="tnum">{prob(data.allotment.p_mean)} → {prob(d.allotment.p_mean)}</b>; E[profit]/min app{" "}
                <b className="tnum">{rupee(data.min_application_ev?.expected_profit)} → {rupee(d.min_application_ev?.expected_profit)}</b>.
              </div>
            )}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Why this prediction?" subtitle="Occlusion attribution: change in predicted mean return when each information group is replaced by its historical average.">
          {d.explanation?.length ? <ExplainBars data={d.explanation} /> : <p className="text-sm text-ink-3">Train the model to see explanations (python -m ipopilot.cli train).</p>}
        </Card>
        <Card title="Allotment — more applicants, better odds?" subtitle={`Retail final subscription forecast: ${times(d.allotment.rii_final_forecast?.[0])} – ${times(d.allotment.rii_final_forecast?.[2])} (80%).`}>
          <table className="data">
            <thead><tr><th>Distinct PANs applying</th><th>P(≥1 allotment)</th><th>Naive 1−(1−p)ⁿ</th></tr></thead>
            <tbody>
              {d.allotment.multi.map((m) => (
                <tr key={m.n}><td className="tnum">{m.n}</td><td className="tnum font-semibold">{prob(m.p_at_least_one)}</td><td className="tnum text-ink-3">{prob(m.naive)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs leading-relaxed text-ink-3">
            The exact figure is lower than the naive formula because all your applications share the same unknown allotment ratio.
            Only <b>distinct PANs</b> count — several demat accounts on one PAN are a single application under SEBI rules.
          </p>
          {d.min_application_ev && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs text-ink-3">
              <div><div className="tnum text-sm font-semibold text-ink">{rupee(d.min_application_ev.expected_profit)}</div>E[profit] (coupled)</div>
              <div><div className="tnum text-sm font-semibold text-ink">{rupee(d.min_application_ev.naive_expected_profit)}</div>naive P × E[r]</div>
              <div><div className="tnum text-sm font-semibold text-ink">{pct(d.min_application_ev.return_given_allotted_pct, 1, true)}</div>return | allotted</div>
            </div>
          )}
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Grey market premium" subtitle="Unofficial. Every observation is stored with source and date; history is never overwritten.">
          {data.gmp_history.length ? <GmpChart data={data.gmp_history} /> : <p className="text-sm text-ink-3">No GMP observations yet.</p>}
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-ink-3">
            {data.gmp_latest.map((g) => <span key={g.source}><b className="text-ink-2">{g.source}</b>: {g.gmp_rs != null ? `₹${g.gmp_rs}` : "—"} ({pct(g.gmp_pct, 1, true)}) · {fdate(g.date)}</span>)}
          </div>
        </Card>
        <Card title="Subscription by category" subtitle={data.subscription ? `${data.subscription.source.toUpperCase()} · as of ${fdatetime(data.subscription.as_of)}${data.subscription.final ? " · final" : ""}` : "Bidding not started"}>
          {data.subscription ? <SubscriptionBars snap={data.subscription} /> : <p className="text-sm text-ink-3">No subscription data yet — the model uses the historical demand distribution as a prior.</p>}
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="News & FinBERT sentiment" subtitle="Headlines from Google News, scored by ProsusAI/finbert. Sentiment is one weak signal, not proof.">
          {data.news.length ? (
            <ul className="divide-y divide-[var(--border)]">
              {data.news.slice(0, 12).map((n) => {
                const net = (n.p_pos ?? 0) - (n.p_neg ?? 0);
                return (
                  <li key={n.url} className="flex items-start justify-between gap-3 py-2 text-sm">
                    <div>
                      <a href={n.url} target="_blank" rel="noreferrer" className="text-ink hover:underline">{n.title} <ExternalLink size={11} className="inline" /></a>
                      <div className="text-xs text-ink-3">{n.publisher ?? ""} · {fdatetime(n.published_at)} · {n.topic}</div>
                    </div>
                    {n.p_pos != null && (net > 0.3 ? <StatusPill status="good">positive</StatusPill> : net < -0.3 ? <StatusPill status="critical">negative</StatusPill> : <StatusPill status="info">neutral</StatusPill>)}
                  </li>
                );
              })}
            </ul>
          ) : <p className="text-sm text-ink-3">No headlines collected yet (they refresh hourly).</p>}
        </Card>
        <Card title="Application options" subtitle="Valid categories for one PAN under SEBI rules.">
          <table className="data">
            <thead><tr><th>Option</th><th>Amount</th></tr></thead>
            <tbody>{data.options.map((o) => <tr key={o.label}><td>{o.label}</td><td className="tnum">{rupee(o.amount)}</td></tr>)}</tbody>
          </table>
          <div className="mt-3"><Notice>Bids above ₹5 lakh can’t use UPI — use net-banking ASBA.</Notice></div>
        </Card>
      </div>
    </div>
  );
}

function Slider({ label, min, max, step, value, onChange, fmt, base }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; fmt: (v: number) => string; base?: number | null;
}) {
  return (
    <Field label={<span className="flex justify-between"><span>{label}</span><span className="tnum text-ink">{fmt(value)}{base != null && <span className="text-ink-3"> (now {fmt(base)})</span>}</span></span>}>
      <input type="range" className="w-full accent-[var(--accent)]" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </Field>
  );
}
