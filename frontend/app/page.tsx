"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { ArrowRight, RefreshCw } from "lucide-react";
import { Badge, Card, ErrorBox, Notice, RangeBar, Seg, Skeleton, Stat, StatusPill } from "@/components/ui";
import { fdate, fdatetime, fetcher, IpoRow, pct, prob, rupee, STAGE_LABEL, times } from "@/lib/api";

type Filter = "all" | "mainboard" | "sme";

export default function Dashboard() {
  const [demo, setDemo] = useState(false);
  const [board, setBoard] = useState<Filter>("all");
  const { data, error, isLoading, mutate, isValidating } = useSWR(`/api/ipos?include_demo=${demo}`, fetcher, { refreshInterval: 5 * 60_000 });
  const health = useSWR("/api/health", fetcher);
  const items: IpoRow[] = useMemo(() => (data?.items ?? []).filter((r: IpoRow) => board === "all" || r.board === board), [data, board]);
  const open = items.filter((r) => r.stage === "open");
  const bestEv = [...open].filter((r) => r.min_application_ev).sort((a, b) => (b.min_application_ev!.ev_per_rupee) - (a.min_application_ev!.ev_per_rupee))[0];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Live IPO intelligence</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink-2">
            Every open and upcoming Indian IPO with official subscription data, grey-market premium, a deep-learning
            listing-return distribution and SEBI-rule allotment odds. Plan applications in the <Link href="/planner" className="text-accent underline">Planner</Link>.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Seg<Filter> value={board} onChange={setBoard} options={[{ value: "all", label: "All" }, { value: "mainboard", label: "Mainboard" }, { value: "sme", label: "SME" }]} />
          <label className="flex items-center gap-1.5 text-xs text-ink-2">
            <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} /> show fictional demo
          </label>
          <button className="btn btn-ghost" onClick={() => mutate()} disabled={isValidating}>
            <RefreshCw size={14} className={isValidating ? "animate-spin" : ""} /> Refresh
          </button>
        </div>
      </div>

      {error && <ErrorBox error={error} />}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Open now" value={open.length} hint={`${items.length} active / upcoming`} />
        <Stat label="Best expected return on blocked ₹ (open)" value={bestEv ? pct(100 * bestEv.min_application_ev!.ev_per_rupee, 2, true) : "—"}
          hint={bestEv?.name ?? "no open issue"} tone={bestEv && bestEv.min_application_ev!.ev_per_rupee > 0 ? "pos" : undefined} />
        <Stat label="Model" value={<span className="text-base">{health.data?.model_version ?? "—"}</span>} hint="deep MDN ensemble" />
        <Stat label="Data refreshed" value={<span className="text-base">{fdatetime(health.data?.last_ingest)}</span>} hint="auto-refresh every 30 min in market hours" />
      </div>

      {isLoading ? <Skeleton h={320} /> : (
        <Card title="Open & upcoming issues" subtitle="Model range = 80% predictive interval of listing-day return. Allotment = one minimum individual application.">
          <div className="-mx-4 overflow-x-auto md:-mx-5">
            <table className="data min-w-[1050px]">
              <thead>
                <tr>
                  <th className="pl-5">Issue</th><th>Dates</th><th>Price · Min ₹</th><th>Subscription</th><th>GMP</th>
                  <th>Model listing return</th><th>P(loss)</th><th>Allotment odds</th><th>E[profit] / min app</th><th />
                </tr>
              </thead>
              <tbody>
                {items.map((r) => <Row key={r.slug} r={r} />)}
                {!items.length && <tr><td colSpan={10} className="py-10 text-center text-ink-3">No active issues right now. Tick “show fictional demo” to explore the interface.</td></tr>}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <div className="grid gap-3 md:grid-cols-3">
        <Notice>
          <b>Why the minimum application?</b> In an oversubscribed retail book the lottery gives winners exactly one lot, so
          applying for more lots blocks more money without improving the odds.
        </Notice>
        <Notice>
          <b>Winner’s curse.</b> You are most likely to be allotted when demand — and therefore the listing pop — is weakest.
          IPOPilot couples demand and return in every scenario, so its expected profit is lower than the naive “P × GMP”.
        </Notice>
        <Notice kind="warning">
          <b>GMP is unofficial</b> and can be stale or manipulated. The model learns how much to trust it from history and
          shows each source with its timestamp.
        </Notice>
      </div>
    </div>
  );
}

function Row({ r }: { r: IpoRow }) {
  const p = r.prediction;
  const sub = r.subscription;
  const ev = r.min_application_ev;
  return (
    <tr>
      <td className="pl-5">
        <Link href={`/ipo/${r.slug}`} className="font-semibold text-ink hover:underline">{r.name}</Link>
        <div className="mt-1 flex flex-wrap items-center gap-1.5">
          <Badge kind={r.board === "sme" ? "sme" : "main"}>{r.board === "sme" ? "SME" : "Mainboard"}</Badge>
          <Badge>{STAGE_LABEL[r.stage] ?? r.stage}</Badge>
          {r.is_demo && <Badge kind="demo">fictional</Badge>}
        </div>
      </td>
      <td className="tnum whitespace-nowrap text-ink-2">{fdate(r.open_date)} – {fdate(r.close_date)}
        {r.stage === "open" && <div className="mt-1 h-1 w-24 rounded-full bg-surface-2"><div className="h-1 rounded-full bg-accent" style={{ width: `${100 * r.bid_elapsed}%` }} /></div>}
      </td>
      <td className="tnum whitespace-nowrap">
        {r.price_low && r.price_high && r.price_low !== r.price_high ? `₹${r.price_low}–${r.price_high}` : r.price ? `₹${r.price}` : "—"}
        <div className="text-xs text-ink-3">{r.min_investment ? rupee(r.min_investment) : "lot TBA"}{r.lot_size ? ` · ${r.lot_size} sh/lot` : ""}</div>
      </td>
      <td className="tnum whitespace-nowrap">
        {sub ? <>
          <div>{times(sub.total)} <span className="text-xs text-ink-3">total</span></div>
          <div className="text-xs text-ink-3">QIB {times(sub.qib)} · Retail {times(sub.rii)}</div>
        </> : <span className="text-ink-3">not open</span>}
        <div className="mt-0.5">
          {r.demand_stage === "final" ? <StatusPill status="good">final</StatusPill>
            : r.demand_stage === "live" ? <StatusPill status="info">live · forecast {times(r.allotment.rii_final_forecast?.[1])} retail</StatusPill>
              : <StatusPill status="warning">prior only</StatusPill>}
        </div>
      </td>
      <td className="tnum whitespace-nowrap">
        {r.gmp_consensus_pct != null ? pct(r.gmp_consensus_pct, 1, true) : <span className="text-ink-3">—</span>}
        <div className="text-xs text-ink-3">{r.gmp_latest.length ? `${r.gmp_latest.length} source${r.gmp_latest.length > 1 ? "s" : ""}` : "no data"}</div>
      </td>
      <td className="min-w-[170px]">
        <div className="tnum mb-1 text-sm"><b className={p.mean_pct >= 0 ? "text-pos" : "text-neg"}>{pct(p.mean_pct, 1, true)}</b>
          <span className="text-xs text-ink-3"> mean · {pct(p.q10_pct, 0)} to {pct(p.q90_pct, 0)}</span></div>
        <RangeBar lo={p.q10_pct} mid={p.median_pct} hi={p.q90_pct} />
      </td>
      <td className="tnum">
        {p.p_loss > 0.45 ? <StatusPill status="critical">{prob(p.p_loss, 0)}</StatusPill>
          : p.p_loss > 0.25 ? <StatusPill status="warning">{prob(p.p_loss, 0)}</StatusPill>
            : <StatusPill status="good">{prob(p.p_loss, 0)}</StatusPill>}
      </td>
      <td className="tnum whitespace-nowrap">{prob(r.allotment.p_mean)}
        <div className="text-xs text-ink-3">{prob(r.allotment.p_q10, 0)}–{prob(r.allotment.p_q90, 0)} range</div>
      </td>
      <td className="tnum whitespace-nowrap">
        {ev ? <><b className={ev.expected_profit >= 0 ? "text-pos" : "text-neg"}>{rupee(ev.expected_profit)}</b>
          <div className="text-xs text-ink-3">{pct(100 * ev.ev_per_rupee, 2, true)} of ₹ blocked</div></> : <span className="text-ink-3">—</span>}
      </td>
      <td className="pr-5"><Link href={`/ipo/${r.slug}`} aria-label={`Open ${r.name}`} className="text-ink-3 hover:text-ink"><ArrowRight size={16} /></Link></td>
    </tr>
  );
}
