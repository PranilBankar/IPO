"use client";
import Link from "next/link";
import { useState } from "react";
import { Loader2, Plus, Sparkles, Trash2 } from "lucide-react";
import { CapitalTimeline, ProfitHistogram } from "@/components/charts";
import { Badge, Card, Field, Notice, Seg, Stat, StatusPill } from "@/components/ui";
import { api, ComparisonRow, fdate, Plan, PlanResponse, pct, prob, rupee, rupeeShort } from "@/lib/api";

type Objective = "compare" | "max_ev" | "max_allot_prob" | "capital_efficiency" | "risk_adjusted";
type Risk = "conservative" | "moderate" | "aggressive";
type Applicant = { name: string; pan: string; capital?: number };

const OBJ_OPTS: { value: Objective; label: string; desc: string }[] = [
  { value: "compare", label: "E · Compare all four", desc: "Generate the four plans side by side." },
  { value: "max_ev", label: "A · Max expected profit", desc: "Highest expected rupee profit." },
  { value: "max_allot_prob", label: "B · Max P(≥1 allotment)", desc: "Most likely to get any shares (EV ≥ 0)." },
  { value: "capital_efficiency", label: "C · Max return on blocked ₹", desc: "Best expected profit per rupee blocked." },
  { value: "risk_adjusted", label: "D · Risk-adjusted", desc: "Expected profit minus a downside (CVaR) penalty." },
];

export default function Planner() {
  const [capital, setCapital] = useState(100000);
  const [apps, setApps] = useState<Applicant[]>([{ name: "Me", pan: "PAN-1" }, { name: "Spouse", pan: "PAN-2" }, { name: "Parent", pan: "PAN-3" }]);
  const [objective, setObjective] = useState<Objective>("compare");
  const [risk, setRisk] = useState<Risk>("moderate");
  const [boards, setBoards] = useState<string[]>(["mainboard", "sme"]);
  const [holding, setHolding] = useState<"listing" | "long_term">("listing");
  const [allowNii, setAllowNii] = useState(true);
  const [demo, setDemo] = useState(false);
  const [res, setRes] = useState<PlanResponse | null>(null);
  const [tab, setTab] = useState<string>("max_ev");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.post<PlanResponse>("/api/plan", {
        capital, objective, risk, boards, holding, allow_nii: allowNii, include_demo: demo,
        applicants: apps.map((a) => ({ name: a.name || "Applicant", pan: a.pan || undefined, capital: a.capital || undefined })),
      });
      setRes(r);
      if ("comparison" in r) setTab("max_ev");
    } catch (e) { setErr(String((e as Error).message)); } finally { setBusy(false); }
  };

  const plan: Plan | null = res ? ("plans" in res ? res.plans[tab] : (res as Plan)) : null;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Investment planner</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          Tell IPOPilot your capital and who can apply. An integer-programming optimizer (OR-Tools CP-SAT) chooses which IPO,
          which category and how many lots each PAN should apply for, respecting SEBI’s one-application-per-PAN rule and the days
          your money is blocked. Every plan is then stress-tested with thousands of correlated scenarios.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
        <Card title="Your situation" className="h-fit">
          <div className="space-y-4">
            <Field label="Total available capital (₹)" hint={rupeeShort(capital)}>
              <input className="input tnum" type="number" min={1000} step={1000} value={capital} onChange={(e) => setCapital(Number(e.target.value))} />
            </Field>
            <div>
              <div className="mb-1 flex items-center justify-between text-xs font-medium text-ink-2">
                <span>Eligible applicants (one per PAN)</span>
                <button className="inline-flex items-center gap-1 text-accent" onClick={() => setApps([...apps, { name: `Applicant ${apps.length + 1}`, pan: `PAN-${apps.length + 1}` }])}><Plus size={13} /> add</button>
              </div>
              <div className="space-y-2">
                {apps.map((a, i) => (
                  <div key={i} className="grid grid-cols-[1fr_92px_28px] gap-1.5">
                    <input className="input" value={a.name} onChange={(e) => setApps(apps.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} placeholder="Name" />
                    <input className="input" value={a.pan} onChange={(e) => setApps(apps.map((x, j) => (j === i ? { ...x, pan: e.target.value } : x)))} placeholder="PAN label" />
                    <button aria-label="remove" className="text-ink-3 hover:text-neg disabled:opacity-30" disabled={apps.length === 1} onClick={() => setApps(apps.filter((_, j) => j !== i))}><Trash2 size={15} /></button>
                  </div>
                ))}
              </div>
              <p className="mt-1 text-[0.7rem] text-ink-3">Use any unique label per PAN. Two demat accounts on one PAN are merged automatically.</p>
            </div>
            <Field label="Objective">
              <select className="input" value={objective} onChange={(e) => setObjective(e.target.value as Objective)}>
                {OBJ_OPTS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
              <span className="mt-1 block text-[0.7rem] text-ink-3">{OBJ_OPTS.find((o) => o.value === objective)?.desc}</span>
            </Field>
            <Field label="Risk preference">
              <Seg<Risk> value={risk} onChange={setRisk} options={[{ value: "conservative", label: "Conservative" }, { value: "moderate", label: "Moderate" }, { value: "aggressive", label: "Aggressive" }]} />
            </Field>
            <Field label="Categories">
              <div className="flex gap-3 text-sm">
                {["mainboard", "sme"].map((b) => (
                  <label key={b} className="flex items-center gap-1.5">
                    <input type="checkbox" checked={boards.includes(b)} onChange={(e) => setBoards(e.target.checked ? [...boards, b] : boards.filter((x) => x !== b))} />
                    {b === "sme" ? "SME" : "Mainboard"}
                  </label>
                ))}
              </div>
            </Field>
            <Field label="Holding preference">
              <Seg value={holding} onChange={setHolding} options={[{ value: "listing", label: "Sell on listing" }, { value: "long_term", label: "Hold long term" }]} />
            </Field>
            <label className="flex items-center gap-2 text-sm text-ink-2"><input type="checkbox" checked={allowNii} onChange={(e) => setAllowNii(e.target.checked)} /> allow NII (&gt;₹2L) applications</label>
            <label className="flex items-center gap-2 text-sm text-ink-2"><input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} /> include fictional demo IPO</label>
            <button className="btn w-full justify-center" onClick={run} disabled={busy || !boards.length}>
              {busy ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />} {busy ? "Optimising…" : "Generate plan"}
            </button>
            {err && <Notice kind="critical">{err}</Notice>}
          </div>
        </Card>

        <div className="space-y-4">
          {!res && !busy && (
            <Card title="How it decides">
              <ol className="list-decimal space-y-1.5 pl-5 text-sm text-ink-2">
                <li>For every open/upcoming IPO the deep model gives a full distribution of listing returns.</li>
                <li>The final subscription is forecast per category; allotment odds follow SEBI lottery rules.</li>
                <li>Demand and return are simulated <i>jointly</i> (high demand ⇒ bigger pop but lower odds).</li>
                <li>CP-SAT picks the best set of applications for your objective under capital-blocking constraints.</li>
                <li>The plan is simulated end-to-end: profit distribution, P(loss), CVaR, capital timeline.</li>
              </ol>
            </Card>
          )}
          {res?.warnings?.map((w, i) => <Notice key={i} kind="warning">{w}</Notice>)}
          {res && "comparison" in res && <Comparison rows={res.comparison} tab={tab} setTab={setTab} />}
          {plan && <PlanView plan={plan} />}
          {res && <p className="text-xs text-ink-3">{res.n_candidates} candidate IPOs · model {res.model_version} · generated {res.generated_at}</p>}
        </div>
      </div>
    </div>
  );
}

function Comparison({ rows, tab, setTab }: { rows: ComparisonRow[]; tab: string; setTab: (s: string) => void }) {
  const best = (k: keyof ComparisonRow, lowIsBest = false) => {
    const vals = rows.map((r) => Number(r[k]));
    return lowIsBest ? Math.min(...vals) : Math.max(...vals);
  };
  return (
    <Card title="Four objectives, four different decisions" subtitle="Click a row to inspect that plan. Best value in each column is bold.">
      <div className="-mx-4 overflow-x-auto md:-mx-5">
        <table className="data min-w-[720px]">
          <thead><tr><th className="pl-5">Objective</th><th>E[profit]</th><th>P(≥1 allotment)</th><th>P(loss)</th><th>CVaR 5%</th><th>Peak blocked</th><th>Return on peak ₹</th><th>Apps</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.objective} onClick={() => setTab(r.objective)} className="cursor-pointer" style={tab === r.objective ? { background: "var(--surface-2)" } : {}}>
                <td className="pl-5 font-medium">{tab === r.objective && "▸ "}{r.label}</td>
                <td className={`tnum ${r.expected_profit === best("expected_profit") ? "font-bold" : ""}`}>{rupee(r.expected_profit)}</td>
                <td className={`tnum ${r.p_any_allotment === best("p_any_allotment") ? "font-bold" : ""}`}>{prob(r.p_any_allotment)}</td>
                <td className={`tnum ${r.p_loss === best("p_loss", true) ? "font-bold" : ""}`}>{prob(r.p_loss)}</td>
                <td className={`tnum ${r.cvar5 === best("cvar5") ? "font-bold" : ""}`}>{rupee(r.cvar5)}</td>
                <td className="tnum">{rupee(r.peak_capital)}</td>
                <td className={`tnum ${r.return_on_peak_capital === best("return_on_peak_capital") ? "font-bold" : ""}`}>{pct(100 * r.return_on_peak_capital, 2, true)}</td>
                <td className="tnum">{r.applications}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function PlanView({ plan }: { plan: Plan }) {
  const s = plan.simulation;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Expected profit" value={rupee(s.expected_profit)} tone={s.expected_profit >= 0 ? "pos" : "neg"} hint={plan.objective_label} />
        <Stat label="P(≥1 allotment)" value={prob(s.p_any_allotment)} hint={`${(s.expected_allotments ?? 0).toFixed(2)} allotments expected`} />
        <Stat label="P(plan loses money)" value={prob(s.p_loss)} hint={`5% worst case avg ${rupee(s.cvar5)}`} />
        <Stat label="Peak capital blocked" value={rupee(plan.peak_capital_blocked)} hint={`${rupee(plan.min_free_cash)} always free`} />
        <Stat label="Applications" value={plan.total_applications} hint={`${plan.selected.length} IPOs`} />
      </div>

      <Card title="Recommended applications" subtitle="Bid on the closing day; funds unblock on allotment (T+1).">
        {plan.selected.length === 0 ? <p className="text-sm text-ink-3">No application passes your objective and risk filters — keeping cash is the recommendation.</p> : (
          <div className="space-y-4">
            {plan.selected.map((r) => (
              <div key={r.slug} className="rounded-lg border border-line p-3">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div>
                    <Link href={`/ipo/${r.slug}`} className="font-semibold hover:underline">{r.name}</Link>{" "}
                    <Badge kind={r.board === "sme" ? "sme" : "main"}>{r.board === "sme" ? "SME" : "Mainboard"}</Badge>{" "}
                    {r.is_demo && <Badge kind="demo">fictional</Badge>}
                    <div className="text-xs text-ink-3">₹{r.price} × {r.lot} sh/lot · bid by {fdate(r.close_date)} · unblock {fdate(r.unblock_date)}</div>
                  </div>
                  <div className="tnum text-right text-sm">
                    <b className={r.expected_profit >= 0 ? "text-pos" : "text-neg"}>{rupee(r.expected_profit)}</b> expected ·{" "}
                    {prob(r.p_any_allotment)} ≥1 allotment · {rupee(r.capital_blocked)} blocked
                  </div>
                </div>
                <table className="data mt-2">
                  <thead><tr><th>Applicant</th><th>Category</th><th>Lots (shares)</th><th>Amount</th><th>P(allot)</th><th>E[profit]</th></tr></thead>
                  <tbody>{r.applications.map((a) => (
                    <tr key={a.applicant_id}><td>{a.applicant}</td><td>{a.category_label}</td><td className="tnum">{a.lots} ({a.shares})</td>
                      <td className="tnum">{rupee(a.amount)}</td><td className="tnum">{prob(a.p_allot)}</td><td className="tnum">{rupee(a.expected_profit)}</td></tr>
                  ))}</tbody>
                </table>
                <p className="mt-2 text-xs leading-relaxed text-ink-2">{r.why}</p>
                {r.breakeven_shift_pct != null && (
                  <p className="mt-1 text-xs text-ink-3">Sensitivity: stays profitable in expectation unless the listing return comes in
                    ≥ <b className="tnum">{r.breakeven_shift_pct.toFixed(1)} pp</b> below the model’s forecast.</p>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Simulated profit distribution" subtitle={`${s.n_applications} applications · median ${rupee(s.median_profit)} · 95th pct ${rupee(s.q95_profit)}`}>
          {s.histogram.length ? <ProfitHistogram bins={s.histogram} /> : <p className="text-sm text-ink-3">Nothing to simulate.</p>}
          <p className="text-xs text-ink-3">The tall bar at ₹0 is the chance of not being allotted at all — your money simply comes back.</p>
        </Card>
        <Card title="Capital blocked by day" subtitle="Freed capital is reused for later IPOs.">
          {plan.capital_timeline.length ? <CapitalTimeline data={plan.capital_timeline} capital={plan.capital} /> : <p className="text-sm text-ink-3">—</p>}
        </Card>
      </div>

      {plan.excluded.length > 0 && (
        <Card title="Not selected — and why">
          <ul className="space-y-2 text-sm">
            {plan.excluded.map((e) => (
              <li key={e.slug} className="flex gap-2"><StatusPill status="warning">{""}</StatusPill>
                <div><Link href={`/ipo/${e.slug}`} className="font-medium hover:underline">{e.name}</Link> <span className="text-ink-2">— {e.reason}</span></div></li>
            ))}
          </ul>
        </Card>
      )}
      <Card title="Assumptions">
        <ul className="list-disc space-y-1 pl-5 text-xs text-ink-2">{plan.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
      </Card>
    </div>
  );
}
