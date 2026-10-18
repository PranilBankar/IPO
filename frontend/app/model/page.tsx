"use client";
import useSWR from "swr";
import { Card, ErrorBox, Notice, Skeleton, Stat } from "@/components/ui";
import { fetcher, pct, prob, rupee } from "@/lib/api";

const MODEL_LABEL: Record<string, string> = {
  climatology: "Climatology (board average)", gmp_only: "GMP only", ridge: "Ridge regression", lightgbm: "LightGBM quantile", deep_mdn: "Deep MDN ensemble (ours)",
};
const STRAT_LABEL: Record<string, string> = {
  apply_all: "Apply to everything", gmp_rank: "Rank by GMP", rule_based: "Rule: subscription ≥ 10×",
  lightgbm_opt: "Optimizer + LightGBM", deep_mdn_opt: "Optimizer + Deep MDN (ours)", oracle: "Oracle (perfect foresight)",
};

export default function ModelPage() {
  const { data, error } = useSWR("/api/model", fetcher);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Skeleton h={400} />;
  const m = data.metrics;
  const bt = data.backtest;
  const bestOf = (rows: any[], k: string, low = true) => {
    const v = rows.map((r) => r[k]).filter((x) => x != null);
    return low ? Math.min(...v) : Math.max(...v);
  };
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Model & research</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          Research question: <i>does feeding deep-learning listing-return distributions into the allocation optimizer improve realised,
          risk-adjusted outcomes versus GMP-based and rule-based strategies?</i> Everything below is measured on IPOs the models never saw
          (walk-forward split by closing date).
        </p>
      </div>
      {!m && <Notice kind="warning">No trained model yet. Run <code>python -m ipopilot.cli train</code> in the backend.</Notice>}
      {m && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Model version" value={<span className="text-base">{m.version}</span>} hint={`ensemble of ${m.ensemble_size}`} />
            <Stat label="Train / valid / test" value={<span className="text-base tnum">{m.rows.train} / {m.rows.valid} / {m.rows.test}</span>} hint={`test from ${m.split?.test ?? "2025-04-01"}`} />
            <Stat label="GMP coverage (test)" value={prob(m.gmp_coverage.test, 0)} hint={`train ${prob(m.gmp_coverage.train, 0)}`} />
            <Stat label="Variance temperature" value={m.temperature.toFixed(2)} hint="fitted on validation NLL" />
          </div>
          <Card title="Listing-return forecasting — out-of-sample test" subtitle="Lower is better for MAE / RMSE / CRPS / Brier; coverage should be close to 80%. Best in bold.">
            <MetricsTable rows={m.test_metrics} bestOf={bestOf} />
          </Card>
          {m.test_metrics_gmp_subset?.length > 0 && (
            <Card title="Same metrics on issues where GMP was available" subtitle="The fair comparison against the GMP-only heuristic.">
              <MetricsTable rows={m.test_metrics_gmp_subset} bestOf={bestOf} />
            </Card>
          )}
          <Card title="By board (deep model)" subtitle="Mainboard and SME behave very differently — the model is evaluated on each.">
            <table className="data">
              <thead><tr><th>Board</th><th>n</th><th>MAE</th><th>Directional acc.</th><th>80% coverage</th><th>CRPS</th></tr></thead>
              <tbody>{Object.entries(m.test_metrics.find((r: any) => r.model === "deep_mdn")?.by_board ?? {}).map(([b, r]: any) => (
                <tr key={b}><td>{b}</td><td className="tnum">{r.n}</td><td className="tnum">{pct(r.mae_pct)}</td><td className="tnum">{prob(r.directional_acc)}</td><td className="tnum">{prob(r.coverage80)}</td><td className="tnum">{pct(r.crps_pct)}</td></tr>
              ))}</tbody>
            </table>
          </Card>
        </>
      )}
      {bt && (
        <Card title="Strategy backtest — realised outcomes" subtitle={`${bt.n_issues} test-period IPOs in weekly windows · ₹${(bt.capital / 1000).toFixed(0)}k capital · ${bt.applicants} PANs · lottery simulated with actual subscription and actual listing gains.`}>
          <div className="-mx-4 overflow-x-auto md:-mx-5">
            <table className="data min-w-[760px]">
              <thead><tr><th className="pl-5">Strategy</th><th>Total E[profit]</th><th>Per window</th><th>Return on capital</th><th>Loss windows</th><th>Worst window</th><th>Sharpe-like</th><th>P(≥1 allot)</th></tr></thead>
              <tbody>{bt.summary.map((r: any) => (
                <tr key={r.strategy} style={r.strategy === "deep_mdn_opt" ? { background: "var(--surface-2)" } : {}}>
                  <td className="pl-5 font-medium">{STRAT_LABEL[r.strategy] ?? r.strategy}</td>
                  <td className="tnum">{rupee(r.total_expected_profit)}</td><td className="tnum">{rupee(r.mean_profit_per_window)}</td>
                  <td className="tnum">{pct(r.return_on_capital_pct, 2, true)}</td><td className="tnum">{prob(r.loss_window_share, 0)}</td>
                  <td className="tnum">{rupee(r.worst_window)}</td><td className="tnum">{r.sharpe_like?.toFixed(2) ?? "—"}</td><td className="tnum">{prob(r.mean_p_any_allotment, 0)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-3">The oracle is an unreachable upper bound (it knows the listing price); it shows how much value is left on the table.</p>
        </Card>
      )}
      <Card title="Architecture">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
          <li><b>Model 1 — listing return:</b> PyTorch residual MLP with board/sector embeddings and a 4-component mixture-density head over log(1+r); deep ensemble of 5 seeds; variance temperature calibrated on validation.</li>
          <li><b>Model 2 — allotment:</b> explicit SEBI lottery mathematics with a Bayesian prior on average application size; exact when application counts are known.</li>
          <li><b>Model 3 — sentiment:</b> FinBERT (ProsusAI/finbert) on Google-News headlines, aggregated point-in-time into features.</li>
          <li><b>Demand forecaster:</b> final category subscription from intraday snapshots (self-refitting as data accumulates).</li>
          <li><b>Optimizer:</b> OR-Tools CP-SAT integer program; objectives A–D; capital-blocking day constraints; Monte-Carlo evaluation with a Gaussian copula.</li>
        </ul>
      </Card>
      {m?.lightgbm_feature_gain && (
        <Card title="Most informative features (LightGBM gain)">
          <table className="data"><tbody>{m.lightgbm_feature_gain.slice(0, 12).map((f: any) => (
            <tr key={f.feature}><td className="w-1/3 font-mono text-xs">{f.feature}</td>
              <td><div className="h-2 rounded-full bg-accent" style={{ width: `${(100 * f.gain) / m.lightgbm_feature_gain[0].gain}%` }} /></td></tr>
          ))}</tbody></table>
        </Card>
      )}
    </div>
  );
}

function MetricsTable({ rows, bestOf }: { rows: any[]; bestOf: (r: any[], k: string, low?: boolean) => number }) {
  const b = (k: string, v: number | null, low = true) => (v != null && v === bestOf(rows, k, low) ? "font-bold" : "");
  return (
    <div className="-mx-4 overflow-x-auto md:-mx-5">
      <table className="data min-w-[760px]">
        <thead><tr><th className="pl-5">Model</th><th>n</th><th>MAE</th><th>RMSE</th><th>CRPS</th><th>Directional acc.</th><th>80% coverage</th><th>AUC P(loss)</th><th>Brier</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.model} style={r.model === "deep_mdn" ? { background: "var(--surface-2)" } : {}}>
            <td className="pl-5 font-medium">{MODEL_LABEL[r.model] ?? r.model}</td><td className="tnum">{r.n}</td>
            <td className={`tnum ${b("mae_pct", r.mae_pct)}`}>{pct(r.mae_pct)}</td>
            <td className={`tnum ${b("rmse_pct", r.rmse_pct)}`}>{pct(r.rmse_pct)}</td>
            <td className={`tnum ${b("crps_pct", r.crps_pct)}`}>{pct(r.crps_pct)}</td>
            <td className={`tnum ${b("directional_acc", r.directional_acc, false)}`}>{prob(r.directional_acc)}</td>
            <td className="tnum">{prob(r.coverage80)}</td>
            <td className={`tnum ${b("auc_loss", r.auc_loss, false)}`}>{r.auc_loss?.toFixed(3) ?? "—"}</td>
            <td className={`tnum ${b("brier_loss", r.brier_loss)}`}>{r.brier_loss.toFixed(3)}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}
