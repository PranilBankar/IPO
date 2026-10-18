// Thin typed client for the IPOPilot FastAPI backend. Requests go to /api/* which Next.js
// rewrites to BACKEND_URL (see next.config.ts), so the browser never needs CORS.

export type Snapshot = {
  source: string; as_of: string; final: boolean;
  total: number | null; qib: number | null; nii: number | null; snii: number | null; bnii: number | null;
  rii: number | null; emp: number | null; rii_applications: number | null;
};
export type Gmp = { date: string; source: string; gmp_rs: number | null; gmp_pct: number | null; fetched_at?: string };
export type Prediction = {
  mean_pct: number; median_pct: number; q05_pct: number; q10_pct: number; q25_pct: number; q75_pct: number;
  q90_pct: number; q95_pct: number; p_loss: number; model_version: string;
};
export type Allotment = {
  category: string; unit_lots: number; min_amount: number | null; p_mean: number; p_q10: number; p_q90: number;
  rii_final_forecast: number[]; multi: { n: number; p_at_least_one: number; naive: number }[]; L_prior: [number, number] | null;
};
export type IpoRow = {
  slug: string; name: string; symbol: string | null; board: "mainboard" | "sme"; stage: string;
  open_date: string | null; close_date: string | null; listing_date: string | null;
  price_low: number | null; price_high: number | null; price: number | null; lot_size: number | null;
  issue_size_cr: number | null; anchor_pct: number | null; sector: string | null; registrar: string | null;
  is_demo: boolean; min_investment: number | null; subscription: Snapshot | null; demand_stage: "prior" | "live" | "final";
  bid_elapsed: number; gmp_latest: Gmp[]; gmp_consensus_pct: number | null; prediction: Prediction;
  allotment: Allotment;
  min_application_ev: { expected_profit: number; amount: number; ev_per_rupee: number; naive_expected_profit: number;
    winners_curse: number; return_given_allotted_pct: number } | null;
  news_count: number; news_sentiment: number | null; updated_at: string;
};
export type IpoDetail = IpoRow & {
  gmp_history: Gmp[];
  news: { title: string; url: string; publisher: string | null; published_at: string | null; p_pos: number | null;
    p_neg: number | null; p_neu: number | null; topic: string | null }[];
  explanation: { feature: string; contribution_pct: number }[];
  features: Record<string, number | string | null>;
  distribution: { ret_pct: number; density: number }[];
  subscription_history: Snapshot[];
  options: { category: string; lots: number; unit_lots: number; amount: number; label: string }[];
  actual_listing_gain_pct?: number;
};

export type PlanApp = { applicant_id: string; applicant: string; category: string; category_label: string; option: string;
  lots: number; shares: number; amount: number; p_allot: number; expected_profit: number; downside_cvar10: number };
export type PlanRow = { slug: string; name: string; board: string; price: number; lot: number; close_date: string;
  unblock_date: string; is_demo: boolean; model_mean_return_pct: number; model_p_loss: number; applications: PlanApp[];
  n_applications: number; capital_blocked: number; p_any_allotment: number; expected_profit: number;
  breakeven_shift_pct: number | null; why: string };
export type Simulation = { expected_profit: number; median_profit?: number; std_profit?: number; p_loss: number;
  p_profit?: number; p_any_allotment: number; expected_allotments?: number; var5?: number; cvar5?: number; q95_profit?: number;
  histogram: { lo: number; hi: number; count: number }[]; n_applications: number };
export type Plan = { objective: string; objective_label: string; risk: string; capital: number; peak_capital_blocked: number;
  min_free_cash: number; total_applications: number; selected: PlanRow[]; excluded: { slug: string; name: string; reason: string }[];
  simulation: Simulation; capital_timeline: { date: string; blocked: number; free: number }[]; assumptions: string[] };
export type PlanResponse = (Plan | { comparison: ComparisonRow[]; plans: Record<string, Plan> }) & {
  warnings: string[]; model_version: string; generated_at: string; n_candidates: number; disclaimer: string; 
};
export type ComparisonRow = { objective: string; label: string; expected_profit: number; p_any_allotment: number; p_loss: number;
  cvar5: number; peak_capital: number; applications: number; return_on_peak_capital: number };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j); } catch {}
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  get: <T,>(p: string) => req<T>(p),
  post: <T,>(p: string, body: unknown) => req<T>(p, { method: "POST", body: JSON.stringify(body) }),
};
export const fetcher = (p: string) => req<any>(p);

// ---------------------------------------------------------------- formatters
const inr0 = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
export const rupee = (v: number | null | undefined) => (v == null ? "—" : `₹${inr0.format(Math.round(v))}`);
export const rupeeShort = (v: number | null | undefined) => {
  if (v == null) return "—";
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return rupee(v);
};
export const pct = (v: number | null | undefined, d = 1, sign = false) =>
  v == null || Number.isNaN(v) ? "—" : `${sign && v > 0 ? "+" : ""}${v.toFixed(d)}%`;
export const prob = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(100 * v).toFixed(d)}%`);
export const times = (v: number | null | undefined) => (v == null ? "—" : `${v >= 100 ? v.toFixed(0) : v.toFixed(2)}×`);
export const fdate = (s: string | null | undefined) =>
  s ? new Date(s + (s.length === 10 ? "T00:00:00" : "")).toLocaleDateString("en-IN", { day: "numeric", month: "short" }) : "—";
export const fdatetime = (s: string | null | undefined) =>
  s ? new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z").toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "—";
export const STAGE_LABEL: Record<string, string> = {
  open: "Open", closed: "Closed", upcoming: "Upcoming", rhp_filed: "RHP filed", allotment: "Allotment", pre_listing: "Pre-listing",
};
