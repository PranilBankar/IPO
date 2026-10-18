"use client";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import { fdate, pct, rupee, rupeeShort, times } from "@/lib/api";

const tipStyle = {
  contentStyle: { background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12, color: "var(--text-primary)" },
  labelStyle: { color: "var(--text-secondary)", fontWeight: 600 },
  itemStyle: { color: "var(--text-primary)" },
  cursor: { stroke: "var(--mid)", strokeWidth: 1, fill: "var(--surface-2)", fillOpacity: 0.5 },
};
const axis = { stroke: "var(--border)", tickLine: false, axisLine: false } as const;

/** Predictive listing-return density with the 80% interval shaded. */
export function DistributionChart({ data, q10, q90, median, actual }: {
  data: { ret_pct: number; density: number }[]; q10: number; q90: number; median: number; actual?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <AreaChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="ret_pct" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => `${v.toFixed(0)}%`} {...axis} />
        <YAxis hide />
        <ReferenceArea x1={q10} x2={q90} fill="var(--series-1)" fillOpacity={0.08} />
        <ReferenceLine x={0} stroke="var(--mid)" strokeWidth={1.5} label={{ value: "issue price", position: "insideTopLeft", fill: "var(--text-muted)", fontSize: 10 }} />
        <ReferenceLine x={median} stroke="var(--series-1)" strokeDasharray="4 3" />
        {actual != null && <ReferenceLine x={actual} stroke="var(--series-2)" strokeWidth={2}
          label={{ value: `actual ${pct(actual, 1, true)}`, position: "top", fill: "var(--text-secondary)", fontSize: 11 }} />}
        <Tooltip {...tipStyle} labelFormatter={(v) => `Listing return ${pct(Number(v), 1, true)}`}
          formatter={(v: number) => [v.toFixed(3), "density"]} />
        <Area type="monotone" dataKey="density" stroke="var(--series-1)" strokeWidth={2} fill="var(--series-1)" fillOpacity={0.18} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Diverging bars: contribution of each information group to the predicted mean return. */
export function ExplainBars({ data }: { data: { feature: string; contribution_pct: number }[] }) {
  const rows = [...data].sort((a, b) => b.contribution_pct - a.contribution_pct);
  return (
    <ResponsiveContainer width="100%" height={Math.max(140, rows.length * 30)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }} barCategoryGap={6}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}pp`} {...axis} />
        <YAxis type="category" dataKey="feature" width={170} {...axis} tick={{ fontSize: 11, fill: "var(--text-secondary)" }} />
        <ReferenceLine x={0} stroke="var(--mid)" />
        <Tooltip {...tipStyle} formatter={(v: number) => [`${v > 0 ? "+" : ""}${v.toFixed(2)} pp`, "effect on mean return"]} />
        <Bar dataKey="contribution_pct" radius={4} isAnimationActive={false}>
          {rows.map((r) => <Cell key={r.feature} fill={r.contribution_pct >= 0 ? "var(--pos)" : "var(--neg)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** GMP % over time, one line per source (max 3 sources -> slots 1..3). */
export function GmpChart({ data }: { data: { date: string; source: string; gmp_pct: number | null }[] }) {
  const sources = Array.from(new Set(data.map((d) => d.source))).slice(0, 3);
  const byDate: Record<string, any> = {};
  data.forEach((d) => { if (d.gmp_pct != null) (byDate[d.date] ||= { date: d.date })[d.source] = d.gmp_pct; });
  const rows = Object.values(byDate).sort((a: any, b: any) => a.date.localeCompare(b.date));
  const colors = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="date" tickFormatter={fdate} {...axis} />
        <YAxis tickFormatter={(v) => `${v}%`} width={44} {...axis} />
        <ReferenceLine y={0} stroke="var(--mid)" />
        <Tooltip {...tipStyle} labelFormatter={fdate} formatter={(v: number, n) => [pct(v, 1, true), n]} />
        {sources.length > 1 && <Legend iconType="plainline" wrapperStyle={{ fontSize: 11, color: "var(--text-secondary)" }} />}
        {sources.map((s, i) => (
          <Line key={s} dataKey={s} type="monotone" stroke={colors[i]} strokeWidth={2} dot={{ r: 3, strokeWidth: 2, fill: "var(--surface-1)" }}
            connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Category subscription (times) — a single measure, single hue. */
export function SubscriptionBars({ snap }: { snap: { qib: number | null; nii: number | null; snii?: number | null; bnii?: number | null; rii: number | null; total: number | null } }) {
  const rows = [
    { k: "QIB", v: snap.qib }, { k: "NII", v: snap.nii }, { k: "sNII", v: snap.snii }, { k: "bNII", v: snap.bnii },
    { k: "Retail", v: snap.rii }, { k: "Total", v: snap.total },
  ].filter((r) => r.v != null);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={rows} margin={{ top: 16, right: 8, left: 0, bottom: 0 }} barCategoryGap="28%">
        <CartesianGrid vertical={false} />
        <XAxis dataKey="k" {...axis} />
        <YAxis width={40} tickFormatter={(v) => `${v}×`} {...axis} />
        <ReferenceLine y={1} stroke="var(--mid)" strokeDasharray="3 3" />
        <Tooltip {...tipStyle} formatter={(v: number) => [times(v), "subscribed"]} />
        <Bar dataKey="v" fill="var(--series-1)" radius={[4, 4, 0, 0]} isAnimationActive={false}
          label={{ position: "top", fontSize: 11, fill: "var(--text-secondary)", formatter: (v: number) => times(v) }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Simulated portfolio profit distribution (diverging by sign). */
export function ProfitHistogram({ bins }: { bins: { lo: number; hi: number; count: number }[] }) {
  const rows = bins.map((b) => ({ mid: (b.lo + b.hi) / 2, count: b.count, lo: b.lo, hi: b.hi }));
  const total = rows.reduce((s, r) => s + r.count, 0) || 1;
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barCategoryGap={1}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="mid" tickFormatter={(v) => rupeeShort(v)} {...axis} minTickGap={30} />
        <YAxis hide />
        <Tooltip {...tipStyle} labelFormatter={(_, p) => p?.[0] ? `${rupee(p[0].payload.lo)} to ${rupee(p[0].payload.hi)}` : ""}
          formatter={(v: number) => [`${((100 * v) / total).toFixed(1)}% of scenarios`, "probability"]} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {rows.map((r, i) => <Cell key={i} fill={Math.abs(r.mid) < 1 ? "var(--mid)" : r.mid > 0 ? "var(--pos)" : "var(--neg)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Capital blocked per day vs available capital. */
export function CapitalTimeline({ data, capital }: { data: { date: string; blocked: number }[]; capital: number }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="date" tickFormatter={fdate} {...axis} />
        <YAxis tickFormatter={(v) => rupeeShort(v)} width={64} {...axis} domain={[0, Math.max(capital, 1)]} />
        <ReferenceLine y={capital} stroke="var(--mid)" strokeDasharray="4 3" label={{ value: "your capital", position: "insideTopRight", fill: "var(--text-muted)", fontSize: 10 }} />
        <Tooltip {...tipStyle} labelFormatter={fdate} formatter={(v: number) => [rupee(v), "blocked"]} />
        <Bar dataKey="blocked" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={44} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/** Mean listing gain by year, mainboard vs SME (two series -> legend + direct identity). */
export function YearlyGains({ data }: { data: { year: number; board: string; mean_gain: number; n: number }[] }) {
  const by: Record<number, any> = {};
  data.forEach((d) => { (by[d.year] ||= { year: d.year })[d.board] = d.mean_gain; (by[d.year])[`${d.board}_n`] = d.n; });
  const rows = Object.values(by);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 0 }} barGap={2}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="year" {...axis} />
        <YAxis tickFormatter={(v) => `${v}%`} width={44} {...axis} />
        <ReferenceLine y={0} stroke="var(--mid)" />
        <Tooltip {...tipStyle} formatter={(v: number, n: string, p: any) => [`${pct(v, 1, true)} (n=${p.payload[`${n}_n`] ?? "?"})`, n === "mainboard" ? "Mainboard" : "SME"]} />
        <Legend formatter={(v) => (v === "mainboard" ? "Mainboard" : "SME")} wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="mainboard" fill="var(--series-1)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
        <Bar dataKey="sme" fill="var(--series-2)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
