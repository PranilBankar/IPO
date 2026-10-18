"use client";
import { useEffect, useState } from "react";
import { Card, Field, Notice, Seg, Stat } from "@/components/ui";
import { api, prob } from "@/lib/api";

type Res = { method: string; p_mean: number; p_q10: number; p_q90: number; table: { n: number; p_at_least_one: number; naive: number }[] };

export default function Calculator() {
  const [board, setBoard] = useState<"mainboard" | "sme">("mainboard");
  const [cat, setCat] = useState<"RII" | "SNII" | "BNII">("RII");
  const [T, setT] = useState(10);
  const [n, setN] = useState(5);
  const [apps, setApps] = useState("");
  const [offered, setOffered] = useState("");
  const [unit, setUnit] = useState("");
  const [res, setRes] = useState<Res | null>(null);

  useEffect(() => {
    const t = setTimeout(async () => {
      const body: any = { board, category: cat, subscription_times: T, applicants: n };
      if (apps && offered && unit) Object.assign(body, { applications: +apps, offered_shares: +offered, unit_shares: +unit });
      try { setRes(await api.post<Res>("/api/allotment/calc", body)); } catch { setRes(null); }
    }, 250);
    return () => clearTimeout(t);
  }, [board, cat, T, n, apps, offered, unit]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Allotment probability calculator</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">
          SEBI lottery maths: in an oversubscribed category every winner receives one minimum application, so
          P(allotment) = min(1, L / T), where T is the subscription multiple and L the average application size in minimum-application units.
          If you know the actual number of applications (from the registrar’s basis of allotment), enter it for an exact answer.
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-[340px_1fr]">
        <Card title="Inputs" className="h-fit">
          <div className="space-y-4">
            <Field label="Board"><Seg value={board} onChange={setBoard} options={[{ value: "mainboard", label: "Mainboard" }, { value: "sme", label: "SME" }]} /></Field>
            <Field label="Category"><Seg value={cat} onChange={setCat} options={[{ value: "RII", label: "Retail/Individual" }, { value: "SNII", label: "sNII" }, { value: "BNII", label: "bNII" }]} /></Field>
            <Field label={`Category subscription: ${T}×`}>
              <input type="range" min={0.5} max={500} step={0.5} value={T} onChange={(e) => setT(+e.target.value)} className="w-full accent-[var(--accent)]" />
            </Field>
            <Field label={`Distinct applicants (PANs): ${n}`}>
              <input type="range" min={1} max={10} value={n} onChange={(e) => setN(+e.target.value)} className="w-full accent-[var(--accent)]" />
            </Field>
            <details className="text-sm">
              <summary className="cursor-pointer text-ink-2">Exact mode (optional)</summary>
              <div className="mt-2 space-y-2">
                <Field label="Valid applications in category"><input className="input" value={apps} onChange={(e) => setApps(e.target.value)} placeholder="e.g. 1000000" /></Field>
                <Field label="Shares offered in category"><input className="input" value={offered} onChange={(e) => setOffered(e.target.value)} placeholder="e.g. 3000000" /></Field>
                <Field label="Shares per minimum application"><input className="input" value={unit} onChange={(e) => setUnit(e.target.value)} placeholder="e.g. 30" /></Field>
              </div>
            </details>
          </div>
        </Card>
        <div className="space-y-4">
          {res && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <Stat label="P(allotment), one application" value={prob(res.p_mean)} hint={res.method} />
                <Stat label="Plausible range (10–90%)" value={`${prob(res.p_q10, 0)}–${prob(res.p_q90, 0)}`} hint="uncertainty in L" />
                <Stat label={`P(≥1) with ${n} PANs`} value={prob(res.table[res.table.length - 1]?.p_at_least_one)} hint={`naive ${prob(res.table[res.table.length - 1]?.naive)}`} />
              </div>
              <Card title="Applicants vs odds">
                <table className="data">
                  <thead><tr><th>PANs</th><th>P(≥1 allotment)</th><th>Naive 1−(1−p)ⁿ</th><th>Visual</th></tr></thead>
                  <tbody>{res.table.map((r) => (
                    <tr key={r.n}><td className="tnum">{r.n}</td><td className="tnum font-semibold">{prob(r.p_at_least_one)}</td><td className="tnum text-ink-3">{prob(r.naive)}</td>
                      <td className="w-1/2"><div className="h-2 rounded-full bg-surface-2"><div className="h-2 rounded-full bg-accent" style={{ width: `${100 * r.p_at_least_one}%` }} /></div></td></tr>
                  ))}</tbody>
                </table>
              </Card>
            </>
          )}
          <Notice kind="warning">Multiple demat accounts under the <b>same PAN</b> are one application — duplicates are rejected, not extra lottery tickets.</Notice>
          <Notice>From 1 July 2025 SME IPOs require a minimum application of two lots (above ₹2 lakh) and individual investors are allotted that minimum by lottery, so L = 1 exactly for SME individuals.</Notice>
        </div>
      </div>
    </div>
  );
}
