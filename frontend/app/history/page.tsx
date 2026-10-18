"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { YearlyGains } from "@/components/charts";
import { Badge, Card, ErrorBox, Seg, Skeleton } from "@/components/ui";
import { fdate, fetcher, pct, times } from "@/lib/api";

export default function History() {
  const [q, setQ] = useState("");
  const [board, setBoard] = useState<"" | "mainboard" | "sme">("");
  const [page, setPage] = useState(0);
  const stats = useSWR("/api/history/stats", fetcher);
  const list = useSWR(`/api/history?limit=50&offset=${page * 50}${q ? `&q=${encodeURIComponent(q)}` : ""}${board ? `&board=${board}` : ""}`, fetcher, { keepPreviousData: true });
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">IPO history</h1>
        <p className="mt-1 text-sm text-ink-2">The training universe: every listed mainboard and SME IPO with its actual listing-day gain.</p>
      </div>
      <Card title="Average listing-day gain by year" subtitle="Mean of listing gains; hover for the number of issues.">
        {stats.data ? <YearlyGains data={stats.data.by_year} /> : <Skeleton h={260} />}
      </Card>
      <Card title="Listed issues" right={
        <div className="flex gap-2">
          <input className="input !w-56" placeholder="Search name or symbol" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} />
          <Seg value={board} onChange={(v) => { setBoard(v); setPage(0); }} options={[{ value: "", label: "All" }, { value: "mainboard", label: "Main" }, { value: "sme", label: "SME" }]} />
        </div>}>
        {list.error && <ErrorBox error={list.error} />}
        <div className="-mx-4 overflow-x-auto md:-mx-5">
          <table className="data min-w-[760px]">
            <thead><tr><th className="pl-5">Issue</th><th>Closed</th><th>Listed</th><th>Price</th><th>Size</th><th>Subscription</th><th>GMP data</th><th>Listing gain</th></tr></thead>
            <tbody>
              {(list.data?.items ?? []).map((r: any) => (
                <tr key={r.slug}>
                  <td className="pl-5"><Link className="font-medium hover:underline" href={`/ipo/${r.slug}`}>{r.name}</Link> <Badge kind={r.board === "sme" ? "sme" : "main"}>{r.board === "sme" ? "SME" : "Main"}</Badge></td>
                  <td className="tnum">{fdate(r.close_date)} {r.close_date?.slice(0, 4)}</td>
                  <td className="tnum">{fdate(r.listing_date)}</td>
                  <td className="tnum">{r.price ? `₹${r.price}` : "—"}</td>
                  <td className="tnum">{r.issue_size_cr ? `₹${r.issue_size_cr.toFixed(0)} Cr` : "—"}</td>
                  <td className="tnum">{times(r.sub_total)}</td>
                  <td>{r.has_gmp ? "yes" : <span className="text-ink-3">—</span>}</td>
                  <td className={`tnum font-semibold ${r.listing_gain_pct >= 0 ? "text-pos" : "text-neg"}`}>{pct(r.listing_gain_pct, 1, true)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex items-center justify-between text-sm text-ink-2">
          <span>{list.data?.total ?? 0} issues</span>
          <div className="flex gap-2">
            <button className="btn btn-ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
            <button className="btn btn-ghost" disabled={(page + 1) * 50 >= (list.data?.total ?? 0)} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        </div>
      </Card>
    </div>
  );
}
