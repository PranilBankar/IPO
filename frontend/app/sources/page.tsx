"use client";
import { ExternalLink } from "lucide-react";
import useSWR from "swr";
import { Card, ErrorBox, Skeleton, Stat, StatusPill } from "@/components/ui";
import { fdatetime, fetcher } from "@/lib/api";

export default function Sources() {
  const { data, error } = useSWR("/api/sources", fetcher, { refreshInterval: 60_000 });
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Skeleton h={400} />;
  const c = data.counts;
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Data sources & pipeline</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-2">Official exchange data is the reference; GMP and news are treated as unofficial signals. Every observation is time-stamped and append-only, so models are trained only on what was knowable at the time.</p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="IPOs in universe" value={c.ipos.toLocaleString("en-IN")} />
        <Stat label="With listing outcome" value={c.listed_with_gain.toLocaleString("en-IN")} />
        <Stat label="Subscription snapshots" value={c.subscription_snapshots.toLocaleString("en-IN")} />
        <Stat label="GMP observations" value={c.gmp_observations.toLocaleString("en-IN")} />
        <Stat label="IPOs with GMP history" value={c.ipos_with_gmp.toLocaleString("en-IN")} />
      </div>
      <Card title="Sources">
        <table className="data">
          <thead><tr><th>Source</th><th>Type</th><th>Used for</th></tr></thead>
          <tbody>{data.sources.map((s: any) => (
            <tr key={s.name}><td><a href={s.url} target="_blank" rel="noreferrer" className="font-medium hover:underline">{s.name} <ExternalLink size={11} className="inline" /></a></td>
              <td className="text-ink-2">{s.kind}</td><td className="text-ink-2">{s.fields}</td></tr>
          ))}</tbody>
        </table>
      </Card>
      <Card title="Recent ingestion runs">
        <div className="-mx-4 overflow-x-auto md:-mx-5">
          <table className="data min-w-[700px]">
            <thead><tr><th className="pl-5">Job</th><th>Started</th><th>Status</th><th>Stats</th></tr></thead>
            <tbody>{data.runs.map((r: any, i: number) => (
              <tr key={i}><td className="pl-5 font-mono text-xs">{r.job}</td><td className="tnum">{fdatetime(r.started_at)}</td>
                <td>{r.ok ? <StatusPill status="good">ok</StatusPill> : r.finished_at ? <StatusPill status="critical">failed</StatusPill> : <StatusPill status="info">running</StatusPill>}</td>
                <td className="font-mono text-[0.7rem] text-ink-3">{r.error ?? JSON.stringify(r.stats ?? {})}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
