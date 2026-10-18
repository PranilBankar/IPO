"use client";
import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import type { ReactNode } from "react";

export function Card({ title, subtitle, right, children, className }: {
  title?: ReactNode; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={clsx("card p-4 md:p-5", className)}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-[0.95rem] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }: { label: ReactNode; value: ReactNode; hint?: ReactNode; tone?: "pos" | "neg" }) {
  return (
    <div className="card px-4 py-3">
      <div className="text-xs font-medium text-ink-3">{label}</div>
      <div className={clsx("tnum mt-1 text-2xl font-semibold tracking-tight",
        tone === "pos" && "text-pos", tone === "neg" && "text-neg")}>{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

export function Badge({ children, kind = "neutral" }: { children: ReactNode; kind?: "neutral" | "sme" | "main" | "demo" | "accent" }) {
  return (
    <span className={clsx("inline-flex items-center rounded-md px-1.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide",
      kind === "neutral" && "bg-surface-2 text-ink-2",
      kind === "main" && "bg-surface-2 text-ink",
      kind === "sme" && "border border-line text-ink-2",
      kind === "demo" && "border border-dashed border-[var(--serious)] text-ink-2",
      kind === "accent" && "bg-accent text-white")}>{children}</span>
  );
}

const STATUS = {
  good: { icon: CheckCircle2, color: "var(--good)" },
  warning: { icon: AlertTriangle, color: "var(--warning)" },
  serious: { icon: AlertTriangle, color: "var(--serious)" },
  critical: { icon: XCircle, color: "var(--critical)" },
  info: { icon: Info, color: "var(--accent)" },
} as const;

/** Status colour always ships with an icon + text label (never colour alone). */
export function StatusPill({ status, children }: { status: keyof typeof STATUS; children: ReactNode }) {
  const S = STATUS[status];
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-ink-2">
      <S.icon size={14} style={{ color: S.color }} aria-hidden />
      {children}
    </span>
  );
}

export function Notice({ kind = "info", children }: { kind?: "info" | "warning" | "critical"; children: ReactNode }) {
  const S = STATUS[kind];
  return (
    <div className="card flex gap-2.5 px-3.5 py-2.5 text-sm text-ink-2">
      <S.icon size={17} style={{ color: S.color }} className="mt-0.5 shrink-0" aria-hidden />
      <div>{children}</div>
    </div>
  );
}

export function Skeleton({ h = 120 }: { h?: number }) {
  return <div className="card animate-pulse" style={{ height: h }} />;
}

export function ErrorBox({ error }: { error: unknown }) {
  return (
    <Notice kind="critical">
      Could not reach the IPOPilot API — is the backend running on port 8000?{" "}
      <span className="text-ink-3">({String((error as Error)?.message ?? error)})</span>
    </Notice>
  );
}

export function Seg<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[];
}) {
  return (
    <div className="seg" role="group">
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[0.7rem] text-ink-3">{hint}</span>}
    </label>
  );
}

/** A small range bar: q10..q90 band with a median tick, on a fixed -50%..+100% scale. */
export function RangeBar({ lo, mid, hi, min = -40, max = 100 }: { lo: number; mid: number; hi: number; min?: number; max?: number }) {
  const x = (v: number) => `${((Math.max(min, Math.min(max, v)) - min) / (max - min)) * 100}%`;
  return (
    <div className="relative h-2.5 w-full min-w-[90px] rounded-full bg-surface-2" aria-label={`80% range ${lo.toFixed(0)}% to ${hi.toFixed(0)}%`}>
      <div className="absolute top-0 h-full w-px bg-[var(--mid)]" style={{ left: x(0) }} />
      <div className="absolute top-0 h-full rounded-full opacity-35"
        style={{ left: x(lo), width: `calc(${x(hi)} - ${x(lo)})`, background: mid >= 0 ? "var(--pos)" : "var(--neg)" }} />
      <div className="absolute -top-0.5 h-3.5 w-[3px] rounded-full" style={{ left: x(mid), background: mid >= 0 ? "var(--pos)" : "var(--neg)" }} />
    </div>
  );
}
