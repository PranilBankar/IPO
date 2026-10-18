"use client";
import clsx from "clsx";
import { Moon, Plane, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const LINKS = [
  { href: "/", label: "Live IPOs" },
  { href: "/planner", label: "Planner" },
  { href: "/calculator", label: "Allotment calc" },
  { href: "/history", label: "History" },
  { href: "/model", label: "Model & research" },
  { href: "/sources", label: "Data" },
];

export function Nav() {
  const path = usePathname();
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const t = document.documentElement.getAttribute("data-theme");
    setDark(t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);
  const toggle = () => {
    const next = dark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("ipopilot-theme", next); } catch {}
    setDark(!dark);
  };
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-[var(--bg)]/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-2.5 md:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-white"><Plane size={15} /></span>
          IPOPilot <span className="text-ink-3">AI</span>
        </Link>
        <nav className="flex flex-1 gap-1 overflow-x-auto text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" || path.startsWith("/ipo") : path.startsWith(l.href);
            return (
              <Link key={l.href} href={l.href}
                className={clsx("whitespace-nowrap rounded-md px-2.5 py-1.5", active ? "bg-surface-2 font-semibold text-ink" : "text-ink-2 hover:text-ink")}>
                {l.label}
              </Link>
            );
          })}
        </nav>
        <button onClick={toggle} aria-label="Toggle dark mode" className="rounded-md p-1.5 text-ink-2 hover:bg-surface-2">
          {dark ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}
