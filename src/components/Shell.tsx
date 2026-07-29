"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { logoutAction } from "@/app/actions/auth";

type Role = "admin" | "student";
export type ShellUser = { name?: string | null; email: string; role: Role };

const LABS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/labs/prompting", label: "Prompting Lab" },
  { href: "/labs/rag", label: "RAG Lab" },
  { href: "/labs/agent", label: "Agent Lab" },
  { href: "/labs/ml", label: "ML Lab" },
  { href: "/labs/dl", label: "DL Lab" },
  { href: "/labs/etl", label: "ETL Lab" },
];
const STUDIO = [
  { href: "/compose", label: "Compose" },
  { href: "/templates", label: "Templates" },
];

export default function Shell({ user, title, children }: { user: ShellUser; title: string; children: React.ReactNode }) {
  const path = usePathname();
  const on = (href: string) => path === href || (href !== "/dashboard" && path.startsWith(href));
  const initial = (user.name || user.email).charAt(0).toUpperCase();
  return (
    <div className="app">
      <aside className="side">
        <div className="brand"><div className="logo">◆</div><div><b>AI Workbench</b><small>build · not read</small></div></div>
        <div className="nav-label">Labs</div>
        <nav className="nav">{LABS.map((l) => <Link key={l.href} href={l.href} className={on(l.href) ? "on" : ""}>{l.label}</Link>)}</nav>
        <div className="nav-label">Studio</div>
        <nav className="nav">{STUDIO.map((l) => <Link key={l.href} href={l.href} className={on(l.href) ? "on" : ""}>{l.label}</Link>)}</nav>
        {user.role === "admin" && (<>
          <div className="nav-label">Admin</div>
          <nav className="nav"><Link href="/admin" className={on("/admin") ? "on" : ""}>Admin panel</Link></nav>
        </>)}
        <div className="foot">{user.role === "admin" ? "admin" : "student"} · {user.email}</div>
      </aside>
      <div className="main">
        <header className="top">
          <h1>{title}</h1>
          <div className="spacer" />
          <span className={`badge ${user.role === "admin" ? "accent" : ""}`}>{user.role}</span>
          <form action={logoutAction}><button className="btn ghost sm" type="submit">Sign out</button></form>
          <div className="avatar">{initial}</div>
        </header>
        <div className="work">{children}</div>
      </div>
    </div>
  );
}
