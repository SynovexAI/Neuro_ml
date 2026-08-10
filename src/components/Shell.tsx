"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import AddKeyBanner from "@/components/AddKeyBanner";
import { useEffect, useState } from "react";
import { logoutAction } from "@/app/actions/auth";
import Toaster from "./Toaster";
import ThemeToggle from "./ThemeToggle";

type Role = "admin" | "student";
export type ShellUser = { name?: string | null; email: string; role: Role };

type Zone = { id: string; label: string; desc: string; icon: string; home: string; adminOnly?: boolean; items: { href: string; label: string }[] };

const ZONES: Zone[] = [
  {
    id: "studio", label: "Studio", desc: "Dashboard, build, knowledge", icon: "◈", home: "/dashboard",
    items: [
      { href: "/dashboard", label: "Dashboard" },
      { href: "/compose", label: "Compose" },
      { href: "/kb", label: "Knowledge bases" },
      { href: "/studio/mcp", label: "MCP servers" },
      { href: "/admin/providers", label: "Providers & models" },
      { href: "/settings/keys", label: "My API keys" },
      { href: "/projects", label: "My Projects" },
      { href: "/templates", label: "Templates" },
    ],
  },
  {
    id: "labs", label: "Labs", desc: "Prompting, RAG, agents, ML", icon: "⚗", home: "/labs/prompting",
    items: [
      { href: "/labs/prompting", label: "Prompting Lab" },
      { href: "/labs/rag", label: "RAG Lab" },
      { href: "/labs/agent", label: "Agent Lab" },
      { href: "/labs/ml", label: "ML Lab" },
      { href: "/labs/dl", label: "DL Lab" },
      { href: "/labs/etl", label: "ETL Lab" },
    ],
  },
  {
    id: "workroom", label: "Workroom", desc: "Use & deploy your agents", icon: "◐", home: "/workroom",
    items: [
      { href: "/workroom", label: "Published agents" },
      { href: "/workroom/channels", label: "Channels & deploy" },
    ],
  },
  {
    id: "control", label: "Control Room", desc: "Users, usage, monitoring", icon: "▤", home: "/admin", adminOnly: true,
    items: [
      { href: "/admin", label: "Overview" },
      { href: "/admin/users", label: "Users" },
      { href: "/admin/usage", label: "Usage & Monitoring" },
      { href: "/admin/agents", label: "Agent analytics" },
      { href: "/admin/analytics", label: "Analytics" },
      { href: "/admin/storage", label: "Storage" },
    ],
  },
];

function zoneOf(path: string): string {
  if (path === "/studio/mcp" || path === "/admin/providers") return "studio"; // build tools live in Studio
  if (path.startsWith("/workroom")) return "workroom";
  if (path.startsWith("/labs")) return "labs";
  if (path.startsWith("/admin")) return "control";
  return "studio";
}

export default function Shell({ user, title, children }: { user: ShellUser; title: string; children: React.ReactNode }) {
  const path = usePathname();
  const on = (href: string) => path === href || (href !== "/dashboard" && href !== "/admin" && href !== "/workroom" && path.startsWith(href));
  const initial = (user.name || user.email).charAt(0).toUpperCase();

  const [focus, setFocus] = useState(false);
  const [menu, setMenu] = useState(false);

  useEffect(() => {
    setFocus(localStorage.getItem("awb_focus") === "1");
    
    // Cursor glow tracking removed for a cleaner look

  }, []);

  const toggleFocus = () => setFocus((f) => { const n = !f; localStorage.setItem("awb_focus", n ? "1" : "0"); return n; });

  const zones = ZONES.filter((z) => !z.adminOnly || user.role === "admin");
  const zone = ZONES.find((z) => z.id === zoneOf(path)) || ZONES[0];

  return (
    <div className={`app${focus ? " focus" : ""}`}>

      <aside className="side">
        <div className="brand">
          <div className="logo">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L22 12L12 22L2 12L12 2Z" fill="currentColor"/>
            </svg>
          </div>
          <div><b>AI Workbench</b><small>BUILD &bull; NOT READ</small></div>
        </div>

        <div className="zone-switch">
          <button className="zone-btn" onClick={() => setMenu((m) => !m)} aria-expanded={menu}>
            <span className="zi">{zone.icon}</span>
            <span className="zl"><b>{zone.label}</b><small>{zone.desc}</small></span>
            <span className="chev">{menu ? "▲" : "▼"}</span>
          </button>
          {menu && (<>
            <div className="zone-backdrop" onClick={() => setMenu(false)} />
            <div className="zone-menu">
              {zones.map((z) => (
                <Link key={z.id} href={z.home} className={`zone-item${z.id === zone.id ? " on" : ""}`} onClick={() => setMenu(false)}>
                  <span className="zi">{z.icon}</span>
                  <span className="zl"><b>{z.label}</b><small>{z.desc}</small></span>
                  {z.id === zone.id && <span className="ck">✓</span>}
                </Link>
              ))}
            </div>
          </>)}
        </div>

        <nav className="nav">{zone.items.map((l) => <Link key={l.href} href={l.href} className={on(l.href) ? "on" : ""}>{l.label}</Link>)}</nav>

        <div className="side-account">
          <div className="avatar">{initial}</div>
          <div className="acct-info"><b>{user.name || user.email.split("@")[0]}</b><small>{user.role}</small></div>
          <form action={logoutAction}><button className="iconbtn" type="submit" title="Sign out" aria-label="Sign out">⎋</button></form>
        </div>
      </aside>

      <div className="main">
        <header className="top">
          <button className="iconbtn" onClick={toggleFocus} title={focus ? "Show sidebar" : "Focus mode — full-screen view"} aria-label="Toggle focus mode">{focus ? "☰" : "⛶"}</button>
          <div>
            <p className="eyebrow">Futuristic AI lab</p>
            <h1>{title}</h1>
          </div>
          <div className="spacer" />
          <span className="badge accent">{zone.label}</span>
          <ThemeToggle />
        </header>
        <div className="work"><AddKeyBanner />{children}</div>
      </div>
      <Toaster />
    </div>
  );
}
