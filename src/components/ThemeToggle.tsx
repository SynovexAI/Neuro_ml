"use client";

import { useEffect, useState } from "react";

type ThemeMode = "light" | "dark" | "cosmic";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<ThemeMode>("light");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("awb_theme") as ThemeMode | null;
    const sys = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const next = stored === "light" || stored === "dark" || stored === "cosmic" ? stored : sys;
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    setReady(true);
  }, []);

  function toggle() {
    const next: ThemeMode = theme === "light" ? "dark" : theme === "dark" ? "cosmic" : "light";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("awb_theme", next);
  }

  if (!ready) return <span className="iconbtn" aria-hidden style={{ opacity: 0 }} />;
  return (
    <button className="iconbtn" onClick={toggle} title={`Switch to ${theme === "light" ? "dark" : theme === "dark" ? "cosmic" : "light"} mode`} aria-label="Toggle color theme">
      {theme === "light" ? "☾" : theme === "dark" ? "✦" : "☀"}
    </button>
  );
}
