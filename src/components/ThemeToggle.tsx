"use client";

import { useEffect, useState } from "react";

// Light/dark toggle. First visit follows the OS (via the CSS media query);
// once toggled, the explicit choice is stored and wins.
export default function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("awb_theme");
    const sys = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    setTheme(stored === "light" || stored === "dark" ? stored : sys);
    setReady(true);
  }, []);

  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("awb_theme", next);
  }

  if (!ready) return <span className="iconbtn" aria-hidden style={{ opacity: 0 }} />;
  return (
    <button className="iconbtn" onClick={toggle} title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} aria-label="Toggle color theme">
      {theme === "dark" ? "☀" : "☾"}
    </button>
  );
}
