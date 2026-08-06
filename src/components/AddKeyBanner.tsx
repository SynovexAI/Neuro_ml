"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

// Platform-wide nudge: if the signed-in user has NO usable LLM provider (neither their own key
// nor a shared/global one), prompt them to add their own key. Hidden on the pages where they'd
// add it, and while loading, so there's no flash.
export default function AddKeyBanner() {
  const path = usePathname() || "";
  const [need, setNeed] = useState(false);

  useEffect(() => {
    if (path.startsWith("/settings/keys") || path.startsWith("/admin/providers") || path === "/login" || path === "/signup") { setNeed(false); return; }
    let alive = true;
    fetch("/api/models")
      .then((r) => (r.ok ? r.json() : { providers: [] }))
      .then((j) => { if (alive) setNeed(!(Array.isArray(j.providers) && j.providers.length > 0)); })
      .catch(() => {});
    return () => { alive = false; };
  }, [path]);

  if (!need) return null;
  return (
    <div className="warnbar" style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
      <span>⚠ <b>Add an LLM provider to get started.</b> The labs need a key to run chat, RAG, and agents — add your own free key (Groq / Gemini / Cerebras) and it&apos;s used just for you.</span>
      <Link href="/settings/keys" className="btn sm" style={{ marginLeft: "auto", whiteSpace: "nowrap" }}>Add your key →</Link>
    </div>
  );
}
