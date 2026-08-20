"use client";
import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";

// Beacons each page navigation to /api/track for in-app analytics. Fire-and-forget,
// de-duped per path so a single view isn't counted repeatedly.
export default function TrackPageView() {
  const path = usePathname() || "/";
  const last = useRef<string>("");
  useEffect(() => {
    if (path === last.current) return;
    last.current = path;
    try {
      const body = JSON.stringify({ path });
      if (navigator.sendBeacon) navigator.sendBeacon("/api/track", new Blob([body], { type: "application/json" }));
      else fetch("/api/track", { method: "POST", headers: { "content-type": "application/json" }, body, keepalive: true });
    } catch { /* ignore */ }
  }, [path]);
  return null;
}
