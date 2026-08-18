"use client";

import { useEffect, useState } from "react";

type T = { id: number; msg: string; type: string };
type Confirm = { msg: string; resolve: (v: boolean) => void; confirmLabel?: string; danger?: boolean };
let counter = 0;

export default function Toaster() {
  const [toasts, setToasts] = useState<T[]>([]);
  const [cf, setCf] = useState<Confirm | null>(null);

  useEffect(() => {
    const onToast = (e: Event) => {
      const d = (e as CustomEvent).detail as { msg: string; type?: string; ms?: number };
      const id = ++counter;
      setToasts((t) => [...t, { id, msg: d.msg, type: d.type || "info" }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), d.ms || 3200);
    };
    const onConfirm = (e: Event) => setCf((e as CustomEvent).detail as Confirm);
    window.addEventListener("awb-toast", onToast);
    window.addEventListener("awb-confirm", onConfirm);
    return () => { window.removeEventListener("awb-toast", onToast); window.removeEventListener("awb-confirm", onConfirm); };
  }, []);

  const close = (v: boolean) => { cf?.resolve(v); setCf(null); };

  return (
    <>
      <div className="toaster">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span className="ti">{t.type === "success" ? "✓" : t.type === "error" ? "✕" : "•"}</span>
            <span>{t.msg}</span>
            <button className="tx" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>×</button>
          </div>
        ))}
      </div>
      {cf && (
        <div className="modal-wrap show" onClick={(e) => { if (e.target === e.currentTarget) close(false); }}>
          <div className="modal" style={{ maxWidth: 420 }}>
            <div className="mb">
              <div style={{ fontSize: 14, lineHeight: 1.5, marginBottom: 18 }}>{cf.msg}</div>
              <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
                <button className="btn ghost" onClick={() => close(false)}>Cancel</button>
                <button className={cf.danger ? "btn danger-solid" : "btn"} onClick={() => close(true)}>{cf.confirmLabel || "Confirm"}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
