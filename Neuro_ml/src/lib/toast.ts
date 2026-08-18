// Tiny client-side toast + confirm bus. Components call these; <Toaster/> (mounted
// in Shell) renders them — replaces the browser's native alert()/confirm().
export type ToastType = "success" | "error" | "info";

export function toast(msg: string, type: ToastType = "info", ms = 3200): void {
  if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("awb-toast", { detail: { msg, type, ms } }));
}

export function confirmDialog(msg: string, opts?: { confirmLabel?: string; danger?: boolean }): Promise<boolean> {
  return new Promise((resolve) => {
    if (typeof window === "undefined") { resolve(false); return; }
    window.dispatchEvent(new CustomEvent("awb-confirm", { detail: { msg, resolve, confirmLabel: opts?.confirmLabel, danger: opts?.danger } }));
  });
}
