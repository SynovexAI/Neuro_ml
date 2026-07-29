"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type Row = { id: string; email: string; name: string | null; role: "admin" | "student"; status: "pending" | "active" | "suspended" };

export default function UsersManager({ initial, meId }: { initial: Row[]; meId: string }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);

  async function patch(id: string, body: object) {
    setBusy(id);
    await fetch(`/api/admin/users/${id}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    router.refresh();
    setBusy(null);
  }

  const badge = (s: string) => s === "active" ? "badge good" : s === "pending" ? "badge warn" : "badge";

  return (
    <div className="card">
      <div className="card-h"><span className="t">Users</span><span className="mono r">{initial.length}</span></div>
      <div className="card-b" style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead><tr><th>User</th><th>Role</th><th>Status</th><th style={{ textAlign: "right" }}>Actions</th></tr></thead>
          <tbody>
            {initial.map((u) => (
              <tr key={u.id}>
                <td><b>{u.name || "—"}</b><div className="note">{u.email}{u.id === meId ? " · you" : ""}</div></td>
                <td>
                  <select value={u.role} disabled={u.id === meId || busy === u.id} onChange={(e) => patch(u.id, { role: e.target.value })} style={{ width: 110 }}>
                    <option value="student">student</option><option value="admin">admin</option>
                  </select>
                </td>
                <td><span className={badge(u.status)}>{u.status}</span></td>
                <td style={{ textAlign: "right" }}>
                  <div className="row" style={{ justifyContent: "flex-end" }}>
                    {u.status === "pending" && <button className="btn sm" disabled={busy === u.id} onClick={() => patch(u.id, { status: "active" })}>Approve</button>}
                    {u.status === "active" && u.id !== meId && <button className="btn ghost sm" disabled={busy === u.id} onClick={() => patch(u.id, { status: "suspended" })}>Suspend</button>}
                    {u.status === "suspended" && <button className="btn ghost sm" disabled={busy === u.id} onClick={() => patch(u.id, { status: "active" })}>Reactivate</button>}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
