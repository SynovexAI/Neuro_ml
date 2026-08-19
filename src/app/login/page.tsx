"use client";

import { useActionState } from "react";
import Link from "next/link";
import { loginAction } from "@/app/actions/auth";

export default function LoginPage() {
  const [state, action, pending] = useActionState(loginAction, undefined);
  return (
    <div className="auth-wrap">
      <form className="auth-card" action={action}>
        <div className="brand"><div className="logo">◆</div><div><h1>AI Workbench</h1><p className="sub">Sign in to continue</p></div></div>
        {state?.error && <div className="err">{state.error}</div>}
        <div className="field"><label className="fld">Email</label><input type="email" name="email" required autoComplete="email" placeholder="you@example.com" /></div>
        <div className="field"><label className="fld">Password</label><input type="password" name="password" required autoComplete="current-password" placeholder="••••••••" /></div>
        <button className="btn block" type="submit" disabled={pending}>{pending ? "Signing in…" : "Sign in"}</button>
        <div className="auth-alt">No account? <Link href="/signup">Create one</Link></div>
      </form>
    </div>
  );
}
