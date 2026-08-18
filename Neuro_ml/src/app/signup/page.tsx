"use client";

import { useActionState } from "react";
import Link from "next/link";
import { signupAction } from "@/app/actions/auth";

export default function SignupPage() {
  const [state, action, pending] = useActionState(signupAction, undefined);
  return (
    <div className="auth-wrap">
      <form className="auth-card" action={action}>
        <div className="brand"><div className="logo">◆</div><div><h1>Create your account</h1><p className="sub">The first account becomes the admin</p></div></div>
        {state?.error && <div className="err">{state.error}</div>}
        <div className="field"><label className="fld">Name</label><input type="text" name="name" autoComplete="name" placeholder="Your name" /></div>
        <div className="field"><label className="fld">Email</label><input type="email" name="email" required autoComplete="email" placeholder="you@example.com" /></div>
        <div className="field"><label className="fld">Password</label><input type="password" name="password" required autoComplete="new-password" placeholder="At least 8 characters" /></div>
        <button className="btn block" type="submit" disabled={pending}>{pending ? "Creating…" : "Create account"}</button>
        <div className="auth-alt">Already have an account? <Link href="/login">Sign in</Link></div>
      </form>
    </div>
  );
}
