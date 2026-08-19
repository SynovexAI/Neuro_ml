"use client";

import { useActionState, useState } from "react";
import Link from "next/link";
import { signupAction } from "@/app/actions/auth";

export default function SignupPage() {
  const [state, action, pending] = useActionState(signupAction, undefined);
  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="auth-wrap">
      <form className="auth-card" action={action}>
        <div className="brand"><div className="logo">◆</div><div><h1>Create your account</h1><p className="sub">The first account becomes the admin</p></div></div>
        {state?.error && <div className="err">{state.error}</div>}
        <div className="field"><label className="fld">Name</label><input type="text" name="name" autoComplete="name" placeholder="Your name" /></div>
        <div className="field"><label className="fld">Email</label><input type="email" name="email" required autoComplete="email" placeholder="you@example.com" /></div>
        <div className="field">
          <label className="fld">Password</label>
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              required
              autoComplete="new-password"
              placeholder="At least 8 characters"
              style={{ paddingRight: 38 }}
            />
            <button
              type="button"
              onClick={() => setShowPassword((prev) => !prev)}
              style={{
                position: "absolute",
                right: 10,
                background: "none",
                border: "none",
                color: "var(--muted)",
                cursor: "pointer",
                padding: 4,
                display: "grid",
                placeItems: "center",
                borderRadius: 4,
                transition: "color 0.15s ease",
              }}
              title={showPassword ? "Hide password" : "Show password"}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24" />
                  <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68" />
                  <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61" />
                  <line x1="2" y1="2" x2="22" y2="22" />
                </svg>
              ) : (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
              )}
            </button>
          </div>
        </div>
        <button className="btn block" type="submit" disabled={pending}>{pending ? "Creating…" : "Create account"}</button>
        <div className="auth-alt">Already have an account? <Link href="/login">Sign in</Link></div>
      </form>
    </div>
  );
}
