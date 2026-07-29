import Link from "next/link";

export default function PendingPage() {
  return (
    <div className="auth-wrap">
      <div className="auth-card" style={{ textAlign: "center" }}>
        <div className="logo" style={{ margin: "0 auto 14px" }}>◆</div>
        <h1>Account created</h1>
        <p className="sub" style={{ margin: "8px 0 18px" }}>
          Your account is awaiting approval from an administrator. You&apos;ll be able to sign in once it&apos;s approved.
        </p>
        <Link className="btn ghost block" href="/login">Back to sign in</Link>
      </div>
    </div>
  );
}
