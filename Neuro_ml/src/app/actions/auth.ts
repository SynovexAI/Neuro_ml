"use server";

import { redirect } from "next/navigation";
import { eq } from "drizzle-orm";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import { hashPassword, verifyPassword, createSession, destroySession, uid, userCount, getSessionUser } from "@/lib/auth";
import { audit } from "@/lib/monitor";

type State = { error?: string } | undefined;

export async function signupAction(_prev: State, form: FormData): Promise<State> {
  const name = String(form.get("name") || "").trim();
  const email = String(form.get("email") || "").trim().toLowerCase();
  const password = String(form.get("password") || "");
  if (!email || !password) return { error: "Email and password are required." };
  if (password.length < 8) return { error: "Password must be at least 8 characters." };

  const existing = await db.select({ id: users.id }).from(users).where(eq(users.email, email)).limit(1);
  if (existing.length) return { error: "An account with that email already exists." };

  const first = (await userCount()) === 0;
  const id = uid();
  await db.insert(users).values({
    id, email, name: name || null,
    passwordHash: hashPassword(password),
    role: first ? "admin" : "student",
    status: first ? "active" : "pending",
  });
  await audit("signup", id, { email, role: first ? "admin" : "student" }).catch(() => {});

  if (first) { await createSession(id); redirect("/dashboard"); }
  redirect("/pending");
}

export async function loginAction(_prev: State, form: FormData): Promise<State> {
  const email = String(form.get("email") || "").trim().toLowerCase();
  const password = String(form.get("password") || "");
  const rows = await db.select().from(users).where(eq(users.email, email)).limit(1);
  const u = rows[0];
  if (!u || !verifyPassword(password, u.passwordHash)) {
    await audit("login_failed", u?.id ?? null, { email });
    return { error: "Invalid email or password." };
  }
  if (u.status === "pending") return { error: "Your account is awaiting admin approval." };
  if (u.status === "suspended") return { error: "This account has been suspended." };
  await audit("login", u.id, { role: u.role });
  await createSession(u.id);
  redirect("/dashboard"); // everyone lands in Studio
}

export async function logoutAction(): Promise<void> {
  try { const me = await getSessionUser(); if (me) await audit("logout", me.id, {}); } catch { /* ignore */ }
  await destroySession();
  redirect("/login");
}
