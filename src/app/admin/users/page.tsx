import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema";
import Shell from "@/components/Shell";
import UsersManager from "@/components/UsersManager";

export default async function UsersPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const rows = await db.select({
    id: users.id, email: users.email, name: users.name, role: users.role, status: users.status,
  }).from(users);

  return (
    <Shell user={user} title="Admin · Users">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">Users</h2>
      <p className="page-sub">Approve pending student sign-ups, change roles, and suspend accounts. New sign-ups start as pending until you approve them.</p>
      <UsersManager initial={rows} meId={user.id} />
    </Shell>
  );
}
