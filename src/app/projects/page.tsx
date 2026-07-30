import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import MyProjects from "@/components/MyProjects";

export default async function ProjectsPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="My Projects">
      <MyProjects />
    </Shell>
  );
}
