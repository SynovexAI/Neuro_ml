import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import Shell from "@/components/Shell";
import TemplatesGallery from "@/components/TemplatesGallery";

export default async function TemplatesPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  return (
    <Shell user={user} title="Templates">
      <div className="eyebrow">Studio</div>
      <h2 className="page-h">Templates</h2>
      <p className="page-sub">One-click starter projects for every Lab — a working build you can run immediately, then make your own.</p>
      <TemplatesGallery />
    </Shell>
  );
}
