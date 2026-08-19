import { redirect } from "next/navigation";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { providers } from "@/lib/db/schema";
import { decrypt, maskKey } from "@/lib/crypto";
import Shell from "@/components/Shell";
import ProvidersManager from "@/components/ProvidersManager";

export default async function ProvidersPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");
  if (user.role !== "admin") redirect("/dashboard");

  const rows = await db.select().from(providers);
  const initial = rows.map((p) => ({
    id: p.id, provider: p.provider, label: p.label, baseUrl: p.baseUrl,
    defaultModel: p.defaultModel, enabled: p.enabled,
    maskedKey: p.apiKeyEnc ? maskKey(decrypt(p.apiKeyEnc)) : "",
  }));

  return (
    <Shell user={user} title="Admin · Providers">
      <div className="eyebrow">Admin</div>
      <h2 className="page-h">Providers &amp; models</h2>
      <p className="page-sub">Choose a provider, load its live model list with your API key, and save. The key is stored AES-256-GCM encrypted and used for every LLM call across the platform.</p>
      <ProvidersManager initial={initial} />
    </Shell>
  );
}
