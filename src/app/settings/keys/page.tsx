import { redirect } from "next/navigation";
import { eq, desc } from "drizzle-orm";
import { getSessionUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { userProviders } from "@/lib/db/schema";
import { decrypt, maskKey } from "@/lib/crypto";
import Shell from "@/components/Shell";
import ProvidersManager from "@/components/ProvidersManager";

export const dynamic = "force-dynamic";

export default async function MyKeysPage() {
  const user = await getSessionUser();
  if (!user) redirect("/login");

  // Load the user's own providers. Tolerate a missing user_providers table (before migration).
  let initial: { id: string; provider: string; label: string | null; baseUrl: string; defaultModel: string | null; enabled: boolean; maskedKey: string }[] = [];
  try {
    const rows = await db.select().from(userProviders).where(eq(userProviders.userId, user.id)).orderBy(desc(userProviders.createdAt));
    initial = rows.map((p) => ({
      id: p.id, provider: p.provider, label: p.label, baseUrl: p.baseUrl,
      defaultModel: p.defaultModel, enabled: p.enabled,
      maskedKey: p.apiKeyEnc ? maskKey(decrypt(p.apiKeyEnc)) : "",
    }));
  } catch { initial = []; }

  return (
    <Shell user={user} title="My API keys">
      <div className="eyebrow">Settings</div>
      <h2 className="page-h">My API keys</h2>
      <p className="page-sub">Add your own LLM provider keys — the labs will use <b>your</b> key first, falling back to a shared one. This spreads load across everyone&apos;s free quotas, so heavy concurrent use doesn&apos;t hit a single shared rate limit. Keys are stored AES-256-GCM encrypted and never returned to the browser.</p>
      <ProvidersManager initial={initial} basePath="/api/me/providers" />
    </Shell>
  );
}
