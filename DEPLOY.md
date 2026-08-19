# Deploy guide — free stack (Vercel + TiDB Serverless + Cloudflare R2)

This app is now ready for a **100% free** stack that handles ~100 concurrent users:

- **Compute:** Vercel Hobby (auto-scales, no spin-down)
- **Database:** TiDB Cloud Serverless (you already use it) — via the **HTTP driver** for concurrency
- **File storage:** Cloudflare R2 (10 GB free, zero egress)
- **LLM concurrency:** each user configures **their own free provider key**

Code already changed for this:
- `src/lib/db/index.ts` — dual driver: mysql2 (default) **or** TiDB HTTP driver when `TIDB_DRIVER=http`.
- `src/lib/r2.ts` — R2 storage helper (no-ops if R2 env vars are unset).
- `src/app/api/storage/upload/route.ts` — `POST` a file → stored in R2 → `{ key, url }`.

---

## 1. TiDB (database) — you likely already have this

You need `DATABASE_URL` and, on Vercel, `TIDB_DRIVER=http`.

1. Go to **tidbcloud.com** → your **Serverless** cluster → **Connect**.
2. Choose connection type **General** (or "Connect With: mysql"). Copy the values (host, port `4000`, user, password, database).
3. Build the URL: `mysql://<user>:<password>@<host>:4000/<database>` (URL-encode special chars in the password).
   - This is the same `DATABASE_URL` you already use on Render.
4. On Vercel, **also set `TIDB_DRIVER=http`** — this switches to the serverless HTTP driver so 100 concurrent requests don't exhaust connections. (On Render you can leave it unset to keep mysql2.)

> The HTTP driver works only with **TiDB Cloud Serverless** clusters — which is what you have.

---

## 2. File storage — Vercel Blob (recommended, no card) or R2

Storage is optional (without it, docs are stored as text in the DB and uploads still work).
Backend is auto-selected: **Vercel Blob** if `BLOB_READ_WRITE_TOKEN` is set, else **R2** if the `R2_*` vars are set.

### Vercel Blob (no card — recommended)
1. On Vercel → your project → **Storage** tab → **Create** → **Blob** → give it a name → Create.
2. Vercel automatically adds the **`BLOB_READ_WRITE_TOKEN`** env var to the project. That's it — nothing else to set.
3. Free allowance ≈ **1 GB** (Hobby). Manage/delete files in-app at **Control Room → Storage** when it fills up.

### Cloudflare R2 (needs a card — optional, 10 GB)

1. Sign in at **dash.cloudflare.com** → **R2** (left sidebar). Enable R2 if prompted (free plan is fine; no card needed for the free tier).
2. **Create a bucket** → give it a name, e.g. `workbench-files`. Note the name → `R2_BUCKET`.
3. Get your **Account ID**: R2 overview page (top right) → copy **Account ID** → `R2_ACCOUNT_ID`.
4. Create an **API token**: R2 → **Manage R2 API Tokens** → **Create API token** →
   - Permission: **Object Read & Write**
   - (Optionally scope it to your bucket)
   - Create → copy the **Access Key ID** → `R2_ACCESS_KEY_ID`, and the **Secret Access Key** → `R2_SECRET_ACCESS_KEY` (shown once).
5. (Optional) **Public reads:** in the bucket → **Settings** → connect a **custom domain** or enable the public `r2.dev` URL, then set `R2_PUBLIC_URL` to that base (e.g. `https://files.example.com`). If you skip this, the app serves files via short-lived **signed URLs** automatically.

R2 env vars:
```
R2_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxx
R2_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
R2_BUCKET=workbench-files
R2_PUBLIC_URL=            # optional
```

Test after deploy: `POST /api/storage/upload` with a `multipart/form-data` field named `file`. It returns `{ key, url }`. (Wiring this into the RAG/ETL upload UI is a follow-up — see §5.)

---

## 3. ENCRYPTION_KEY (required)

The app encrypts provider API keys with AES-256-GCM. Generate a 32-byte key (64 hex chars):

```bash
openssl rand -hex 32
```
or
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

Set the output as `ENCRYPTION_KEY`. **Use the SAME value you already use on Render** — changing it makes existing stored keys undecryptable.

---

## 4. Deploy to Vercel

1. Push your repo to GitHub (already at `github.com/SynovexAI/Neuro_ml`).
2. **vercel.com** → sign in with GitHub → **Add New… → Project** → import the repo.
3. Framework preset auto-detects **Next.js** — leave build/output defaults.
4. **Environment Variables** — add all of these (Production + Preview):

   | Var | Required | Value |
   |---|---|---|
   | `DATABASE_URL` | ✅ | your TiDB URL (§1) |
   | `TIDB_DRIVER` | ✅ (Vercel) | `http` |
   | `ENCRYPTION_KEY` | ✅ | 64-hex key (§3, same as Render) |
   | `APP_URL` | ✅ | `https://<your-app>.vercel.app` |
   | `NEXT_PUBLIC_APP_URL` | ✅ | same as `APP_URL` |
   | `BLOB_READ_WRITE_TOKEN` | for storage (recommended) | auto-added when you create a Vercel Blob store (§2) |
   | `R2_ACCOUNT_ID` | only if using R2 instead | §2 |
   | `R2_ACCESS_KEY_ID` | for storage | §2 |
   | `R2_SECRET_ACCESS_KEY` | for storage | §2 |
   | `R2_BUCKET` | for storage | §2 |
   | `R2_PUBLIC_URL` | optional | §2 |
   | `DEFAULT_MONTHLY_TOKEN_LIMIT` | optional | e.g. `1000000` |
   | `GITHUB_TOKEN` | optional | for GitHub Models / higher rate limit |

5. **Deploy.** You get a `*.vercel.app` URL. Add a custom domain later under Settings → Domains.

### Vercel timeout note (important for the agent)
Vercel Hobby caps a function at **60 s**. Your **agentic** runs make several LLM calls and can exceed that. Two mitigations:
- Keep the agent **Max steps ≤ 3–4** (in the RAG agent setup) so a run finishes under 60 s.
- Add `export const maxDuration = 60;` to long routes (`src/app/api/chat/route.ts`, and the agent path) to use the full window.

---

## 5. LLM concurrency — per-user keys (DONE — one migration required)

Implemented: each user can add **their own** provider keys under **My API keys** (`/settings/keys`).
The labs use the **user's own key first**, falling back to the shared/global providers — so 100
concurrent generations spread across everyone's free quotas instead of hitting one shared limit.

This uses a **new `user_providers` table** (the admin `providers` table is untouched). The code
degrades gracefully until the table exists, so **run this migration once** on TiDB:

**Option A — SQL** (TiDB Cloud → SQL Editor, paste and run):
```sql
CREATE TABLE IF NOT EXISTS user_providers (
  id             varchar(36)  NOT NULL,
  user_id        varchar(36)  NOT NULL,
  provider       varchar(40)  NOT NULL,
  label          varchar(120),
  base_url       varchar(255) NOT NULL,
  api_key_enc    text,
  default_model  varchar(120),
  enabled        boolean      NOT NULL DEFAULT true,
  created_at     timestamp    DEFAULT (now()),
  updated_at     timestamp    DEFAULT (now()) ON UPDATE now(),
  PRIMARY KEY (id),
  KEY user_providers_user_idx (user_id)
);
```

**Option B — Drizzle** (from your machine, with `DATABASE_URL` set):
```bash
npx drizzle-kit push
```

Until you run it, "My API keys" shows an error on save and the app keeps using shared providers (no breakage).

## 6. Storage + KB wiring (DONE)

- **RAG uploads** archive the original file to storage (Blob/R2) — best-effort; doc shows a **☁ stored** link. No-ops if storage isn't set.
- **Studio KB uploads** archive originals too.
- **RAG lab "Knowledge base" source connector** — pick a saved KB and pull its docs into the pipeline (`GET /api/kb/[id]/docs`).
- **Agent ↔ KB** — the agent's Knowledge node can **load a saved KB** ("Load from a saved Knowledge base"), so agents ground on stored KBs (not just pasted text).
- **Control Room → Storage** — admin lists all stored files with a usage bar and **Delete** per file (free space when full).
- **Control Room → Analytics** — links to Vercel Web Analytics (traffic; `@vercel/analytics` tracking is wired in the root layout) + the in-app Usage/Agent/Storage pages.

Env: set **`BLOB_READ_WRITE_TOKEN`** (auto-added when you create a Vercel Blob store) to enable storage.
Note: an `.npmrc` with `legacy-peer-deps=true` was added so `@vercel/analytics` installs cleanly (Vercel honors it).

---

## Quick checklist
- [ ] TiDB `DATABASE_URL` set · `TIDB_DRIVER=http` on Vercel
- [ ] `ENCRYPTION_KEY` = same 64-hex value as Render
- [ ] R2 bucket + API token → `R2_*` env vars
- [ ] `APP_URL` / `NEXT_PUBLIC_APP_URL` = your Vercel URL
- [ ] Agent Max steps ≤ 3–4 (Vercel 60 s cap) or add `maxDuration`
- [ ] (later) per-user LLM keys for true 100-concurrent generations
