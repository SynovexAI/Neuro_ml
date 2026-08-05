import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Real GitHub connector — pulls file/README/folder text from PUBLIC repos via the
// GitHub REST API (no auth needed; set GITHUB_TOKEN in env to raise the rate limit).
const TEXT_EXT = /\.(md|markdown|mdx|txt|rst|py|js|jsx|ts|tsx|json|ya?ml|toml|ini|cfg|go|rs|java|kt|rb|php|c|h|cpp|cc|hpp|cs|swift|scala|sh|bash|sql|html?|css|scss|xml|csv|tsv|env|dockerfile|gradle|properties|proto|graphql|vue|svelte|r|jl|lua|pl|ex|exs)$/i;
const MAX_FILES = 12;
const MAX_CHARS = 200000;

type GhContent = { name: string; path: string; type: string; download_url: string | null; size: number };

function ghHeaders(): Record<string, string> {
  const h: Record<string, string> = { accept: "application/vnd.github+json", "user-agent": "AI-Workbench-RAG" };
  if (process.env.GITHUB_TOKEN) h.authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  return h;
}

async function ghFetch(url: string) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 15000);
  try { return await fetch(url, { headers: ghHeaders(), signal: ctrl.signal }); }
  finally { clearTimeout(t); }
}

// Parse the many shapes a user may paste into {owner, repo, branch?, path?, kind}.
function parseTarget(input: string): { owner: string; repo: string; branch?: string; path?: string; kind: "repo" | "blob" | "tree" } | null {
  const s = (input || "").trim();
  if (!s) return null;
  // raw.githubusercontent.com/owner/repo/branch/path...
  const raw = s.match(/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/i);
  if (raw) return { owner: raw[1], repo: raw[2], branch: raw[3], path: raw[4], kind: "blob" };
  // full github.com URL
  const m = s.match(/github\.com\/([^/]+)\/([^/?#]+)(?:\/(blob|tree)\/([^/]+)\/(.+?))?\/?(?:[?#].*)?$/i);
  if (m) {
    const owner = m[1], repo = m[2].replace(/\.git$/, "");
    if (m[3] && m[4]) return { owner, repo, branch: m[4], path: m[5], kind: m[3] === "blob" ? "blob" : "tree" };
    return { owner, repo, kind: "repo" };
  }
  // shorthand: owner/repo  or  owner/repo/path...
  const sh = s.match(/^([\w.-]+)\/([\w.-]+)(?:\/(.+))?$/);
  if (sh) return { owner: sh[1], repo: sh[2].replace(/\.git$/, ""), path: sh[3], kind: sh[3] ? "tree" : "repo" };
  return null;
}

async function readFile(owner: string, repo: string, path: string, branch?: string): Promise<{ name: string; text: string } | null> {
  const ref = branch ? `?ref=${encodeURIComponent(branch)}` : "";
  const res = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/contents/${path.split("/").map(encodeURIComponent).join("/")}${ref}`);
  if (!res.ok) return null;
  const j = (await res.json()) as GhContent & { content?: string; encoding?: string };
  if (j.type !== "file") return null;
  let text = "";
  if (j.content && j.encoding === "base64") text = Buffer.from(j.content, "base64").toString("utf-8");
  else if (j.download_url) text = await (await ghFetch(j.download_url)).text();
  return { name: j.path || j.name, text: text.slice(0, MAX_CHARS) };
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { url } = await req.json().catch(() => ({}));
  const tgt = parseTarget(url);
  if (!tgt) return NextResponse.json({ error: "Enter a GitHub repo/file URL or owner/repo." }, { status: 400 });
  const { owner, repo, branch, path, kind } = tgt;
  try {
    const docs: { name: string; text: string }[] = [];

    if (kind === "repo") {
      // fetch the repository README
      const res = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/readme`);
      if (res.status === 404) return NextResponse.json({ error: `No README found in ${owner}/${repo} (or the repo is private).` }, { status: 404 });
      if (res.status === 403) return NextResponse.json({ error: "GitHub rate limit hit. Add a GITHUB_TOKEN env var, or try again shortly." }, { status: 429 });
      if (!res.ok) return NextResponse.json({ error: `GitHub error ${res.status} for ${owner}/${repo}.` }, { status: 502 });
      const j = (await res.json()) as { path: string; content?: string; encoding?: string; download_url?: string };
      const text = j.content && j.encoding === "base64" ? Buffer.from(j.content, "base64").toString("utf-8") : j.download_url ? await (await ghFetch(j.download_url)).text() : "";
      docs.push({ name: `${repo}/${j.path}`, text: text.slice(0, MAX_CHARS) });
    } else if (kind === "blob" && path) {
      const f = await readFile(owner, repo, path, branch);
      if (!f) return NextResponse.json({ error: `Could not read ${path} from ${owner}/${repo}.` }, { status: 404 });
      docs.push(f);
    } else if (kind === "tree" && path) {
      // list a directory and pull its text files
      const ref = branch ? `?ref=${encodeURIComponent(branch)}` : "";
      const res = await ghFetch(`https://api.github.com/repos/${owner}/${repo}/contents/${path.split("/").map(encodeURIComponent).join("/")}${ref}`);
      if (res.status === 403) return NextResponse.json({ error: "GitHub rate limit hit. Add a GITHUB_TOKEN env var, or try again shortly." }, { status: 429 });
      if (!res.ok) return NextResponse.json({ error: `Could not list ${path} in ${owner}/${repo} (${res.status}).` }, { status: 502 });
      const listing = await res.json();
      if (!Array.isArray(listing)) {
        // it was actually a file
        const f = await readFile(owner, repo, path, branch);
        if (f) docs.push(f);
      } else {
        const files = (listing as GhContent[]).filter((e) => e.type === "file" && TEXT_EXT.test(e.name) && e.size < 400000).slice(0, MAX_FILES);
        for (const e of files) {
          const f = await readFile(owner, repo, e.path, branch);
          if (f && f.text.trim()) docs.push(f);
        }
        if (!files.length) return NextResponse.json({ error: `No text files found in ${owner}/${repo}/${path}.` }, { status: 404 });
      }
    }

    if (!docs.length) return NextResponse.json({ error: "Nothing to import." }, { status: 404 });
    return NextResponse.json({ repo: `${owner}/${repo}`, docs });
  } catch (e) {
    return NextResponse.json({ error: `GitHub fetch failed: ${(e as Error).message}` }, { status: 502 });
  }
}
