import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BYTES = 15 * 1024 * 1024; // 15 MB

// Extracts plain text from an uploaded document so the RAG lab can chunk it.
// Text formats are handled client-side; this route parses the binary ones:
// PDF (pdf-parse), Word .docx (mammoth), Excel .xlsx/.xls/.csv (SheetJS).
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  let form: FormData;
  try { form = await req.formData(); }
  catch { return NextResponse.json({ error: "Expected a multipart/form-data upload." }, { status: 400 }); }

  const file = form.get("file");
  if (!(file instanceof File)) return NextResponse.json({ error: "No file provided." }, { status: 400 });
  if (file.size > MAX_BYTES) return NextResponse.json({ error: `File too large (max ${MAX_BYTES / 1024 / 1024} MB).` }, { status: 413 });

  const name = file.name || "upload";
  const ext = (name.split(".").pop() || "").toLowerCase();
  const buf = Buffer.from(await file.arrayBuffer());

  let text = "";
  try {
    if (ext === "pdf") {
      const { PDFParse } = await import("pdf-parse");
      const parser = new PDFParse({ data: new Uint8Array(buf) });
      const res = await parser.getText();
      text = res.text || "";
    } else if (ext === "doc") {
      return NextResponse.json({ error: "Legacy .doc (Word 97–2003) isn't supported — open it in Word and Save As .docx, then upload." }, { status: 415 });
    } else if (ext === "docx") {
      const mammoth = await import("mammoth");
      const res = await mammoth.extractRawText({ buffer: buf });
      text = res.value || "";
    } else if (ext === "xlsx" || ext === "xls" || ext === "xlsm" || ext === "csv") {
      const XLSX = await import("xlsx");
      const wb = XLSX.read(buf, { type: "buffer" });
      text = wb.SheetNames.map((n) => `# Sheet: ${n}\n${XLSX.utils.sheet_to_csv(wb.Sheets[n])}`).join("\n\n");
    } else {
      text = buf.toString("utf8");
    }
  } catch (e) {
    return NextResponse.json({ error: `Could not parse ${name}: ${(e as Error).message}` }, { status: 422 });
  }

  text = text
    .replace(/^-{2,}\s*\d+\s+of\s+\d+\s*-{2,}$/gim, "") // pdf-parse page markers
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  if (!text) return NextResponse.json({ error: `No extractable text found in ${name}.` }, { status: 422 });

  return NextResponse.json({ title: name, kind: ext || "file", text: text.slice(0, 600_000) });
}
