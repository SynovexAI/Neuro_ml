// Minimal Python pickle encoder (protocol 2) for JSON-like values — enough to
// export a trained model's parameters as a real .pkl that `pickle.load` reads
// back as a plain dict (no server, no Python). Supports null/bool/int/float/
// string/array/object; no shared-ref memo (we never emit backreferences).
export function pickle(root: unknown): Uint8Array {
  const out: number[] = [];
  const enc = new TextEncoder();
  const push = (...b: number[]) => { for (const x of b) out.push(x & 0xff); };
  const u32le = (n: number) => push(n, n >> 8, n >> 16, n >>> 24);

  function write(v: unknown): void {
    if (v === null || v === undefined) { push(0x4e); return; }              // NONE
    if (typeof v === "boolean") { push(v ? 0x88 : 0x89); return; }           // NEWTRUE/NEWFALSE
    if (typeof v === "number") {
      if (Number.isInteger(v) && v >= -2147483648 && v <= 2147483647) {
        if (v >= 0 && v < 256) push(0x4b, v);                                // BININT1
        else if (v >= 0 && v < 65536) push(0x4d, v, v >> 8);                 // BININT2
        else { push(0x4a); const dv = new DataView(new ArrayBuffer(4)); dv.setInt32(0, v, true); for (let i = 0; i < 4; i++) push(dv.getUint8(i)); } // BININT
      } else {
        push(0x47); const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, v, false); for (let i = 0; i < 8; i++) push(dv.getUint8(i)); // BINFLOAT (big-endian)
      }
      return;
    }
    if (typeof v === "string") { const b = enc.encode(v); push(0x58); u32le(b.length); for (const x of b) out.push(x); return; } // BINUNICODE
    if (Array.isArray(v)) { push(0x5d, 0x28); for (const item of v) write(item); push(0x65); return; }                          // EMPTY_LIST MARK … APPENDS
    if (typeof v === "object") { push(0x7d, 0x28); for (const [k, val] of Object.entries(v as Record<string, unknown>)) { write(k); write(val); } push(0x75); return; } // EMPTY_DICT MARK … SETITEMS
    const s = enc.encode(String(v)); push(0x58); u32le(s.length); for (const x of s) out.push(x);
  }

  push(0x80, 0x02);   // PROTO 2
  write(root);
  push(0x2e);         // STOP
  return new Uint8Array(out);
}
