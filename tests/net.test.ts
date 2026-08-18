import { describe, it, expect } from "vitest";
import { isBlockedHost, rateLimit } from "@/lib/net";

describe("isBlockedHost (SSRF guard)", () => {
  it("blocks loopback / localhost", () => {
    expect(isBlockedHost("localhost")).toBe(true);
    expect(isBlockedHost("127.0.0.1")).toBe(true);
    expect(isBlockedHost("127.9.9.9")).toBe(true);
    expect(isBlockedHost("::1")).toBe(true);
    expect(isBlockedHost("0.0.0.0")).toBe(true);
  });

  it("blocks private ranges and link-local metadata", () => {
    expect(isBlockedHost("10.0.0.5")).toBe(true);
    expect(isBlockedHost("192.168.1.1")).toBe(true);
    expect(isBlockedHost("172.16.0.1")).toBe(true);
    expect(isBlockedHost("172.31.255.255")).toBe(true);
    expect(isBlockedHost("169.254.169.254")).toBe(true); // cloud metadata
    expect(isBlockedHost("fc00::1")).toBe(true);
    expect(isBlockedHost("fe80::1")).toBe(true);
  });

  it("blocks internal TLDs", () => {
    expect(isBlockedHost("db.internal")).toBe(true);
    expect(isBlockedHost("service.local")).toBe(true);
  });

  it("allows public hosts", () => {
    expect(isBlockedHost("example.com")).toBe(false);
    expect(isBlockedHost("8.8.8.8")).toBe(false);
    expect(isBlockedHost("172.32.0.1")).toBe(false); // just outside the private 172.16-31 block
    expect(isBlockedHost("gateway01.ap-southeast-1.prod.aws.tidbcloud.com")).toBe(false);
  });
});

describe("rateLimit (in-memory window)", () => {
  it("allows up to max then blocks within the window", () => {
    const key = `test:${Math.random()}`;
    for (let i = 0; i < 3; i++) expect(rateLimit(key, 3, 60_000)).toBe(true);
    expect(rateLimit(key, 3, 60_000)).toBe(false);
    expect(rateLimit(key, 3, 60_000)).toBe(false);
  });

  it("keeps separate counters per key", () => {
    const a = `a:${Math.random()}`, b = `b:${Math.random()}`;
    expect(rateLimit(a, 1, 60_000)).toBe(true);
    expect(rateLimit(a, 1, 60_000)).toBe(false);
    expect(rateLimit(b, 1, 60_000)).toBe(true); // b unaffected by a
  });
});
