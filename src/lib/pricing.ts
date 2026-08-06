// Rough public blended ($/1M tokens) prices, only for *estimated* cost figures
// in the analytics UI. Not billing-grade — a coarse map + a sane default.
const PRICES: { match: RegExp; perM: number }[] = [
  { match: /gpt-4o-mini|4\.1-mini|o4-mini/i, perM: 0.3 },
  { match: /gpt-4o|gpt-4\.1|gpt-5/i, perM: 5 },
  { match: /llama-3\.3-70b|llama-3\.1-70b|70b/i, perM: 0.6 },
  { match: /llama.*8b|8b/i, perM: 0.1 },
  { match: /mixtral|mistral/i, perM: 0.5 },
  { match: /flash/i, perM: 0.15 },
  { match: /gemini/i, perM: 2 },
  { match: /haiku/i, perM: 1 },
  { match: /sonnet/i, perM: 6 },
  { match: /opus/i, perM: 30 },
];
const DEFAULT_PER_M = 0.5;

export function pricePerMillion(model: string): number {
  for (const p of PRICES) if (p.match.test(model || "")) return p.perM;
  return DEFAULT_PER_M;
}

export function estCostUsd(model: string, totalTokens: number): number {
  return (totalTokens / 1_000_000) * pricePerMillion(model);
}

export function fmtUsd(n: number): string {
  return n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`;
}
