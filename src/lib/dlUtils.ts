// A real from-scratch MLP (forward + backprop, online SGD) for 2-D binary
// classification — powers the DL Lab's live decision-boundary demo. No libraries.

export interface Net { sizes: number[]; W: number[][][]; b: number[][]; act: string; }
export interface DataSet { X: number[][]; y: number[]; }

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const R = 5; // data lives in roughly [-R, R]^2

export function genDataset(kind: string, n = 200, noise = 0.15, seed = 3): DataSet {
  const rng = mulberry32(seed); const X: number[][] = [], y: number[] = [];
  const jit = () => (rng() * 2 - 1) * noise * R;
  if (kind === "xor") {
    for (let i = 0; i < n; i++) { const x = (rng() * 2 - 1) * R, z = (rng() * 2 - 1) * R; X.push([x + jit(), z + jit()]); y.push((x > 0) === (z > 0) ? 1 : 0); }
  } else if (kind === "circles") {
    for (let i = 0; i < n; i++) { const inner = i < n / 2; const r = inner ? rng() * R * 0.45 : R * 0.6 + rng() * R * 0.4; const t = rng() * Math.PI * 2; X.push([r * Math.cos(t) + jit(), r * Math.sin(t) + jit()]); y.push(inner ? 0 : 1); }
  } else if (kind === "moons") {
    for (let i = 0; i < n; i++) { const up = i < n / 2; const t = rng() * Math.PI; if (up) X.push([R * Math.cos(t) + jit(), R * Math.sin(t) - R * 0.3 + jit()]); else X.push([R - R * Math.cos(t) - R * 0.7 + jit(), -R * Math.sin(t) + R * 0.3 + jit()]); y.push(up ? 0 : 1); }
  } else { // spiral
    const per = Math.floor(n / 2);
    for (let c = 0; c < 2; c++) for (let i = 0; i < per; i++) { const r = (i / per) * R; const t = c * Math.PI + (i / per) * 3.2; X.push([r * Math.sin(t) + jit(), r * Math.cos(t) + jit()]); y.push(c); }
  }
  return { X, y };
}

const actF = (name: string, z: number) => name === "relu" ? Math.max(0, z) : name === "sigmoid" ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
const actD = (name: string, a: number) => name === "relu" ? (a > 0 ? 1 : 0) : name === "sigmoid" ? a * (1 - a) : 1 - a * a;

export function initNet(hidden: number[], act: string): Net {
  const sizes = [2, ...hidden, 1]; const rng = mulberry32(42); const W: number[][][] = [], b: number[][] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fin = sizes[l], fout = sizes[l + 1]; const scale = act === "relu" ? Math.sqrt(2 / fin) : Math.sqrt(1 / fin);
    W.push(Array.from({ length: fout }, () => Array.from({ length: fin }, () => (rng() * 2 - 1) * scale)));
    b.push(new Array(fout).fill(0));
  }
  return { sizes, W, b, act };
}

export function forward(net: Net, x: number[]): { as: number[][] } {
  const as: number[][] = [x.slice()];
  for (let l = 0; l < net.W.length; l++) {
    const out = new Array(net.W[l].length); const last = l === net.W.length - 1;
    for (let i = 0; i < net.W[l].length; i++) { let z = net.b[l][i]; const row = net.W[l][i]; for (let j = 0; j < row.length; j++) z += row[j] * as[l][j]; out[i] = last ? 1 / (1 + Math.exp(-z)) : actF(net.act, z); }
    as.push(out);
  }
  return { as };
}
export const predict = (net: Net, x: number[]): number => forward(net, x).as[net.W.length][0];

function sgdStep(net: Net, x: number[], y: number, lr: number, l2: number) {
  const { as } = forward(net, x); const L = net.W.length; const deltas: number[][] = new Array(L);
  deltas[L - 1] = [as[L][0] - y];
  for (let l = L - 2; l >= 0; l--) {
    const dl = new Array(net.sizes[l + 1]).fill(0);
    for (let j = 0; j < net.sizes[l + 1]; j++) { let s = 0; for (let i = 0; i < net.sizes[l + 2]; i++) s += net.W[l + 1][i][j] * deltas[l + 1][i]; dl[j] = s * actD(net.act, as[l + 1][j]); }
    deltas[l] = dl;
  }
  for (let l = 0; l < L; l++) {
    for (let i = 0; i < net.W[l].length; i++) {
      const row = net.W[l][i]; for (let j = 0; j < row.length; j++) row[j] -= lr * (deltas[l][i] * as[l][j] + l2 * row[j]);
      net.b[l][i] -= lr * deltas[l][i];
    }
  }
}

export function trainEpochs(net: Net, d: DataSet, p: { lr: number; l2: number; epochs: number }): { loss: number; acc: number } {
  const idx = d.X.map((_, i) => i); const rng = mulberry32(99);
  for (let e = 0; e < p.epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    for (const i of idx) sgdStep(net, d.X[i], d.y[i], p.lr, p.l2);
  }
  let loss = 0, correct = 0;
  for (let i = 0; i < d.X.length; i++) { const pr = predict(net, d.X[i]); loss += -(d.y[i] * Math.log(Math.max(1e-7, pr)) + (1 - d.y[i]) * Math.log(Math.max(1e-7, 1 - pr))); if ((pr > 0.5 ? 1 : 0) === d.y[i]) correct++; }
  return { loss: loss / d.X.length, acc: correct / d.X.length };
}

export const DATA_BOUND = R;
