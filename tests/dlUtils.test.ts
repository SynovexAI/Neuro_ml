import { describe, it, expect } from "vitest";
import { genDataset, initNet, newOpt, trainEpoch, evalNet, fullEval, predictVec, fitScaler, applyScaler, type DlTask } from "@/lib/dlUtils";

function train(kind: string, task: DlTask, outDim: number, hidden: number[], act: string, opt: "sgd" | "momentum" | "adam", lr: number, epochs: number) {
  const d = genDataset(kind, 240, 0.12, 3);
  const sc = fitScaler(d.X); const Xs = applyScaler(d.X, sc);
  const net = initNet(Xs[0].length, hidden, outDim, act, task);
  const st = newOpt(net, opt);
  const first = trainEpoch(net, Xs, d.y, { lr, l2: 0, batchSize: 16 }, st);
  let last = first; for (let e = 0; e < epochs; e++) last = trainEpoch(net, Xs, d.y, { lr, l2: 0, batchSize: 16 }, st);
  return { d, sc, Xs, net, first, last };
}

describe("generalized MLP engine", () => {
  it("binary (moons) learns with SGD", () => {
    const { last, first } = train("moons", "binary", 1, [12, 12], "tanh", "sgd", 0.1, 400);
    expect(last.loss).toBeLessThan(first.loss); expect(last.acc).toBeGreaterThan(0.9);
  });

  it("multiclass (3 blobs) learns with softmax + Adam; proba sums to 1", () => {
    const { net, Xs, last } = train("blobs3", "multiclass", 3, [10], "relu", "adam", 0.02, 300);
    expect(last.acc).toBeGreaterThan(0.9);
    const p = predictVec(net, Xs[0]); expect(p.length).toBe(3); expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6);
    const ev = fullEval(net, Xs, genDataset("blobs3", 240, 0.12, 3).y, ["A", "B", "C"]);
    expect(ev.confusion?.length).toBe(3);
  });

  it("regression (sine) fits: R² rises well above 0", () => {
    const { net, Xs, d, last } = train("sine", "regression", 1, [16, 16], "tanh", "adam", 0.01, 500);
    expect(last.acc).toBeGreaterThan(0.7); // acc = R² for regression
    const ev = fullEval(net, Xs, d.y, []);
    expect(ev.predActual?.length).toBe(d.X.length);
  });

  it("optimizers all reduce loss (momentum, adam)", () => {
    for (const o of ["momentum", "adam"] as const) { const { first, last } = train("xor", "binary", 1, [8], "tanh", o, o === "adam" ? 0.03 : 0.1, 300); expect(last.loss).toBeLessThan(first.loss); }
  });

  it("evalNet returns finite loss/metric", () => {
    const { net, Xs, d } = train("circles", "binary", 1, [8], "tanh", "sgd", 0.1, 50);
    const e = evalNet(net, Xs, d.y); expect(Number.isFinite(e.loss)).toBe(true); expect(Number.isFinite(e.acc)).toBe(true);
  });
});
