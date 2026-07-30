import { describe, it, expect } from "vitest";
import { genDataset, initNet, trainEpochs, predict } from "@/lib/dlUtils";

describe("dlUtils MLP", () => {
  it("generates a labelled dataset of the requested size", () => {
    const d = genDataset("xor", 120, 0.1, 3);
    expect(d.X).toHaveLength(120);
    expect(d.y).toHaveLength(120);
    expect(d.X[0]).toHaveLength(2);
    expect(d.y.every((v) => v === 0 || v === 1)).toBe(true);
  });

  it("learns XOR: loss drops and accuracy improves with training", () => {
    const d = genDataset("xor", 200, 0.08, 3);
    const net = initNet([8, 6], "tanh");
    const start = trainEpochs(net, d, { lr: 0.3, l2: 0, epochs: 1 });
    let last = start;
    for (let k = 0; k < 8; k++) last = trainEpochs(net, d, { lr: 0.3, l2: 0, epochs: 50 });
    expect(last.loss).toBeLessThan(start.loss);   // training reduces loss
    expect(last.acc).toBeGreaterThanOrEqual(0.6); // and it actually separates the classes
  });

  it("predict returns a probability in [0,1]", () => {
    const net = initNet([4], "tanh");
    const p = predict(net, [0.3, -0.2]);
    expect(p).toBeGreaterThanOrEqual(0);
    expect(p).toBeLessThanOrEqual(1);
  });
});
