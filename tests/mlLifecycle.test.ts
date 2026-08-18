import { describe, expect, it } from "vitest";
import { trainLogReg, predict } from "../src/lib/mlUtils";
import { driftReport, limeExplain, makeVersionRecord, promoteChallenger, shapExplain } from "../src/lib/mlLifecycle";
import { validateDataset } from "../src/lib/dataQuality";

describe("ML lifecycle utilities", () => {
  const X = [[0,0],[0,1],[1,0],[1,1],[2,2],[2,3],[3,2],[3,3]];
  const y = [0,0,0,1,1,1,1,1];
  const model = trainLogReg(X, y, 2, { lr: 0.2, epochs: 120, l2: 0.01 });

  it("produces real SHAP contributions that approximately sum to prediction-base", () => {
    const r = shapExplain(model, X[3], X, ["a", "b"], "classification", 96, 1);
    expect(r.rows).toHaveLength(2);
    expect(r.prediction - r.baseValue).toBeCloseTo(r.rows.reduce((s, x) => s + x.contribution, 0), 1);
  });

  it("fits a local LIME surrogate", () => {
    const r = limeExplain(model, X[3], X, ["a", "b"], "classification", 96, undefined, 1);
    expect(r.rows).toHaveLength(2);
    expect(Number.isFinite(r.r2)).toBe(true);
  });

  it("detects distribution drift", () => {
    const r = driftReport([0,0,0,0,1,1,1,1], [5,5,6,6,7,7,8,8], "PSI");
    expect(r.drifted).toBe(true);
  });

  it("promotes a better challenger", () => {
    const champion = makeVersionRecord({ name: "m", version: "v1", datasetVersion: "d1", features: ["a"], preprocessing: [], algorithm: "LR", hyperparameters: {}, metrics: { F1: 0.8 }, status: "Production" });
    const challenger = makeVersionRecord({ name: "m", version: "v2", datasetVersion: "d2", features: ["a","b"], preprocessing: [], algorithm: "RF", hyperparameters: {}, metrics: { F1: 0.9 }, status: "Validation" });
    expect(promoteChallenger(champion, challenger, "F1").winner.version).toBe("v2");
    expect(predict(model, X).length).toBe(X.length);
  });
});


describe("ML validation guards", () => {
  it("flags duplicate columns, invalid numeric values, and a missing target", () => {
    const ds = {
      nrows: 4,
      columns: [
        { name: "x", type: "num" as const, values: [1, 2, Number.NaN, 4] },
        { name: "x", type: "num" as const, values: [1, 2, 3, 4] },
      ],
    };
    const issues = validateDataset(ds, "target");
    expect(issues.some((i) => i.category === "Schema" && i.severity === "ERROR")).toBe(true);
    expect(issues.some((i) => i.category === "Value" && i.severity === "ERROR")).toBe(true);
  });
});


import { describe, expect, it } from "vitest";
import {
  buildMatrix, makeModel, predict, predictProba, trainLogReg, trainLinear,
  imputeNumCol, scaleNumCol, transformNumCol, outlierNumCol, binNumCol,
  type Dataset, type PrepStep, type TrainConfig,
} from "../src/lib/mlUtils";

const ds: Dataset = {
  nrows: 10,
  columns: [
    { name: "age", type: "num", values: [20, 21, null, 23, 24, 25, 26, 27, 28, 29] },
    { name: "income", type: "num", values: [20, 22, 24, 26, 28, 30, 32, 34, 36, 38] },
    { name: "city", type: "cat", values: ["A", "B", "A", null, "B", "C", "A", "C", "B", "A"] },
    { name: "target", type: "cat", values: ["N", "N", "Y", "Y", "N", "Y", "Y", "N", "Y", "N"] },
  ],
};

describe("ML preprocessing and model smoke tests", () => {
  it("runs every numeric preprocessing primitive without non-finite output", () => {
    const raw = [1, null, 3, 4, 100, 6];
    for (const method of ["Mean", "Median", "Most frequent", "Constant", "Min", "Max", "Forward fill", "Backward fill", "Interpolate"])
      expect(imputeNumCol(raw, method).every(Number.isFinite)).toBe(true);
    for (const method of ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileUniform", "None"])
      expect(scaleNumCol(imputeNumCol(raw, "Mean"), method).every(v => v == null || Number.isFinite(v))).toBe(true);
    for (const method of ["Log", "Log1p", "Sqrt", "Square", "Cube root", "Reciprocal", "Absolute"])
      expect(transformNumCol([1, 2, 4], method).every(v => v == null || Number.isFinite(v))).toBe(true);
    for (const method of ["IQR clip", "IQR replace", "Z-score clip", "Winsorize 5%"])
      expect(outlierNumCol([1, 2, 3, 4, 100], method).every(v => v == null || Number.isFinite(v))).toBe(true);
    for (const method of ["Equal-width (5)", "Equal-freq (5)", "Equal-width (10)"])
      expect(binNumCol([1, 2, 3, 4, 5], method).every(v => v == null || Number.isFinite(v))).toBe(true);
  });

  it("builds stable one-hot/ordinal/frequency/count/binary matrices", () => {
    for (const method of ["One-Hot", "Ordinal", "Frequency", "Count", "Binary"]) {
      const steps: PrepStep[] = [{ op: "Impute missing", cols: ["age", "city"], method: "Most frequent" }, { op: "Encode categorical", cols: ["city"], method }];
      const b = buildMatrix(ds, ["age", "income", "city"], "target", "classification", steps);
      expect(b.X).toHaveLength(ds.nrows);
      expect(b.X.every(r => r.every(Number.isFinite))).toBe(true);
      expect(b.classes).toEqual(["N", "Y"]);
    }
  });

  it("trains all supported model families", () => {
    const X = [[0,0],[0,1],[1,0],[1,1],[2,2],[2,3],[3,2],[3,3]];
    const yc = [0,0,0,1,1,1,1,1];
    const yr = [0,1,1,2,2,3,3,4];
    const configs: TrainConfig[] = [
      { task: "classification", algo: "LogisticRegression", params: { C: "1", max_iter: "100", learning_rate: "0.2" }, testSize: .2, cvFolds: 3 },
      { task: "classification", algo: "KNeighborsClassifier", params: { n_neighbors: "3", weights: "uniform" }, testSize: .2, cvFolds: 3 },
      { task: "classification", algo: "GaussianNB", params: {}, testSize: .2, cvFolds: 3 },
      { task: "classification", algo: "DecisionTree", params: { max_depth: "4", min_samples_split: "2" }, testSize: .2, cvFolds: 3 },
      { task: "classification", algo: "RandomForest", params: { n_estimators: "8", max_depth: "4", min_samples_split: "2" }, testSize: .2, cvFolds: 3 },
      { task: "regression", algo: "LinearRegression", params: { fit_intercept: "True" }, testSize: .2, cvFolds: 3 },
      { task: "regression", algo: "Ridge", params: { alpha: "1" }, testSize: .2, cvFolds: 3 },
      { task: "regression", algo: "KNeighborsRegressor", params: { n_neighbors: "3", weights: "uniform" }, testSize: .2, cvFolds: 3 },
      { task: "regression", algo: "DecisionTree", params: { max_depth: "4", min_samples_split: "2" }, testSize: .2, cvFolds: 3 },
      { task: "regression", algo: "RandomForest", params: { n_estimators: "8", max_depth: "4", min_samples_split: "2" }, testSize: .2, cvFolds: 3 },
    ];
    for (const cfg of configs) {
      const y = cfg.task === "classification" ? yc : yr;
      const m = makeModel(cfg, X, y, cfg.task === "classification" ? 2 : 0);
      const p = predict(m, X);
      expect(p).toHaveLength(X.length);
      expect(p.every(Number.isFinite)).toBe(true);
      if (cfg.task === "classification") expect(predictProba(m, X).every(row => row.length === 2)).toBe(true);
    }
    expect(predict(trainLogReg(X, yc, 2, { lr: .2, epochs: 100, l2: .01 }), X)).toHaveLength(X.length);
    expect(predict(trainLinear(X, yr, { lr: .05, epochs: 100, alpha: 0 }), X)).toHaveLength(X.length);
  });
});
