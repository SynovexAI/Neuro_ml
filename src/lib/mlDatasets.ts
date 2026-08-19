// Deterministic sample datasets with genuine learnable structure (seeded),
// returned as CSV so they flow through the exact same parse → train path as uploads.

function rng(seed: number) { let s = seed; return () => (s = (s * 9301 + 49297) % 233280) / 233280; }
const pick = <T,>(r: () => number, arr: T[]) => arr[Math.floor(r() * arr.length)];

function churnCSV(): string {
  const r = rng(7); const rows = ["age,tenure_months,monthly_charges,contract,payment_method,has_support,churn"];
  for (let i = 0; i < 90; i++) {
    const age = 20 + Math.round(r() * 55);
    const tenure = Math.round(r() * 72);
    const monthly = Math.round((28 + r() * 92) * 10) / 10;
    const contract = pick(r, ["Month-to-month", "One year", "Two year"]);
    const pay = pick(r, ["Card", "Bank", "E-check", "Mail"]);
    const support = r() < 0.5 ? "Yes" : "No";
    const score = (tenure < 12 ? 1 : 0) + (monthly > 72 ? 1 : 0) + (contract === "Month-to-month" ? 1 : 0) + (support === "No" ? 0.6 : 0) + (r() - 0.5);
    const churn = score > 1.7 ? "Yes" : "No";
    // sprinkle a few missing values to exercise imputation
    const ageC = i % 17 === 0 ? "" : age;
    const monC = i % 23 === 0 ? "" : monthly;
    rows.push(`${ageC},${tenure},${monC},${contract},${pay},${support},${churn}`);
  }
  return rows.join("\n");
}

function irisCSV(): string {
  const r = rng(3); const rows = ["sepal_len,sepal_wid,petal_len,petal_wid,species"];
  const specs = [
    { name: "setosa", pl: [1.4, 0.25], pw: [0.25, 0.1], sl: [5.0, 0.35], sw: [3.4, 0.35] },
    { name: "versicolor", pl: [4.3, 0.45], pw: [1.3, 0.2], sl: [5.9, 0.5], sw: [2.8, 0.3] },
    { name: "virginica", pl: [5.6, 0.55], pw: [2.0, 0.27], sl: [6.6, 0.6], sw: [3.0, 0.32] },
  ];
  const gauss = () => (r() + r() + r() + r() - 2) / 2;
  for (let c = 0; c < specs.length; c++) for (let i = 0; i < 25; i++) {
    const s = specs[c];
    const v = (m: number[]) => Math.round((m[0] + gauss() * m[1]) * 10) / 10;
    rows.push(`${v(s.sl)},${v(s.sw)},${v(s.pl)},${v(s.pw)},${s.name}`);
  }
  return rows.join("\n");
}

function housingCSV(): string {
  const r = rng(11); const rows = ["rooms,house_age,income_k,distance_km,price_k"];
  const gauss = () => (r() + r() + r() - 1.5);
  for (let i = 0; i < 80; i++) {
    const rooms = 3 + Math.round(r() * 6);
    const age = 1 + Math.round(r() * 45);
    const income = Math.round((25 + r() * 110) * 10) / 10;
    const dist = Math.round((r() * 25) * 10) / 10;
    const price = Math.round((45 + rooms * 28 + income * 0.85 - age * 0.6 - dist * 1.8 + gauss() * 12) * 10) / 10;
    rows.push(`${rooms},${age},${income},${dist},${price}`);
  }
  return rows.join("\n");
}

export interface SampleDataset { key: string; label: string; task: "classification" | "regression"; target: string; csv: string; }
export function sampleDatasets(): SampleDataset[] {
  return [
    { key: "churn", label: "Telco customer churn (classification, mixed types)", task: "classification", target: "churn", csv: churnCSV() },
    { key: "iris", label: "Iris flowers (classification, numeric)", task: "classification", target: "species", csv: irisCSV() },
    { key: "housing", label: "House prices (regression)", task: "regression", target: "price_k", csv: housingCSV() },
  ];
}
