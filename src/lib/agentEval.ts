// Confidence scoring engine and comparative synthesis for Agent Lab.

export interface ConfidenceMetrics {
  score: number; // 0 to 100
  // `null` means "not measurable for this run" — e.g. tool reliability when the
  // agent had no tools. Reporting a default here reads as a measurement and is
  // how this gauge used to claim 90% tool reliability on runs with zero tools.
  grounding: number | null; // how much of the answer is backed by tool observations
  toolReliability: number | null; // tool execution success rate
  reasoningConsistency: number; // thought-action loop convergence quality
  factualDensity: number; // concrete facts/numbers vs vague filler
  label: "High Confidence" | "Moderate Confidence" | "Low Confidence" | "Uncertain";
  explanation: string;
}

export interface ComparisonResult {
  winner: "A" | "B" | "Tie";
  winnerReason: string;
  agreementScore: number; // 0-100% semantic alignment
  keyDifferences: string[];
  synthesizedAnswer: string;
}

/**
 * Computes a grounded confidence score based on the agent's reasoning trace,
 * tool observations, iteration efficiency, and final output structure.
 */
export function computeConfidenceScore(params: {
  finalAnswer: string;
  trace: Array<{ kind: string; text?: string; tool?: string; state?: string }>;
  iterations: number;
  maxIters: number;
  outcome?: string;
  task: string;
  /** For a pipeline composite: how many stages failed, out of how many. */
  degradedSteps?: number;
  totalSteps?: number;
}): ConfidenceMetrics {
  const { finalAnswer, trace, iterations, maxIters, outcome = "success" } = params;

  if (!finalAnswer || finalAnswer.trim().length === 0 || outcome === "error") {
    return {
      score: 15,
      grounding: null,
      toolReliability: null,
      reasoningConsistency: 15,
      factualDensity: 20,
      label: "Uncertain",
      explanation: "The run encountered errors or produced no final response.",
    };
  }

  // 1. Tool Grounding: Check if key facts/entities/numbers in observations appear in final answer
  const observations = trace.filter((t) => t.kind === "observation").map((t) => t.text || "");
  const allObsText = observations.join(" ").toLowerCase();
  const obsWords = new Set(
    allObsText
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length > 3 && !COMMON_STOPWORDS.has(w))
  );

  // Grounding is only meaningful when there are observations to ground against.
  let grounding: number | null = null;
  if (observations.length > 0) {
    const finalLower = finalAnswer.toLowerCase();
    let matches = 0;
    const obsArr = Array.from(obsWords);
    if (obsArr.length > 0) {
      for (const word of obsArr) {
        if (finalLower.includes(word)) matches++;
      }
      grounding = Math.min(100, Math.max(30, Math.round((matches / Math.min(obsArr.length, 15)) * 100)));
    } else {
      grounding = 75;
    }
  } else {
    // Pure reasoning, nothing to check the answer against.
    grounding = null;
  }

  // 2. Tool Reliability: successful executions vs errors
  const actionCount = trace.filter((t) => t.kind === "action").length;
  let toolReliability: number | null = null;
  if (actionCount > 0) {
    const errorObs = observations.filter((o) => /error|unknown tool|failed|invalid/i.test(o)).length;
    toolReliability = Math.max(20, Math.round(((actionCount - errorObs) / actionCount) * 100));
  }

  // 3. Reasoning Consistency: did it solve in reasonable steps without stalling?
  let reasoningConsistency = 88;
  if (iterations >= maxIters && outcome === "max_iters") {
    reasoningConsistency = 42;
  } else if (iterations > maxIters * 0.8) {
    reasoningConsistency = 68;
  } else if (iterations <= 3 && actionCount > 0) {
    reasoningConsistency = 95;
  }

  // 4. Factual Density: numbers, formatting, bullet points, specifics vs generic text
  const numbersCount = (finalAnswer.match(/\b\d+(\.\d+)?%?\b/g) || []).length;
  const structureBonus = finalAnswer.includes("\n") || finalAnswer.includes("•") || finalAnswer.includes("-") ? 15 : 0;
  const factualDensity = Math.min(98, Math.max(35, Math.round(numbersCount * 6 + structureBonus + Math.min(finalAnswer.length / 25, 45))));

  // Weighted total over the dimensions we could actually measure. Unmeasured
  // dimensions are dropped and the remaining weights renormalised, so a
  // tool-less run is scored on what it did rather than padded with defaults.
  const dims: [number | null, number][] = [
    [grounding, 0.35],
    [toolReliability, 0.25],
    [reasoningConsistency, 0.25],
    [factualDensity, 0.15],
  ];
  const measured = dims.filter((d): d is [number, number] => d[0] !== null);
  const totalWeight = measured.reduce((a, [, w]) => a + w, 0);
  const rawScore = Math.round(measured.reduce((a, [v, w]) => a + v * w, 0) / totalWeight);
  const score = Math.max(10, Math.min(99, rawScore));
  const unmeasured = [
    grounding === null ? "grounding" : null,
    toolReliability === null ? "tool reliability" : null,
  ].filter(Boolean) as string[];
  const caveat = unmeasured.length
    ? ` Not measured: ${unmeasured.join(" and ")} — this step ran no tools, so the score reflects reasoning and output structure only.`
    : "";

  // A run that hit the reasoning limit never produced a final answer. It may still
  // score well on grounding and density (it did call tools, it did emit text), so
  // cap it explicitly — otherwise a failed step reports "High Confidence".
  if (outcome === "max_iters") {
    return {
      score: Math.min(score, 40),
      grounding, toolReliability, reasoningConsistency, factualDensity,
      label: "Uncertain",
      explanation: `Did not converge (${iterations}/${maxIters} reasoning steps used) — the agent ran out of steps before producing a final answer, so this output is partial.${caveat}`,
    };
  }

  let label: ConfidenceMetrics["label"] = "High Confidence";
  let explanation = "Strong factual grounding and clean reasoning progression.";
  if (score >= 82) {
    label = "High Confidence";
    explanation = `High certainty (${score}%). Output is grounded in ${observations.length} tool observation(s) and structured clearly.${caveat}`;
  } else if (score >= 65) {
    label = "Moderate Confidence";
    explanation = `Moderate certainty (${score}%). Reasoning resolved successfully, though some details rely on internal model priors.${caveat}`;
  } else if (score >= 45) {
    label = "Low Confidence";
    explanation = `Low certainty (${score}%). Required extensive step iterations or had partial tool execution friction.${caveat}`;
  } else {
    label = "Uncertain";
    explanation = `Uncertain (${score}%). The agent approached step limits or faced tool execution issues.${caveat}`;
  }

  // A pipeline whose stages failed cannot be reported at full confidence just
  // because its last stage returned text — the last stage was working from
  // "[UNAVAILABLE]" inputs and may have filled the gap itself.
  const { degradedSteps = 0, totalSteps = 0 } = params;
  if (degradedSteps > 0 && totalSteps > 0) {
    const cap = Math.max(10, Math.round(100 * (1 - degradedSteps / totalSteps)));
    if (cap < score) {
      const capped = cap;
      const cappedLabel: ConfidenceMetrics["label"] =
        capped >= 82 ? "High Confidence" : capped >= 65 ? "Moderate Confidence" : capped >= 45 ? "Low Confidence" : "Uncertain";
      return {
        score: capped, grounding, toolReliability, reasoningConsistency, factualDensity,
        label: cappedLabel,
        explanation: `Capped at ${capped}% — ${degradedSteps} of ${totalSteps} stages failed, so downstream agents worked from incomplete input. Treat unsourced claims in the final output as unverified.${caveat}`,
      };
    }
  }

  return {
    score,
    grounding,
    toolReliability,
    reasoningConsistency,
    factualDensity,
    label,
    explanation,
  };
}

/**
 * Synthesizes two agent outputs and provides a clear comparison verdict.
 */
export function synthesizeComparison(params: {
  task: string;
  answerA: string;
  scoreA: ConfidenceMetrics;
  modelA: string;
  answerB: string;
  scoreB: ConfidenceMetrics;
  modelB: string;
}): ComparisonResult {
  const { answerA, scoreA, modelA, answerB, scoreB, modelB } = params;

  // Semantic similarity estimate by word overlap
  const wordsA = new Set(answerA.toLowerCase().split(/\W+/).filter((w) => w.length > 3));
  const wordsB = new Set(answerB.toLowerCase().split(/\W+/).filter((w) => w.length > 3));
  let common = 0;
  wordsA.forEach((w) => { if (wordsB.has(w)) common++; });
  const agreementScore = Math.round((common / Math.max(1, Math.max(wordsA.size, wordsB.size))) * 100);

  let winner: "A" | "B" | "Tie" = "Tie";
  let winnerReason = "Both models produced comparable and well-grounded answers.";

  if (scoreA.score > scoreB.score + 5) {
    winner = "A";
    winnerReason = `Variant A (${modelA}) achieved higher confidence (${scoreA.score}% vs ${scoreB.score}%) with stronger factual grounding (${scoreA.grounding}% vs ${scoreB.grounding}%).`;
  } else if (scoreB.score > scoreA.score + 5) {
    winner = "B";
    winnerReason = `Variant B (${modelB}) achieved higher confidence (${scoreB.score}% vs ${scoreA.score}%) with better tool reliability and structure.`;
  } else {
    winner = answerA.length >= answerB.length ? "A" : "B";
    winnerReason = `Both variants performed with similar confidence (${scoreA.score}% ≈ ${scoreB.score}%). Variant ${winner} provides a slightly more comprehensive response.`;
  }

  const keyDifferences: string[] = [
    `Agreement Level: ${agreementScore}% semantic topic alignment on the task.`,
    `Variant A Confidence: ${scoreA.score}% (${scoreA.label}) · Grounding: ${scoreA.grounding}%`,
    `Variant B Confidence: ${scoreB.score}% (${scoreB.label}) · Grounding: ${scoreB.grounding}%`,
    scoreA.score !== scoreB.score
      ? `Variant ${scoreA.score > scoreB.score ? "A" : "B"} had +${Math.abs(scoreA.score - scoreB.score)}% higher confidence margin.`
      : "Identical overall confidence metrics.",
  ];

  // Synthesize best elements
  const preferredAnswer = winner === "A" ? answerA : answerB;
  const secondaryAnswer = winner === "A" ? answerB : answerA;
  const synthesizedAnswer = preferredAnswer.length > 30
    ? preferredAnswer
    : `${preferredAnswer}\n\n*Alternative perspective:*\n${secondaryAnswer}`;

  return {
    winner,
    winnerReason,
    agreementScore,
    keyDifferences,
    synthesizedAnswer,
  };
}

const COMMON_STOPWORDS = new Set([
  "that", "this", "with", "from", "have", "were", "been", "they", "their", "which",
  "about", "would", "there", "what", "when", "where", "will", "more", "also", "into",
  "some", "them", "these", "than", "then", "only", "other", "such", "most", "over",
]);
