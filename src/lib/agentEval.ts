// Confidence scoring engine and comparative synthesis for Agent Lab.

export interface ConfidenceMetrics {
  score: number; // 0 to 100
  grounding: number; // 0 to 100: how much of the answer is backed by tool observations
  toolReliability: number; // 0 to 100: tool execution success rate & relevance
  reasoningConsistency: number; // 0 to 100: thought-action loop convergence quality
  factualDensity: number; // 0 to 100: concrete facts/numbers vs vague filler
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
}): ConfidenceMetrics {
  const { finalAnswer, trace, iterations, maxIters, outcome = "success" } = params;

  if (!finalAnswer || finalAnswer.trim().length === 0 || outcome === "error") {
    return {
      score: 15,
      grounding: 10,
      toolReliability: 10,
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

  let grounding = 70;
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
    // Pure reasoning without tools
    grounding = finalAnswer.length > 100 ? 78 : 65;
  }

  // 2. Tool Reliability: successful executions vs errors
  const actionCount = trace.filter((t) => t.kind === "action").length;
  let toolReliability = 90;
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

  // Weighted Total
  const rawScore = Math.round(
    grounding * 0.35 +
    toolReliability * 0.25 +
    reasoningConsistency * 0.25 +
    factualDensity * 0.15
  );
  const score = Math.max(10, Math.min(99, rawScore));

  let label: ConfidenceMetrics["label"] = "High Confidence";
  let explanation = "Strong factual grounding and clean reasoning progression.";
  if (score >= 82) {
    label = "High Confidence";
    explanation = `High certainty (${score}%). Output is directly grounded in ${observations.length} tool observation(s) and structured clearly.`;
  } else if (score >= 65) {
    label = "Moderate Confidence";
    explanation = `Moderate certainty (${score}%). Reasoning resolved successfully, though some details rely on internal model priors.`;
  } else if (score >= 45) {
    label = "Low Confidence";
    explanation = `Low certainty (${score}%). Required extensive step iterations or had partial tool execution friction.`;
  } else {
    label = "Uncertain";
    explanation = `Uncertain (${score}%). The agent approached step limits or faced tool execution issues.`;
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
