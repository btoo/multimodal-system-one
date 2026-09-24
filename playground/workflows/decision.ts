import { FatalError } from "workflow";
import type { RunInput, RunResult, Answer } from "../lib/contracts";
export async function decisionRun(input: RunInput): Promise<RunResult> {
  "use workflow";
  return await evaluateModel(input);
}
export async function evaluateModel(input: RunInput): Promise<RunResult> {
  "use step";
  const { runInputSchema } = await import("../lib/contracts");
  input = runInputSchema.parse(input);
  const endpoint = process.env.MISO_BACKEND_URL,
    key = process.env.MISO_BACKEND_KEY;
  if (!endpoint || !key)
    throw new FatalError("The MiSO connection is not configured.");
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(`${endpoint}/v1/decisions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(input.payload),
      signal: AbortSignal.timeout(60_000),
      redirect: "error",
    });
  } catch {
    throw new FatalError(
      "The MiSO request did not complete. It will not be retried automatically.",
    );
  }
  const raw = await response.json();
  if (!response.ok)
    throw new FatalError(
      `${raw?.error?.message ?? "MiSO rejected this input."} (HTTP ${response.status})`,
    );
  const inferenceMs = performance.now() - started;
  if (raw.model !== "mmso-joint-v3")
    throw new FatalError(
      "The model identity did not match the requested MiSO version.",
    );
  const source = raw.results;
  if (
    !source ||
    Object.keys(source).length !== Object.keys(input.payload.questions).length
  )
    throw new FatalError("MiSO returned an incomplete answer set.");
  const answers: Answer[] = Object.entries(input.payload.questions).map(
    ([id, question]) => {
      const answer = source[id];
      if (!answer || answer.type !== question.type)
        throw new FatalError("An answer did not match its question.");
      const probabilities: Record<string, number> = answer.probabilities;
      if (
        !probabilities ||
        Object.values(probabilities).some(
          (p) => typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1,
        ) ||
        Math.abs(Object.values(probabilities).reduce((a, b) => a + b, 0) - 1) >
          1e-5
      )
        throw new FatalError("MiSO returned invalid probabilities.");
      const expected =
        "choices" in question
          ? question.choices.map((c) => c.id)
          : "levels" in question
            ? question.levels.map((c) => c.id)
            : ["false", "true"];
      if (
        expected.length !== Object.keys(probabilities).length ||
        expected.some((k) => !(k in probabilities))
      )
        throw new FatalError(
          "Answer options did not match the supplied choices.",
        );
      return {
        id,
        type: question.type,
        value:
          question.type === "score"
            ? answer.value
            : question.type === "noul"
              ? answer.probability_true
              : (("choices" in question
                  ? question.choices.find((c) => c.id === answer.decision)?.text
                  : null) ?? answer.decision),
        confidence: answer.confidence ?? null,
        abstained: answer.abstained ?? false,
        probabilities: Object.entries(probabilities)
          .map(([k, probability]) => ({
            label:
              "choices" in question
                ? (question.choices.find((c) => c.id === k)?.text ?? k)
                : "levels" in question
                  ? (question.levels.find((c) => c.id === k)?.text ?? k)
                  : k,
            probability,
          }))
          .sort((a, b) => b.probability - a.probability),
      };
    },
  );
  return {
    request: input,
    provider: "miso",
    label: input.label,
    model: raw.model,
    completedAt: new Date().toISOString(),
    inferenceMs,
    checkpoint: raw.checkpoint_sha256 ?? null,
    answers,
    raw,
  };
}
evaluateModel.maxRetries = 0;
