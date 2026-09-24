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
  const native = input.provider === "miso";
  const endpoint = native
    ? `${process.env.MISO_BACKEND_URL}/v1/decisions`
    : "https://api.typesafe.ai/v1/systemone";
  const key = native
    ? process.env.MISO_BACKEND_KEY
    : process.env.TYPESAFE_API_KEY;
  if (!key || (native && !process.env.MISO_BACKEND_URL))
    throw new FatalError("This model connection is not configured.");
  const started = performance.now();
  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify(
        native ? input.payload : { model: "jev-1.13.0", ...input.payload },
      ),
      signal: AbortSignal.timeout(60_000),
      redirect: "error",
    });
  } catch {
    throw new FatalError(
      "The model request did not complete. Its cost may be unknown; it will not be retried automatically.",
    );
  }
  const raw = await response.json();
  if (!response.ok) {
    const detail = native
      ? (raw?.error?.message ?? "MiSO rejected this input.")
      : "Jev could not complete this request.";
    throw new FatalError(`${detail} (HTTP ${response.status})`);
  }
  const inferenceMs = performance.now() - started;
  if (raw.model !== (native ? "mmso-joint-v3" : "jev-1.13.0"))
    throw new FatalError(
      "The model identity did not match the requested version.",
    );
  const source = native ? raw.results : raw.answers;
  if (
    !source ||
    Object.keys(source).length !== Object.keys(input.payload.questions).length
  )
    throw new FatalError("The model returned an incomplete answer set.");
  const answers: Answer[] = Object.entries(input.payload.questions).map(
    ([id, question]) => {
      const answer = source[id];
      if (!answer || answer.type !== question.type)
        throw new FatalError("An answer did not match its question.");
      let probabilities: Record<string, number> = answer.probabilities;
      if (!native && question.type === "noul")
        probabilities = { yes: answer.noul, no: 1 - answer.noul };
      if (
        !probabilities ||
        Object.values(probabilities).some(
          (p) => typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1,
        ) ||
        Math.abs(Object.values(probabilities).reduce((a, b) => a + b, 0) - 1) >
          0.00001
      )
        throw new FatalError("The model returned invalid probabilities.");
      const expected = native
        ? "choices" in question
          ? question.choices.map((c: { id: string }) => c.id)
          : "levels" in question
            ? question.levels.map((c: { id: string }) => c.id)
            : ["false", "true"]
        : question.type === "noul"
          ? ["yes", "no"]
          : question.type === "score"
            ? (question.criteria as string[]).map((_, i) => String(i))
            : Object.keys(question.criteria);
      if (
        expected.length !== Object.keys(probabilities).length ||
        expected.some((key: string) => !(key in probabilities))
      )
        throw new FatalError(
          "Answer options did not match the supplied choices.",
        );
      return {
        id,
        type: question.type,
        value: native
          ? question.type === "score"
            ? answer.value
            : question.type === "noul"
              ? answer.probability_true
              : (("choices" in question
                  ? question.choices.find(
                      (c: { id: string; text: string }) =>
                        c.id === answer.decision,
                    )?.text
                  : null) ?? answer.decision)
          : question.type === "score"
            ? answer.score
            : question.type === "noul"
              ? answer.noul
              : answer.choice,
        confidence: answer.confidence ?? null,
        abstained: answer.abstained ?? false,
        probabilities: Object.entries(probabilities)
          .map(([k, p]) => {
            let label = k;
            if (native && "choices" in question)
              label =
                question.choices.find(
                  (c: { id: string; text: string }) => c.id === k,
                )?.text ?? k;
            if (native && "levels" in question)
              label =
                question.levels.find(
                  (c: { id: string; text: string }) => c.id === k,
                )?.text ?? k;
            if (!native && question.type === "score" && "criteria" in question)
              label = (question.criteria as string[])[Number(k)] ?? k;
            return { label, probability: p };
          })
          .sort((a, b) => b.probability - a.probability),
      };
    },
  );
  const inputTokens =
    !native &&
    Number.isInteger(raw.usage?.input_tokens) &&
    raw.usage.input_tokens >= 0
      ? raw.usage.input_tokens
      : null;
  return {
    request: input,
    provider: input.provider,
    label: input.label,
    model: raw.model,
    completedAt: new Date().toISOString(),
    inferenceMs,
    inputTokens,
    costUsd: inputTokens === null ? null : (inputTokens * 0.042) / 1e6,
    checkpoint: raw.checkpoint_sha256 ?? null,
    answers,
    raw,
  };
}
evaluateModel.maxRetries = 0;
