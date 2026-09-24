import { z } from "zod";
const description = z.string().min(1).max(2000);
const base64 = z
  .string()
  .min(1)
  .max(1_400_000)
  .regex(/^[A-Za-z0-9+/]*={0,2}$/);
const media = z
  .object({
    type: z.enum(["image", "audio"]),
    source: z
      .object({
        type: z.literal("base64"),
        media_type: z.enum(["image/png", "image/jpeg", "audio/wav"]),
        data: base64,
      })
      .strict(),
  })
  .strict();
const key = z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/);
const textBlock = z
  .object({ type: z.literal("text"), id: key, text: description })
  .strict();
const questionText = {
  question: description.optional(),
  question_ref: key.optional(),
};
const choice = z
  .object({ id: z.string().min(1).max(128), text: z.string().min(1).max(256) })
  .strict();
const nativeQuestion = z
  .discriminatedUnion("type", [
    z
      .object({
        type: z.literal("choice"),
        ...questionText,
        choices: z.array(choice).min(2).max(32),
      })
      .strict(),
    z
      .object({
        type: z.literal("ranking"),
        ...questionText,
        choices: z.array(choice).min(2).max(32),
      })
      .strict(),
    z.object({ type: z.literal("noul"), ...questionText }).strict(),
    z
      .object({
        type: z.literal("score"),
        ...questionText,
        levels: z
          .array(
            choice.extend({ value: z.number().finite().min(-1e9).max(1e9) }),
          )
          .min(2)
          .max(32),
      })
      .strict(),
  ])
  .refine(
    (q) => (q.question !== undefined) !== (q.question_ref !== undefined),
    "Supply exactly one question or question_ref",
  );
const nativeQuestions = z
  .record(key, nativeQuestion)
  .refine(
    (q) => Object.keys(q).length > 0 && Object.keys(q).length <= 8,
    "Use 1 to 8 questions",
  );
export const nativePayloadSchema = z
  .object({
    model: z.literal("mmso-joint-v3"),
    input: z
      .array(z.union([media, textBlock]))
      .min(2)
      .max(10),
    questions: nativeQuestions,
    abstain_threshold: z.number().min(0).max(1).default(0),
  })
  .strict()
  .superRefine((payload, ctx) => {
    for (const kind of ["image", "audio"]) {
      if (payload.input.filter((b) => b.type === kind).length !== 1)
        ctx.addIssue({
          code: "custom",
          path: ["input"],
          message: `Supply exactly one ${kind} block`,
        });
    }
    const texts = payload.input.filter((b) => b.type === "text");
    const ids = new Set(texts.map((t) => t.id));
    const refs = new Set(
      Object.values(payload.questions).flatMap((q) =>
        q.question_ref ? [q.question_ref] : [],
      ),
    );
    if (ids.size !== texts.length)
      ctx.addIssue({
        code: "custom",
        path: ["input"],
        message: "Text block IDs must be unique",
      });
    if (
      [...refs].some((id) => !ids.has(id)) ||
      [...ids].some((id) => !refs.has(id))
    )
      ctx.addIssue({
        code: "custom",
        path: ["input"],
        message:
          "Every text block must be referenced and every question_ref must resolve; free context is not supported by this checkpoint",
      });
  });
export type NativePayload = z.infer<typeof nativePayloadSchema>;

// Visible instruction fields become actual text blocks consumed by the native
// Python API. No text is silently discarded or sent through another model.
export function makeNativePayload(
  mediaInput: unknown[],
  inlineQuestions: unknown,
): NativePayload {
  const parsed = nativeQuestions.parse(inlineQuestions);
  const textInputs: z.infer<typeof textBlock>[] = [];
  const questions = Object.fromEntries(
    Object.entries(parsed).map(([id, q]) => {
      if (typeof q.question !== "string")
        throw new Error("Each playground instruction needs its question text.");
      const { question, ...rest } = q;
      textInputs.push({ type: "text", id, text: question });
      return [id, { ...rest, question_ref: id }];
    }),
  );
  return nativePayloadSchema.parse({
    model: "mmso-joint-v3",
    input: [...mediaInput, ...textInputs],
    questions,
    abstain_threshold: 0,
  });
}
export function resolveNativeQuestions(payload: NativePayload) {
  const texts = new Map(
    payload.input.filter((b) => b.type === "text").map((b) => [b.id, b.text]),
  );
  return Object.fromEntries(
    Object.entries(payload.questions).map(([id, q]) => {
      const { question_ref, ...rest } = q;
      const question = q.question ?? texts.get(question_ref!);
      if (!question) throw new Error("Saved text input is missing.");
      return [id, { ...rest, question }];
    }),
  );
}
export const runInputSchema = z
  .object({
    provider: z.literal("miso"),
    label: z.string().max(80),
    payload: nativePayloadSchema,
  })
  .strict();
export function isMisoResult(value: unknown): value is RunResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "provider" in value &&
    value.provider === "miso" &&
    "model" in value &&
    value.model === "mmso-joint-v3"
  );
}
export type RunInput = z.infer<typeof runInputSchema>;
export type Answer = {
  id: string;
  type: string;
  value: string | number | null;
  confidence: number | null;
  abstained: boolean;
  probabilities: { label: string; probability: number }[];
};
export type RunResult = {
  request?: RunInput;
  provider: "miso";
  label: string;
  model: string;
  completedAt: string;
  inferenceMs: number;
  checkpoint: string | null;
  answers: Answer[];
  raw: unknown;
};
export type RunState = {
  id: string;
  status: string;
  result?: RunResult;
  error?: string;
};
export const MAX_BODY = 2_900_000;
