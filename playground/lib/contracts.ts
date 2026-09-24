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
const choice = z
  .object({ id: z.string().min(1).max(128), text: z.string().min(1).max(256) })
  .strict();
const nativeQuestion = z.discriminatedUnion("type", [
  z
    .object({
      type: z.literal("choice"),
      question: description,
      choices: z.array(choice).min(2).max(32),
    })
    .strict(),
  z
    .object({
      type: z.literal("ranking"),
      question: description,
      choices: z.array(choice).min(2).max(32),
    })
    .strict(),
  z.object({ type: z.literal("noul"), question: description }).strict(),
  z
    .object({
      type: z.literal("score"),
      question: description,
      levels: z
        .array(choice.extend({ value: z.number().finite().min(-1e9).max(1e9) }))
        .min(2)
        .max(32),
    })
    .strict(),
]);
const key = z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/);
const nativeQuestions = z
  .record(key, nativeQuestion)
  .refine(
    (q) => Object.keys(q).length > 0 && Object.keys(q).length <= 8,
    "Use 1 to 8 questions",
  );
const jevQuestion = z.discriminatedUnion("type", [
  z.object({ type: z.literal("noul"), instructions: description }).strict(),
  z
    .object({
      type: z.literal("choice"),
      instructions: description,
      criteria: z
        .record(z.string().min(1).max(64), description)
        .refine(
          (c) => Object.keys(c).length >= 2 && Object.keys(c).length <= 32,
        ),
    })
    .strict(),
  z
    .object({
      type: z.literal("score"),
      instructions: description,
      criteria: z.array(description).min(2).max(10),
    })
    .strict(),
]);
export const runInputSchema = z.discriminatedUnion("provider", [
  z
    .object({
      provider: z.literal("miso"),
      label: z.string().max(80),
      payload: z
        .object({
          model: z.literal("mmso-joint-v3"),
          input: z
            .array(media)
            .length(2)
            .refine(
              (x) =>
                x.filter((b) => b.type === "image").length === 1 &&
                x.filter((b) => b.type === "audio").length === 1,
            ),
          questions: nativeQuestions,
          abstain_threshold: z.number().min(0).max(1).default(0),
        })
        .strict(),
    })
    .strict(),
  z
    .object({
      provider: z.literal("jev"),
      label: z.string().max(80),
      payload: z
        .object({
          state: z.string().min(1).max(12000),
          questions: z
            .record(key, jevQuestion)
            .refine(
              (q) => Object.keys(q).length > 0 && Object.keys(q).length <= 8,
            ),
        })
        .strict(),
    })
    .strict(),
]);
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
  provider: "miso" | "jev";
  label: string;
  model: string;
  completedAt: string;
  inferenceMs: number;
  inputTokens: number | null;
  costUsd: number | null;
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
