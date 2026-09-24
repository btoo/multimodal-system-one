import { test } from "node:test";
import assert from "node:assert/strict";
import { issueSession, validSession, equalSecret } from "../lib/tokens.ts";
import { runInputSchema } from "../lib/contracts.ts";
import { boundedJSON } from "../lib/body.ts";

test("owner sessions reject tampering, expiry and missing configuration", () => {
  const secret = "a-private-test-secret";
  const now = 1_000_000;
  const token = issueSession(secret, now);
  assert.equal(validSession(token, secret, now), true);
  assert.equal(validSession(token + "x", secret, now), false);
  assert.equal(validSession(token, "wrong", now), false);
  assert.equal(validSession(token, secret, now + 8 * 86400000), false);
  assert.equal(validSession(token, undefined, now), false);
  assert.equal(equalSecret("short", "longer"), false);
});
test("bounded request reader rejects oversized streamed data", async () => {
  const request = new Request("https://miso.example", {
    method: "POST",
    body: JSON.stringify({ value: "x".repeat(1000) }),
  });
  await assert.rejects(() => boundedJSON(request, 100));
});
test("provider identity and input contracts fail before any paid call", () => {
  const text = {
    provider: "jev",
    label: "test",
    payload: {
      state: "hello",
      questions: { q: { type: "noul", instructions: "Is this a greeting?" } },
    },
  };
  assert.equal(runInputSchema.safeParse(text).success, false);
  assert.equal(
    runInputSchema.safeParse({
      ...text,
      payload: { ...text.payload, model: "unapproved" },
    }).success,
    false,
  );
  assert.equal(
    runInputSchema.safeParse({
      ...text,
      payload: { ...text.payload, state: "x".repeat(12001) },
    }).success,
    false,
  );
  assert.equal(
    runInputSchema.safeParse({ ...text, provider: "miso" }).success,
    false,
  );
  const block = {
    type: "image",
    source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
  };
  assert.equal(
    runInputSchema.safeParse({
      provider: "miso",
      label: "test",
      payload: {
        model: "mmso-joint-v3",
        input: [block, block],
        questions: {
          q: { type: "noul", question: "is the spoken command on the screen" },
        },
      },
    }).success,
    false,
  );
});

import { sameOrigin } from "../lib/origin.ts";
test("origin validation handles Next localhost normalization and rejects foreign origins", () => {
  const request = new Request("http://localhost:3047/api/session", {
    headers: { host: "127.0.0.1:3047", origin: "http://127.0.0.1:3047" },
  });
  assert.equal(sameOrigin(request), true);
  assert.equal(
    sameOrigin(
      new Request("https://miso.example/api", {
        headers: { host: "miso.example", origin: "https://evil.example" },
      }),
    ),
    false,
  );
  assert.equal(sameOrigin(new Request("https://miso.example/api")), false);
});

import {
  makeNativePayload,
  resolveNativeQuestions,
  nativePayloadSchema,
} from "../lib/contracts.ts";
const mediaFixture = [
  {
    type: "image",
    source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
  },
  {
    type: "audio",
    source: { type: "base64", media_type: "audio/wav", data: "aGVsbG8=" },
  },
];
test("native text is transported as a third input modality and restored exactly", () => {
  const original = {
    color: {
      type: "choice",
      question: "what color marks the spoken command",
      choices: [
        { id: "r", text: "red" },
        { id: "g", text: "green" },
      ],
    },
  };
  const payload = makeNativePayload(mediaFixture, original);
  assert.deepEqual(
    payload.input.map((b) => b.type),
    ["image", "audio", "text"],
  );
  assert.deepEqual(payload.input[2], {
    type: "text",
    id: "color",
    text: original.color.question,
  });
  assert.equal(payload.questions.color.question_ref, "color");
  assert.equal(payload.questions.color.question, undefined);
  assert.deepEqual(resolveNativeQuestions(payload), original);
  assert.equal(
    runInputSchema.safeParse({
      provider: "miso",
      label: "Three modalities",
      payload,
    }).success,
    true,
  );
});
test("the native gateway cannot drop text or accept dangling references", () => {
  const body = makeNativePayload(mediaFixture, {
    present: { type: "noul", question: "is the spoken command on the screen" },
  });
  assert.equal(
    nativePayloadSchema.safeParse({ ...body, input: body.input.slice(0, 2) })
      .success,
    false,
  );
  assert.equal(
    nativePayloadSchema.safeParse({
      ...body,
      input: [...body.input, { type: "text", id: "context", text: "red" }],
    }).success,
    false,
  );
  assert.equal(
    nativePayloadSchema.safeParse({
      ...body,
      input: [...body.input, body.input[2]],
    }).success,
    false,
  );
  assert.equal(
    nativePayloadSchema.safeParse({
      ...body,
      questions: {
        present: {
          type: "noul",
          question: "is the spoken command on the screen",
          question_ref: "present",
        },
      },
    }).success,
    false,
  );
});
test("changing only text preserves the raw media and answer schemas", () => {
  const first = makeNativePayload(mediaFixture, {
    color: {
      type: "choice",
      question: "what color marks the spoken command",
      choices: [
        { id: "r", text: "red" },
        { id: "g", text: "green" },
      ],
    },
  });
  const second = structuredClone(first);
  if (second.input[2].type === "text")
    second.input[2].text =
      "what color marks the opposite of the spoken command";
  assert.equal(nativePayloadSchema.safeParse(second).success, true);
  assert.deepEqual(first.input.slice(0, 2), second.input.slice(0, 2));
  assert.deepEqual(first.questions, second.questions);
  assert.notDeepEqual(first.input[2], second.input[2]);
});

import {isMisoResult} from '../lib/contracts.ts';
import {misoHistory} from '../lib/history.ts';
test('non-MiSO providers cannot start or render in the playground',()=>{
 const payload=makeNativePayload(mediaFixture,{present:{type:'noul',question:'is the spoken command on the screen'}});
 assert.equal(runInputSchema.safeParse({provider:'miso',label:'Native',payload}).success,true);
 for(const provider of ['jev','openai','other'])assert.equal(runInputSchema.safeParse({provider,label:'No',payload}).success,false);
 assert.equal(isMisoResult({provider:'miso',model:'mmso-joint-v3'}),true);
 assert.equal(isMisoResult({provider:'jev',model:'jev-1.13.0'}),false);
 assert.equal(isMisoResult({provider:'miso',model:'jev-1.13.0'}),false);
});
test('run history filters historical comparison entries',()=>{
 const native={id:'wrun_native',provider:'miso',label:'Native',createdAt:'2026-09-24'};
 const comparison={id:'wrun_comparison',provider:'jev',label:'Comparison',createdAt:'2026-09-24'};
 assert.deepEqual(misoHistory([native,comparison,null,{}]),[native]);
 assert.deepEqual(misoHistory({}),[]);
});
