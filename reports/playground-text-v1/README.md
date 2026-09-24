# Native text input restored in the playground

The [playground](https://miso-playground-nine.vercel.app) now exposes **audio + image + text** together. Text instructions are visible editable fields and are transmitted as native `text` content blocks, bound to the output questions using `question_ref`. The original question encoder and trained checkpoint consume them; no Jev call, transcript conversion or alternate model is inserted in the MiSO path.

The bug was in the playground contract: its input array accepted exactly two media blocks, rejecting a valid request that the Python API already supported. [The captured production failure](before.json) was HTTP 400. The corrected gateway accepts image/audio/text blocks, validates that every text block is consumed, and rejects duplicate IDs, missing references and unreferenced text. Historical runs with inline question text remain readable.

## Verified behavior

The [production comparison](verification.json) held the image, audio and answer schemas fixed and changed only the color instruction:

| Text instruction | MiSO color decision |
|---|---|
| what color marks the spoken command | not present |
| what color marks the opposite of the spoken command | red |

The largest color-probability change was 0.9309758767. Every probability for the other three questions was identical. Both runs used checkpoint `4284d30920615018b500716d406236d919f5e38e268fdc5490c6f559af612bfa`; the report records the shared image/audio hashes and run URLs. This demonstrates that changing the supplied native text affects the neural decision, rather than being discarded by the interface.

The [exact original three-block request](original-request-fixed.json) now passes the gateway and completes. A browser test also changed the visible text field and received the updated native decision. The production interface exposes the third modality and restores saved text from content blocks. All 91 Python tests, seven playground contract tests and the Next.js production build passed.

This is an interface/transport correction using the existing model. V3 still understands its learned question vocabulary and generated-panel task; arbitrary text documents/context and broad language understanding require the planned successor. The checkpoint and its accuracy benchmarks are unchanged.
