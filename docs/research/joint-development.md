# Joint-model development record

## Attempt 1: fixed pairs

The first model uses one acoustic encoder, a shared visual encoder over four fixed panel quadrants, a learned text encoder, two shared fusion blocks, and one scalar candidate scorer. All components are trainable. The eight-class acoustic output head is removed; no scene metadata, transcript, task ID, or candidate ID is a neural input.

A training-only 64-example check fit perfectly; its weights were discarded. Thirty-five unit/integration checks passed, including CPU/MPS agreement, native-modality gradients, candidate permutation/subsetting, question isolation, and oracle counterfactuals.

The full run began overfitting: training loss fell while development accuracy stayed near the roughly 50% missing-target prior. A structural data audit found all 2,048 training recordings had only one associated panel, and all 2,048 panels had only one associated spoken word. This permits sample-pair memorization; it does not prove the model learned to connect acoustic content to visual meaning.

## Preregistered follow-up: break fixed pairings

Keep the architecture, optimizer, input data pool, and output semantics fixed. During each training batch, select a recorded keyword independently of the panel and choose a training recording for that word. Recompute the target using the independent training oracle. Both one-to-one associations are broken. Development and calibration remain fixed; final metrics remain closed until nomination.

This is one targeted training-data change. Any improvement will be evaluated on development first and then against separately trained unimodal controls under the same augmented sampling stream. The first attempt is retained as a development result and is not silently replaced.
