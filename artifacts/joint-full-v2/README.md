# Native Decision v1 — selected paired-panel checkpoint

**668,097 parameters.** One jointly trained model takes audio, pixels, a question, and supplied candidate descriptions. It returns candidate probabilities through a shared scalar scoring head. This is a limited research prototype, not a general computer-use model or a reproduction of Jev's unpublished training algorithm.

## Inputs and outputs

- Audio: PCM16 WAV, resampled to 16 kHz, padded/truncated to one second; eight keyword concepts.
- Image: RGB, resized to 128×128; generated 2×2 symbol panels, represented by four fixed quadrants.
- Language: learned 43-token vocabulary including special tokens; known question families with tested rephrasings. Unknown tokens and silent text truncation are rejected.
- Candidates: two or more distinct descriptions, with opaque IDs. Training uses 2, 5, or 8 candidates; subsets of 2–4 were checked. The interface caps the list at 32; that cap is not a quality guarantee.
- Output: Choice-style distribution over supplied IDs. Boolean questions are expressed using yes/no candidates. A caller-selected threshold can abstain; arbitrary-input rejection has not been trained.

`confidence` is the maximum adjusted candidate probability, not a separately learned epistemic-confidence estimator.

The runtime does not use transcripts, scene labels, task IDs, or the symbolic oracle. The acoustic representation is initialized from the earlier small speech checkpoint with its class head removed; every component is then fine-tuned jointly. Loss is supervised NLL, with temperature scaling fit after development selection.

## Measured scope

Six-family held-out joint accuracy is **72.92%**, compared with **50.00%** and **49.83%** for matched unimodal controls. The held command–color combinations fail at **45.66%**. New wording and moved controls score approximately 73%. One training seed was used; these numbers do not establish real-browser or open-world generalization.

The failed first development attempt is retained. This checkpoint was nominated before final metrics were opened. See the [full report](../../reports/joint-v2/README.md), [nomination](../../evals/joint-nomination-v2.json), [training record](../../reports/joint-full-v2/training.json), and [evaluation](../../reports/joint-full-v2/evaluation.json).

`model.safetensors` contains all neural weights. `config.json` contains architecture, vocabulary, provenance of acoustic initialization, and fitted temperature. The training record preserves the pre-calibration configuration; final calibration is recorded separately.

## Run

From the repository root:

```bash
uv sync --locked
uv run mmso prepare speech_keywords
uv run mmso joint-prepare
uv run python scripts/run_joint_demo.py
```

Training audio is from the pinned Mini Speech Commands source; generated panels are original project data. See [acquisition provenance](../../evals/acquisition/joint_panels_v1.json) and the [source catalog](../../evals/dataset-catalog.json). No live microphone or user-screen recording was used.
