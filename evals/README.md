# Runnable evaluation harness

The current implementation acquires three pinned public-data controls, validates their manifests, trains two small audio classifiers, runs fixed screen controls, and scores prediction files independently of the model.

See [measured results](../reports/pilot-v1/README.md) and the [application research contract](../docs/research/audio-screen-evals.md). Sentence-level intent, overlapping-event streams, paired human-audio/screens, and executed workflows are still unpopulated application evaluations.

## Install and acquire

From the repository root:

```bash
uv sync --locked --group docs
uv run mmso prepare all
uv run mmso validate evals/manifests/speech_keywords.jsonl
uv run python -m unittest discover -s tests -v
```

Preparation downloads the roughly 182 MB Mini Speech Commands archive, 400 ESC-10 clips, and 24 ScreenSpot-Pro images. The CLIP screen control additionally downloads a pinned pretrained checkpoint. The complete local pilot occupied approximately 1.1 GB including model weights on the initial machine. Raw media/model downloads stay in ignored `data/`; manifests and content hashes are versioned. Source notices are retained beside local data. See [the catalog](dataset-catalog.json) for upstream terms and subset scope.

The speech adapter uses a custom speaker-hash split and per-class caps. It does not reproduce the tutorial's random-file split or an official Speech Commands benchmark. The sound adapter uses original recording folds in a custom 1+2 / 3 / 4 / 5 assignment, not five-fold cross-validation. The screen sample is an explicitly disclosed external probe, not the full benchmark.

Re-preparation refuses to silently replace a frozen manifest if records change. Model runs verify local media hashes before consuming them. The split audit catches group overlap and identical media across splits. Same-split repeated media is reported separately in the pilot analysis; no claim of complete semantic deduplication is made.

## Run the controls

Run IDs must be new; completed results are never overwritten.

```bash
uv run mmso audio-baseline speech_keywords --run-id speech-reproduction-01
uv run mmso audio-baseline sound_events --run-id sounds-reproduction-01
uv run mmso screen-baseline --run-id screens-reproduction-01
```

Defaults are fixed in [pilot-protocol.json](pilot-protocol.json): one seed, at most 16 epochs and 120 synchronized training seconds for each audio dataset, development-only checkpoint selection, calibration-only temperature fitting, then test evaluation. The first runs reached the epoch cap well before the time cap. Training has no paid model API dependency. Screen scoring uses downloaded local CLIP weights with a fixed revision and no remote model code.

Audio checkpoints and configs are written to `artifacts/<run-id>/`; reports and complete prediction files to `reports/<run-id>/`. The audio frontend uses offline log-mel features, so it is not a streaming model. A small mean-valued time pad makes the actual MPS pooling shapes supported without moving CNN layers to CPU.

## Independently rescore saved predictions

```bash
uv run mmso score evals/manifests/speech_keywords.jsonl reports/speech-cnn-v1/predictions.jsonl
uv run mmso score evals/manifests/screen_grounding.jsonl reports/screens-clip-v1/predictions.jsonl --split external_probe --field clip_grid_point
```

The scorer requires exactly one output per selected manifest ID; duplicate, omitted, and extra predictions fail. Array columns follow the manifest's explicit label order. `--metadata-only` permits rescoring downloaded reports without local media, but the result then explicitly says that media was not verified.

Implemented metric kinds:

| Kind | Target contract | Prediction contract |
|---|---|---|
| `categorical` | `labels` and one `target` | `id` and a `probabilities` array in label order |
| `multilabel` | `labels` and a `targets` list, possibly empty | Independent probabilities in label order |
| `grounding` | `image_size` and pixel `target_bbox_xyxy` | A named point field such as `clip_grid_point` |
| `joint` | A nonempty list of `acceptable_answers` with `action`/`target` pairs | `id`, `answer` with `action`/`target`, and `confidence` |

Every manifest record has `id`, `dataset`, `kind`, `split`, `group_id`, and a `media` list containing repository-relative paths and SHA-256 values. Audio and image acquisition manifests provide executable examples of this format. Future paired media records should also preserve time, modality presence, and declared history. Grounding coordinates are pixels, not normalized coordinates.

Joint exact match requires both action and target. Abstention only counts as correct if it is an explicitly acceptable target response. Its coverage excludes abstentions. The generic scorer supports this contract, but no real paired dataset or joint model has been evaluated yet.

## Use a trained audio checkpoint

```bash
uv run mmso predict-audio artifacts/speech-cnn-v1/model.safetensors path/to/command.wav
```

Use a PCM16 WAV file. The model resamples to 16 kHz and pads/truncates to the checkpoint's offline duration. The speech model selects among eight known keywords; it has no trained unknown-command rejection. The sound checkpoint selects among ten mutually exclusive environmental classes; it does not detect overlapping events. These are controls for the larger model.

## Preserve evidence

Raw and temperature-adjusted metrics, class priors, confusion matrices, reliability-bin counts, risk/coverage, cluster uncertainty intervals, CPU/MPS comparison, checkpoint hashes, exact source-file fingerprints, and timings are retained in each report. The published audio uncertainty intervals apply to one fixed trained checkpoint; they do not estimate variation across training seeds.

The screen experiment's oracle-crop diagnostic is explicitly separated from grounding. Neither implies that a browser workflow succeeded. This prototype has ordinary process/function boundaries and content checks; it is not an adversarial sandbox for untrusted evolving code.
