# V4 research decision readouts

Each directory records the exact publisher checkpoint revision, fitted readout and calibration temperature selected using development data. These artifacts are not registered in the public API.

The study used SLURP audio under CC BY-NC 4.0. These experiment weights are for noncommercial research; the source-code license does not override model or data terms. [SLURP authors and licensing](https://github.com/pswietojanski/slurp#license), [data audit](../../evals/acquisition/v4_selection_v1.json), [model revisions and licenses](../../evals/v4-candidates-v1.json).

The Qwen-based research heads are **Built with Qwen**. In particular, Qwen2.5-Omni-3B is governed by the Qwen Research License, not Apache 2.0. Its upstream license and notice are retained in that directory.

The `-adapted` directories refer to the separately recorded attention adapters under [v4-adapters](../v4-adapters/README.md). A missing `readout.safetensors` is intentional when `selected_method` is `label_logits`; the output projection is then supplied by the frozen base/adapted model, followed by the recorded temperature.
