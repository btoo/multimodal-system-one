# MiSO-only playground

The [playground](https://miso-playground-nine.vercel.app) now runs only MiSO with audio, image and native text inputs. The prior external-model selector was an incorrect extension of the research comparison work into the product.

Removed the comparison tab, example data, model-card copy, model execution branch and pricing display. The request schema accepts only the MiSO model path. Historical non-MiSO runs are filtered from browser history and blocked by the result API. The external provider credential was removed from this Vercel project's production and preview configuration and from the local playground configuration. The separate research adapter, its credential and historical evidence remain available for benchmark work.

[Production verification](verification.json):

- External-provider run requests return HTTP 400 without starting a run.
- Historical external-model result URLs return HTTP 410.
- Existing native MiSO results remain readable with HTTP 200.
- A new native run completed with all four typed outputs and audio/image/text inputs.
- Browser checks confirmed the home page and model card contain no comparison-model UI, and the history excludes comparison entries.
- All nine playground contract tests, TypeScript checks and the production build passed.

The [before record](before.json) captures the earlier published comparison controls. This change corrects product scope; it does not change model weights, benchmark scores or research access.
