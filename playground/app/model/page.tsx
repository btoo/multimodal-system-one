import { Shell } from "@/components/shell";
import { ArrowUpRight } from "lucide-react";
import card from "@/data/model-card.json";
export default function Page() {
  return (
    <Shell active="model">
      <main className="page-content model-page">
        <section className="page-heading">
          <div>
            <div className="eyebrow">KNOW WHAT YOU ARE TESTING</div>
            <h1>MiSO v3 model card</h1>
            <p>
              MiSO v3 learns decisions from raw audio, pixels, and a bounded
              question vocabulary.
            </p>
          </div>
        </section>
        <div className="model-grid">
          <article className="evidence-card">
            <span className="type-badge choice">MiSO v3</span>
            <h2>Audio + image + text</h2>
            <dl>
              <div>
                <dt>Parameters</dt>
                <dd>668,097</dd>
              </div>
              <div>
                <dt>Audio</dt>
                <dd>Mono PCM16 WAV, ≤1 second</dd>
              </div>
              <div>
                <dt>Images</dt>
                <dd>PNG or JPEG, resized to 128 × 128</dd>
              </div>
              <div>
                <dt>Layout</dt>
                <dd>Generated 2×2 symbol panels</dd>
              </div>
              <div>
                <dt>Inputs required</dt>
                <dd>Image, audio, and text instructions</dd>
              </div>
            </dl>
            <p>
              Supported words: down, go, left, no, right, stop, up, yes. General
              screenshots, free-form speech, and arbitrary text state are
              outside this checkpoint’s scope.
            </p>
            <h3>Question vocabulary</h3>
            <div className="vocab">
              {card.vocabulary.slice(2).map((word: string) => (
                <code key={word}>{word}</code>
              ))}
            </div>
          </article>
          <article className="evidence-card">
            <span className="type-badge noul">Measured results</span>
            <h2>Accuracy has a context.</h2>
            <div className="evidence-number">
              81.32<span>%</span>
            </div>
            <p>
              Familiar combinations on the recorded confirmation set. Known held
              compositions score 66.99%. These use fresh voices and generated
              panels.
            </p>
            <a
              className="evidence-link"
              href="https://github.com/btoo/multimodal-system-one/blob/codex/research-foundation/reports/optimization-v1/README.md"
              target="_blank"
              rel="noreferrer"
            >
              Read the full evaluation <ArrowUpRight size={15} />
            </a>
            <hr />
            <h3>Text is part of the native model.</h3>
            <p>
              MiSO encodes each text instruction with the image and audio.
              Choice, Noul, Score and Ranking are views of its candidate
              probabilities. The checkpoint supports its learned question
              vocabulary; general text context remains future model work.
            </p>
            <a
              className="evidence-link"
              href="https://github.com/btoo/multimodal-system-one/blob/codex/research-foundation/reports/playground-text-v1/README.md"
              target="_blank"
              rel="noreferrer"
            >
              Native text verification <ArrowUpRight size={15} />
            </a>
          </article>
        </div>
        <p className="model-footnote">
          MiSO confidence is the largest calibrated candidate probability. It
          does not establish that an unfamiliar input is understood. Runs and
          their submitted inputs are retained in the private Workflow store;
          recent links are saved in this browser.
        </p>
      </main>
    </Shell>
  );
}
