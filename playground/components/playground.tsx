"use client";
import { useRef, useState } from "react";
import Image from "next/image";
import {
  ArrowRight,
  AudioLines,
  Braces,
  ImagePlus,
  LoaderCircle,
  Play,
  RefreshCcw,
  Type,
  Upload,
} from "lucide-react";
import nativeDefault from "@/data/native-questions.json";
import jevDefault from "@/data/jev-example.json";
import { runInputSchema, type RunResult } from "@/lib/contracts";
import { RunResults, rememberRun } from "./results";
const wave = [
  18, 12, 10, 7, 8, 7, 6, 5, 6, 5, 6, 5, 6, 6, 6, 5, 5, 8, 42, 90, 68, 65, 54,
  39, 30, 26, 19, 20, 16, 9, 6, 6, 8, 12, 6, 5,
];
export function Playground({
  unlocked,
  initialRunId,
}: {
  unlocked: boolean;
  initialRunId: string | null;
}) {
  const [provider, setProvider] = useState<"miso" | "jev">("miso"),
    [questions, setQuestions] = useState(
      JSON.stringify(nativeDefault, null, 2),
    ),
    [text, setText] = useState(jevDefault.state),
    [jevQuestions, setJevQuestions] = useState(
      JSON.stringify(jevDefault.questions, null, 2),
    );
  const [image, setImage] = useState("/examples/panel.png"),
    [audio, setAudio] = useState("/examples/voice.wav"),
    [imageName, setImageName] = useState("symbol-panel.png"),
    [audioName, setAudioName] = useState("recorded-keyword.wav"),
    [runId, setRunId] = useState(initialRunId),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [edit, setEdit] = useState(false),
    [playing, setPlaying] = useState(false);
  const imageInput = useRef<HTMLInputElement>(null),
    audioInput = useRef<HTMLInputElement>(null),
    audioPlayer = useRef<HTMLAudioElement>(null);
  const restoreRun = useRef(initialRunId);
  function restoreInput(result: RunResult) {
    if (!restoreRun.current || restoreRun.current !== runId || !result.request)
      return;
    const input = result.request;
    setProvider(input.provider);
    if (input.provider === "jev") {
      setText(input.payload.state);
      setJevQuestions(JSON.stringify(input.payload.questions, null, 2));
    } else {
      setQuestions(JSON.stringify(input.payload.questions, null, 2));
      for (const media of input.payload.input) {
        const url = `data:${media.source.media_type};base64,${media.source.data}`;
        if (media.type === "image") {
          setImage(url);
          setImageName("saved-image");
        } else {
          setAudio(url);
          setAudioName("saved-audio.wav");
        }
      }
    }
    restoreRun.current = null;
  }
  function changeProvider(next: "miso" | "jev") {
    setProvider(next);
    setEdit(false);
    setError("");
  }
  async function upload(file: File | undefined, kind: "image" | "audio") {
    if (!file) return;
    if (file.size > 1_000_000) {
      setError("Choose a file smaller than 1 MB.");
      return;
    }
    if (kind === "image" && !["image/png", "image/jpeg"].includes(file.type)) {
      setError("Choose a PNG or JPEG image.");
      return;
    }
    if (kind === "audio" && !file.name.toLowerCase().endsWith(".wav")) {
      setError("Choose a mono PCM16 WAV recording, up to one second.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (kind === "image") {
        setImage(String(reader.result));
        setImageName(file.name);
      } else {
        setAudio(String(reader.result));
        setAudioName(file.name);
        setPlaying(false);
      }
      setError("");
    };
    reader.onerror = () => setError("This file could not be read.");
    reader.readAsDataURL(file);
  }
  async function block(url: string, type: "image" | "audio") {
    const response = await fetch(url);
    if (!response.ok) throw new Error("The example media could not be loaded.");
    const blob = await response.blob();
    return await new Promise<object>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () =>
        resolve({
          type,
          source: {
            type: "base64",
            media_type:
              type === "audio" ? "audio/wav" : blob.type || "image/png",
            data: String(reader.result).split(",")[1],
          },
        });
      reader.onerror = () => reject(new Error("Media could not be read."));
      reader.readAsDataURL(blob);
    });
  }
  async function run() {
    setBusy(true);
    setError("");
    try {
      const body =
        provider === "miso"
          ? {
              provider,
              label: "Audio + image decision",
              payload: {
                model: "mmso-joint-v3",
                input: await Promise.all([
                  block(image, "image"),
                  block(audio, "audio"),
                ]),
                questions: JSON.parse(questions),
                abstain_threshold: 0,
              },
            }
          : {
              provider,
              label: "Text decision",
              payload: { state: text, questions: JSON.parse(jevQuestions) },
            };
      const valid = runInputSchema.parse(body);
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(valid),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      setRunId(result.id);
      rememberRun({
        id: result.id,
        provider,
        label: body.label,
        createdAt: new Date().toISOString(),
      });
      window.history.replaceState(null, "", `/?run=${result.id}`);
    } catch (e) {
      setError(
        e instanceof SyntaxError
          ? "Questions must be valid JSON."
          : e instanceof Error
            ? e.message
            : "The run could not be started.",
      );
    } finally {
      setBusy(false);
    }
  }
  function reset() {
    if (provider === "miso") {
      setImage("/examples/panel.png");
      setAudio("/examples/voice.wav");
      setImageName("symbol-panel.png");
      setAudioName("recorded-keyword.wav");
      setQuestions(JSON.stringify(nativeDefault, null, 2));
    } else {
      setText(jevDefault.state);
      setJevQuestions(JSON.stringify(jevDefault.questions, null, 2));
    }
    setEdit(false);
    setError("");
  }
  let parsed: Record<
    string,
    { type: string; question?: string; instructions?: string }
  > = {};
  try {
    parsed = JSON.parse(provider === "miso" ? questions : jevQuestions);
  } catch {
    /* The editor displays the parse error on Run. */
  }
  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" />
            THE DECISION PLAYGROUND
          </div>
          <h1>Give MiSO something to decide.</h1>
          <p>
            {provider === "miso"
              ? "Pair a picture with a voice, ask a question, and inspect the probabilities."
              : "Ask typed questions about text with Jev, then inspect its decisions."}
          </p>
        </div>
        <span className="preview-tag">EXPERIMENTAL</span>
      </section>
      <div className="workspace">
        <section className="input-pane">
          <div
            className="mode-switch"
            role="tablist"
            aria-label="Model input mode"
          >
            <button
              role="tab"
              aria-selected={provider === "miso"}
              className={provider === "miso" ? "selected" : ""}
              onClick={() => changeProvider("miso")}
            >
              <AudioLines size={15} />
              Audio + image<span>MiSO v3</span>
            </button>
            <button
              role="tab"
              aria-selected={provider === "jev"}
              className={provider === "jev" ? "selected" : ""}
              onClick={() => changeProvider("jev")}
            >
              <Type size={15} />
              Text<span>Jev</span>
            </button>
          </div>
          <div className="input-content">
            <div className="section-label">
              <span>
                <span className="step-number">01</span>Observation
              </span>
              <button className="text-button" onClick={reset}>
                <RefreshCcw size={12} />
                Load sample
              </button>
            </div>
            {provider === "miso" ? (
              <>
                <div className="media-grid">
                  <div className="image-card">
                    <div className="media-card-head">
                      <span>
                        <ImagePlus size={14} />
                        Image
                      </span>
                      <span>128 × 128 input</span>
                    </div>
                    <div className="image-preview">
                      <Image
                        src={image}
                        width={168}
                        height={168}
                        alt="Input symbol panel"
                        unoptimized
                        priority
                      />
                    </div>
                    <div className="media-card-foot">
                      <span title={imageName}>{imageName}</span>
                      <button
                        onClick={() => imageInput.current?.click()}
                        aria-label="Upload image"
                      >
                        <Upload size={14} />
                      </button>
                    </div>
                    <input
                      ref={imageInput}
                      type="file"
                      accept="image/png,image/jpeg"
                      aria-label="Image file"
                      hidden
                      onChange={(e) => upload(e.target.files?.[0], "image")}
                    />
                  </div>
                  <div className="audio-card">
                    <div className="media-card-head">
                      <span>
                        <AudioLines size={14} />
                        Audio
                      </span>
                      <span>WAV</span>
                    </div>
                    <div
                      className={`waveform ${playing ? "playing" : ""}`}
                      aria-hidden="true"
                    >
                      {wave.map((h, i) => (
                        <i
                          key={i}
                          style={{
                            height: `${h}%`,
                            animationDelay: `${i * 17}ms`,
                          }}
                        />
                      ))}
                    </div>
                    <div className="audio-play-row">
                      <button
                        className="play-audio"
                        aria-label={playing ? "Pause audio" : "Play audio"}
                        onClick={() => {
                          if (!audioPlayer.current) return;
                          if (playing) {
                            audioPlayer.current.pause();
                            setPlaying(false);
                          } else {
                            audioPlayer.current
                              .play()
                              .then(() => setPlaying(true))
                              .catch(() =>
                                setError("Audio could not be played."),
                              );
                          }
                        }}
                      >
                        {playing ? (
                          <span className="pause-icon" />
                        ) : (
                          <Play size={14} fill="currentColor" />
                        )}
                      </button>
                      <div>
                        <strong>Recorded voice</strong>
                        <span>One-second keyword</span>
                      </div>
                    </div>
                    <div className="media-card-foot">
                      <span title={audioName}>{audioName}</span>
                      <button
                        onClick={() => audioInput.current?.click()}
                        aria-label="Upload audio"
                      >
                        <Upload size={14} />
                      </button>
                    </div>
                    <audio
                      ref={audioPlayer}
                      src={audio}
                      onEnded={() => setPlaying(false)}
                      preload="metadata"
                    />
                    <input
                      ref={audioInput}
                      type="file"
                      accept=".wav,audio/wav"
                      aria-label="Audio file"
                      hidden
                      onChange={(e) => upload(e.target.files?.[0], "audio")}
                    />
                  </div>
                </div>
                <p className="input-help">
                  MiSO v3 understands eight spoken keywords and generated 2×2
                  symbol panels.{" "}
                  <a href="/model">
                    View model limits <ArrowRight size={11} />
                  </a>
                </p>
              </>
            ) : (
              <div className="text-input-card">
                <label htmlFor="state">Text state</label>
                <textarea
                  id="state"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  maxLength={12000}
                  rows={7}
                />
                <div className="text-input-footer">
                  <span>jev-1.13.0 · text only</span>
                  <span>{text.length.toLocaleString()} / 12,000</span>
                </div>
              </div>
            )}
            <div className="section-label questions-label">
              <span>
                <span className="step-number">02</span>Questions
                <span className="count-badge">
                  {Object.keys(parsed).length}
                </span>
              </span>
              <button className="text-button" onClick={() => setEdit(!edit)}>
                <Braces size={13} />
                {edit ? "View questions" : "Edit JSON"}
              </button>
            </div>
            {edit ? (
              <textarea
                aria-label="Questions JSON"
                className="json-editor"
                spellCheck={false}
                value={provider === "miso" ? questions : jevQuestions}
                onChange={(e) =>
                  provider === "miso"
                    ? setQuestions(e.target.value)
                    : setJevQuestions(e.target.value)
                }
                rows={16}
              />
            ) : (
              <div className="question-list">
                {Object.entries(parsed).map(([id, q]) => (
                  <div className="question-row" key={id}>
                    <div className="question-title">
                      <span className={`type-badge ${q.type}`}>{q.type}</span>
                      <span>{id.replaceAll("_", " ")}</span>
                    </div>
                    <p>{q.question ?? q.instructions}</p>
                  </div>
                ))}
              </div>
            )}
            {error ? (
              <div className="inline-error" role="alert">
                {error}
              </div>
            ) : null}
            <div className="run-control">
              <button
                className="primary-button run-button"
                onClick={run}
                disabled={busy || !unlocked}
              >
                {busy ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
                {busy
                  ? "Starting run…"
                  : `Run ${provider === "miso" ? "MiSO" : "Jev"}`}
                <span className="run-arrow">
                  <ArrowRight size={16} />
                </span>
              </button>
              <p>
                {unlocked
                  ? "Results are saved. You can revisit a run after leaving this page."
                  : "Explore the sample, then unlock live runs at the top right."}
              </p>
            </div>
          </div>
        </section>
        <RunResults runId={runId} unlocked={unlocked} onResult={restoreInput} />
      </div>
      <footer className="page-footer">
        <span>MiSO v3 · 668,097 parameters · research preview</span>
        <a
          href="https://github.com/btoo/multimodal-system-one"
          target="_blank"
          rel="noreferrer"
        >
          Read the research <ArrowRight size={12} />
        </a>
      </footer>
    </>
  );
}
