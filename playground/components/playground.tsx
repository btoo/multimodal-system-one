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
import {
  runInputSchema,
  makeNativePayload,
  resolveNativeQuestions,
  type RunResult,
} from "@/lib/contracts";
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
  const [questions, setQuestions] = useState(
    JSON.stringify(nativeDefault, null, 2),
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
    setQuestions(
      JSON.stringify(resolveNativeQuestions(input.payload), null, 2),
    );
    for (const media of input.payload.input) {
      if (media.type === "text") continue;
      const url = `data:${media.source.media_type};base64,${media.source.data}`;
      if (media.type === "image") {
        setImage(url);
        setImageName("saved-image");
      } else {
        setAudio(url);
        setAudioName("saved-audio.wav");
      }
    }
    restoreRun.current = null;
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
      const body = {
        provider: "miso",
        label: "Audio + image + text decision",
        payload: makeNativePayload(
          await Promise.all([block(image, "image"), block(audio, "audio")]),
          JSON.parse(questions),
        ),
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
        provider: "miso",
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
    setImage("/examples/panel.png");
    setAudio("/examples/voice.wav");
    setImageName("symbol-panel.png");
    setAudioName("recorded-keyword.wav");
    setQuestions(JSON.stringify(nativeDefault, null, 2));
    setEdit(false);
    setError("");
  }
  let parsed: Record<
    string,
    { type: string; question?: string; instructions?: string }
  > = {};
  try {
    const value = JSON.parse(questions);
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      Object.values(value).every((q) => q && typeof q === "object")
    )
      parsed = value;
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
            Give MiSO audio, an image, and text instructions. Inspect how all
            three shape its decisions.
          </p>
        </div>
        <span className="preview-tag">EXPERIMENTAL</span>
      </section>
      <div className="workspace">
        <section className="input-pane">
          <div className="model-banner">
            <AudioLines size={16} />
            <strong>MiSO v3</strong>
            <span>Audio + image + text</span>
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
              <section
                className="native-text-card"
                aria-labelledby="native-text-title"
              >
                <div className="native-text-heading">
                  <Type size={15} />
                  <h2 id="native-text-title">Text instructions</h2>
                  <span>Native text input</span>
                </div>
                <p>
                  Each instruction is encoded by MiSO alongside the image and
                  audio.
                </p>
                {Object.entries(parsed).map(([id, q]) => (
                  <div className="native-text-field" key={id}>
                    <label htmlFor={`text-${id}`}>
                      {id.replaceAll("_", " ")}
                    </label>
                    <textarea
                      id={`text-${id}`}
                      aria-label={`Text instruction: ${id}`}
                      value={q.question ?? ""}
                      rows={2}
                      maxLength={512}
                      onChange={(event) => {
                        try {
                          const next = JSON.parse(questions);
                          next[id] = {
                            ...next[id],
                            question: event.target.value,
                          };
                          delete next[id].question_ref;
                          setQuestions(JSON.stringify(next, null, 2));
                        } catch {
                          setError(
                            "Fix the question JSON before editing its text.",
                          );
                        }
                      }}
                    />
                  </div>
                ))}
                <p className="text-scope">
                  This checkpoint understands its learned question vocabulary.
                  Free-form documents and text context still need a
                  language-capable successor.
                </p>
              </section>
              <p className="input-help">
                MiSO v3 understands eight spoken keywords and generated 2×2
                symbol panels.{" "}
                <a href="/model">
                  View model limits <ArrowRight size={11} />
                </a>
              </p>
            </>
            <div className="section-label questions-label">
              <span>
                <span className="step-number">02</span>
                Answer schemas
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
                value={questions}
                onChange={(e) => setQuestions(e.target.value)}
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
                    <p>Uses the text instruction above.</p>
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
                {busy ? "Starting run…" : "Run MiSO"}
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
