"use client";
import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Braces,
  Check,
  ChevronDown,
  Copy,
  LoaderCircle,
  Network,
  Download,
  AlertCircle,
} from "lucide-react";
import type { RunResult, RunState } from "@/lib/contracts";
export type HistoryItem = {
  id: string;
  provider: string;
  label: string;
  createdAt: string;
};
export function rememberRun(item: HistoryItem) {
  try {
    const prior: HistoryItem[] = JSON.parse(
      localStorage.getItem("miso:runs:v1") ?? "[]",
    );
    localStorage.setItem(
      "miso:runs:v1",
      JSON.stringify(
        [item, ...prior.filter((i) => i.id !== item.id)].slice(0, 50),
      ),
    );
  } catch {
    /* Storage availability never determines whether a run succeeds. */
  }
}
export function RunResults({
  runId,
  unlocked,
  onResult,
}: {
  runId: string | null;
  unlocked: boolean;
  onResult?: (result: RunResult) => void;
}) {
  const [state, setState] = useState<RunState | null>(null),
    [error, setError] = useState("");
  const callback = useRef(onResult);
  callback.current = onResult;
  useEffect(() => {
    if (!runId || !unlocked) return;
    let cancelled = false,
      timer: ReturnType<typeof setTimeout>;
    const abort = new AbortController();
    setState({ id: runId, status: "pending" });
    setError("");
    async function poll() {
      try {
        const response = await fetch(`/api/runs/${runId}`, {
          signal: abort.signal,
          cache: "no-store",
        });
        const body = await response.json();
        if (!response.ok)
          throw new Error(body.error ?? "Could not read run status.");
        if (cancelled) return;
        setState(body);
        if (body.result) {
          callback.current?.(body.result);
          return;
        }
        if (body.status === "failed" || body.status === "cancelled") return;
        timer = setTimeout(poll, 1200);
      } catch (e) {
        if (cancelled) return;
        setState({ id: runId!, status: "failed" });
        setError(e instanceof Error ? e.message : "Could not read this run.");
      }
    }
    poll();
    return () => {
      cancelled = true;
      abort.abort();
      clearTimeout(timer);
    };
  }, [runId, unlocked]);
  const pending =
    !!runId &&
    !!state &&
    !["completed", "failed", "cancelled"].includes(state.status);
  return (
    <section className="results-pane" aria-label="Model results">
      <div className="pane-title">
        <div>
          <span className="eyebrow">OUTPUT</span>
          <h2>Decisions</h2>
        </div>
        <span
          className={`result-status ${state?.status === "completed" ? "complete" : ""}`}
        >
          {pending ? (
            <>
              <LoaderCircle size={12} className="spin" />
              Running
            </>
          ) : state?.status === "completed" ? (
            <>
              <Check size={12} />
              Complete
            </>
          ) : (
            state?.status === "failed" || state?.status === "cancelled" ? "Failed" : "Ready"
          )}
        </span>
      </div>
      {error || state?.error ? (
        <div className="inline-error" role="alert">
          <AlertCircle size={17} />
          {error || state?.error}
        </div>
      ) : null}
      {state?.result ? (
        <ResultBody result={state.result} runId={runId!} />
      ) : (
        <div className="result-empty">
          <div className={`decision-glyph ${pending ? "working" : ""}`}>
            <Network size={36} strokeWidth={1.3} />
          </div>
          <h3>
            {pending
              ? "Reading the observation"
              : runId && !unlocked
                ? "Unlock to open this run"
                : "Probabilities for every answer"}
          </h3>
          <p>
            {pending
              ? "You can leave this page. The run will finish and save its result."
              : "Run a sample or add your own inputs to inspect each answer and its probability."}
          </p>
          <div className="empty-types">
            <span>Choice</span>
            <span>Noul</span>
            <span>Score</span>
            <span>Ranking</span>
          </div>
          {runId ? (
            <code className="run-code">{runId.slice(0, 23)}…</code>
          ) : null}
        </div>
      )}
    </section>
  );
}
export function ResultBody({
  result,
  runId,
}: {
  result: RunResult;
  runId: string;
}) {
  const [raw, setRaw] = useState(false),
    [copied, setCopied] = useState(false);
  const json = JSON.stringify(result.raw, null, 2);
  async function copy() {
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setRaw(true);
    }
  }
  function download() {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `miso-${runId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <>
      <div className="run-stats">
        <div>
          <span>MODEL CALL</span>
          <strong>
            {result.inferenceMs.toFixed(0)}
            <small>ms</small>
          </strong>
        </div>
        <div>
          <span>
            {result.provider === "jev" ? "INPUT TOKEN COST" : "CHECKPOINT"}
          </span>
          <strong className={result.provider === "miso" ? "small-stat" : ""}>
            {result.costUsd !== null
              ? `$${result.costUsd.toFixed(6)}`
              : (result.checkpoint?.slice(0, 8) ?? "v3")}
          </strong>
        </div>
        <div>
          <span>ANSWERS</span>
          <strong>{result.answers.length.toString().padStart(2, "0")}</strong>
        </div>
      </div>
      <div className="result-model">
        <span className="status-dot" />
        {result.model}
        <span>Live response</span>
      </div>
      <div className="answers">
        {result.answers.map((answer) => (
          <article className="answer-card" key={answer.id}>
            <div className="answer-heading">
              <span className={`type-badge ${answer.type}`}>{answer.type}</span>
              <span className="answer-id">
                {answer.id.replaceAll("_", " ")}
              </span>
            </div>
            <div className="answer-value">
              {answer.abstained
                ? "Abstained"
                : typeof answer.value === "number"
                  ? answer.type === "noul"
                    ? `${(answer.value * 100).toFixed(1)}% true`
                    : answer.value.toFixed(3)
                  : (answer.probabilities.find((p) => p.label === answer.value)
                      ?.label ??
                    answer.value ??
                    "No single decision")}
            </div>
            <div className="probabilities">
              {answer.probabilities.map((p, index) => (
                <div className="probability" key={p.label}>
                  <div className="prob-label">
                    <span>{p.label}</span>
                    <span>{(p.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="bar-track">
                    <div
                      className={
                        index === 0 ? "bar-fill" : "bar-fill secondary"
                      }
                      style={{ width: `${p.probability * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
      <div className="result-actions">
        <button onClick={() => setRaw(!raw)}>
          <Braces size={14} />
          JSON
          <ChevronDown size={12} />
        </button>
        <button onClick={copy}>
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
        <button onClick={download} aria-label="Download run JSON">
          <Download size={14} />
        </button>
        <Link href={`/runs/${runId}`}>
          Saved run
          <ArrowUpRight size={13} />
        </Link>
      </div>
      {raw ? <pre className="json-output">{json}</pre> : null}
      <p className="measurement-note">
        Model-call time includes the server’s request to the model. Workflow
        queue and browser time are additional.
      </p>
    </>
  );
}
