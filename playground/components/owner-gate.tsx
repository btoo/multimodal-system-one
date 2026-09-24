"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { LockKeyhole, Check, X } from "lucide-react";
export function OwnerGate({ unlocked }: { unlocked: boolean }) {
  const [open, setOpen] = useState(false),
    [key, setKey] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const router = useRouter();
  async function unlock(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const r = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error);
      setKey("");
      setOpen(false);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not unlock.");
    } finally {
      setBusy(false);
    }
  }
  async function lock() {
    await fetch("/api/session", { method: "DELETE" });
    router.refresh();
  }
  return (
    <>
      <button
        className={`session-button ${unlocked ? "unlocked" : ""}`}
        onClick={() => (unlocked ? lock() : setOpen(true))}
      >
        {unlocked ? <Check size={14} /> : <LockKeyhole size={14} />}
        <span>{unlocked ? "Live runs unlocked" : "Unlock live runs"}</span>
      </button>
      {open ? (
        <div className="modal-backdrop" onClick={() => setOpen(false)}>
          <section
            className="unlock-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="unlock-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="icon-button close-dialog"
              aria-label="Close unlock dialog"
              onClick={() => setOpen(false)}
            >
              <X size={18} />
            </button>
            <div className="dialog-icon">
              <LockKeyhole size={24} />
            </div>
            <h2 id="unlock-title">Unlock live runs</h2>
            <p>
              Use your playground access key. Model API keys stay on the server.
            </p>
            <form onSubmit={unlock}>
              <label htmlFor="access-key">Access key</label>
              <input
                id="access-key"
                type="password"
                autoComplete="off"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                autoFocus
                required
              />
              {error ? (
                <p className="error-message" role="alert">
                  {error}
                </p>
              ) : null}
              <button className="primary-button" disabled={busy}>
                {busy ? "Unlocking…" : "Unlock playground"}
              </button>
            </form>
          </section>
        </div>
      ) : null}
    </>
  );
}
