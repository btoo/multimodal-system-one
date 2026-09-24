"use client";
export default function ErrorPage({ reset }: { reset: () => void }) {
  return (
    <main className="history-empty">
      <h1>This page could not load.</h1>
      <p>Your existing runs are saved independently of this page.</p>
      <button className="primary-button" onClick={reset}>
        Try again
      </button>
    </main>
  );
}
