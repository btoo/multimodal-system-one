import Link from "next/link";
export default function NotFound() {
  return (
    <main className="history-empty">
      <h1>Page not found.</h1>
      <Link href="/" className="primary-button">
        Open playground
      </Link>
    </main>
  );
}
