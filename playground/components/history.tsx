"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, History, AudioLines } from "lucide-react";
import { misoHistory, type HistoryItem } from "@/lib/history";
export function RunHistory() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  useEffect(() => {
    try {
      const items = misoHistory(
        JSON.parse(localStorage.getItem("miso:runs:v1") ?? "[]"),
      );
      setItems(items);
      localStorage.setItem("miso:runs:v1", JSON.stringify(items));
    } catch {}
  }, []);
  return items.length ? (
    <div className="history-list">
      {items.map((item) => (
        <Link className="history-row" href={`/runs/${item.id}`} key={item.id}>
          <span className="history-icon">
            <AudioLines size={20} />
          </span>
          <div>
            <strong>{item.label}</strong>
            <span>{item.id}</span>
          </div>
          <time>{new Date(item.createdAt).toLocaleString()}</time>
          <ArrowUpRight size={18} />
        </Link>
      ))}
    </div>
  ) : (
    <div className="history-empty">
      <History size={30} />
      <h2>No runs on this browser yet</h2>
      <p>
        Your recent run links will appear here. Each result has its own saved
        page.
      </p>
      <Link href="/" className="primary-button">
        Open playground
      </Link>
    </div>
  );
}
