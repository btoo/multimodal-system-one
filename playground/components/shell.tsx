import Link from "next/link";
import {
  FlaskConical,
  History,
  Layers3,
  ArrowUpRight,
  Github,
  AudioLines,
} from "lucide-react";
import { isOwner } from "@/lib/auth";
import { OwnerGate } from "./owner-gate";
export async function Shell({
  children,
  active = "playground",
}: {
  children: React.ReactNode;
  active?: string;
}) {
  const unlocked = await isOwner();
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/" className="brand" aria-label="MiSO home">
          <span className="brand-mark">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            MiSO<span className="brand-dot">.</span>
          </span>
        </Link>
        <div className="sidebar-caption">MULTIMODAL SYSTEM ONE</div>
        <nav aria-label="Main navigation">
          {[
            {
              id: "playground",
              href: "/",
              title: "Playground",
              icon: FlaskConical,
            },
            { id: "runs", href: "/runs", title: "Run history", icon: History },
            {
              id: "model",
              href: "/model",
              title: "Model & evidence",
              icon: Layers3,
            },
          ].map(({ id, href, title, icon: Icon }) => (
            <Link
              key={id}
              href={href}
              className={active === id ? "nav-link active" : "nav-link"}
            >
              <Icon size={17} />
              {title}
              {id === "playground" ? (
                <span className="nav-badge">v3</span>
              ) : null}
            </Link>
          ))}
        </nav>
        <div className="sidebar-note">
          <AudioLines size={21} />
          <p>
            Several questions
            <br />
            per observation.
          </p>
          <span>
            Audio, image, and language in.
            <br />
            Typed probabilities out.
          </span>
        </div>
        <div className="sidebar-footer">
          <span className="status-dot" />
          Research preview
          <a
            href="https://github.com/btoo/multimodal-system-one"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub source"
          >
            <Github size={17} />
          </a>
        </div>
      </aside>
      <div className="app-main">
        <header className="topbar">
          <div className="breadcrumb">
            MiSO <span>/</span>{" "}
            <strong>
              {active === "runs"
                ? "Runs"
                : active === "model"
                  ? "Model & evidence"
                  : "Playground"}
            </strong>
          </div>
          <div className="topbar-right">
            <a
              className="source-link"
              href="https://github.com/btoo/multimodal-system-one/blob/codex/research-foundation/docs/developer-api.md"
              target="_blank"
              rel="noreferrer"
            >
              API docs <ArrowUpRight size={14} />
            </a>
            <OwnerGate unlocked={unlocked} />
          </div>
        </header>
        {children}
      </div>
    </div>
  );
}
