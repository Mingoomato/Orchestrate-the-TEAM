# -*- coding: utf-8 -*-
"""
memory_layer.py — Cross-session memory persistence for orchestrate.py

Architecture:
  Phase 1: SQLite (structured store, ~1ms reads, NVMe SSD)
  Phase 2: ChromaDB (semantic search via Gemini text-embedding-004, no local model)
  Phase 4: GitHub async push (optional, silent on failure)

Tables:
  sessions     — full council session records
  agent_memory — per-agent task/result history
  task_log     — sprint task kanban log

Usage:
  from memory_layer import (
      save_session, save_agent_memory, update_task_status,
      get_agent_history, get_recent_sessions,
      search_memory, format_agent_history,
      github_push_async,
  )
"""

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

# ── Storage paths ────────────────────────────────────────────
MEMORY_DIR = Path("./memory")
MEMORY_DIR.mkdir(exist_ok=True)
DB_PATH     = MEMORY_DIR / "memory.db"
CHROMA_DIR  = MEMORY_DIR / "chroma_index"

# ── Thread safety ────────────────────────────────────────────
_conn_lock   = threading.Lock()
_chroma_lock = threading.Lock()
_chroma_col  = None   # lazy singleton


# ─────────────────────────────────────────────────────────────
# Phase 1 — SQLite Core
# ─────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Thread-safe SQLite connection with WAL mode (concurrent read/write)."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if not exist. Idempotent — safe to call at import time."""
    with _conn_lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                date         TEXT,
                agenda       TEXT,
                plan         TEXT,
                final_report TEXT,
                sprint_json  TEXT,
                disputes     TEXT,
                risk_verdict TEXT,
                created_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                agent      TEXT,
                task       TEXT,
                result     TEXT,
                session_id TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent ON agent_memory(agent, created_at DESC);

            CREATE TABLE IF NOT EXISTS task_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                member     TEXT,
                task       TEXT,
                status     TEXT,
                summary    TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_task_session ON task_log(session_id);
        """)
        conn.commit()
        conn.close()


# ── Write ────────────────────────────────────────────────────

def save_session(
    session_id:   str,
    agenda:       str,
    plan:         str,
    final_report: str,
    sprint_results: dict,
    disputes:     str = "",
    risk_verdict: str = "",
) -> None:
    """Persist a completed council session to SQLite + index in ChromaDB."""
    now = datetime.now().isoformat()
    sprint_json = json.dumps(sprint_results, ensure_ascii=False)
    with _conn_lock:
        conn = _get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, date, agenda, plan, final_report, sprint_json,
                disputes, risk_verdict, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (session_id,
             now[:10],
             agenda[:1000],
             plan[:2000],
             final_report[:4000],
             sprint_json[:2000],
             disputes[:500],
             risk_verdict[:100],
             now),
        )
        conn.commit()
        conn.close()

    # Index to ChromaDB in background (non-blocking)
    doc_text = f"Agenda: {agenda[:500]}\nPlan: {plan[:800]}\nReport: {final_report[:800]}"
    threading.Thread(
        target=_index_to_chroma,
        args=(f"session_{session_id}", doc_text,
              {"type": "session", "date": now[:10], "session_id": session_id}),
        daemon=True,
    ).start()


def save_agent_memory(
    agent_name: str,
    task:       str,
    result:     str,
    session_id: str,
) -> None:
    """Persist a team member's task + result to SQLite + index in ChromaDB."""
    now = datetime.now().isoformat()
    with _conn_lock:
        conn = _get_conn()
        cursor = conn.execute(
            """INSERT INTO agent_memory (agent, task, result, session_id, created_at)
               VALUES (?,?,?,?,?)""",
            (agent_name, task[:300], result[:500], session_id, now),
        )
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

    doc_text = f"Agent: {agent_name}\nTask: {task[:300]}\nResult: {result[:400]}"
    threading.Thread(
        target=_index_to_chroma,
        args=(f"agent_{agent_name}_{row_id}", doc_text,
              {"type": "agent", "agent": agent_name,
               "date": now[:10], "session_id": session_id}),
        daemon=True,
    ).start()


def update_task_status(
    session_id: str,
    member:     str,
    task:       str,
    status:     str,
    summary:    str = "",
) -> None:
    """Log sprint task status to task_log table."""
    now = datetime.now().isoformat()
    with _conn_lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO task_log
               (session_id, member, task, status, summary, created_at)
               VALUES (?,?,?,?,?,?)""",
            (session_id, member, task[:80], status, summary[:300], now),
        )
        conn.commit()
        conn.close()


# ── Read ─────────────────────────────────────────────────────

def get_agent_history(agent_name: str, limit: int = 10) -> list[dict]:
    """Return last `limit` task/result records for this agent."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT task, result, session_id, created_at
           FROM agent_memory WHERE agent=?
           ORDER BY created_at DESC LIMIT ?""",
        (agent_name, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_sessions(n: int = 3) -> list[dict]:
    """Return last `n` completed sessions."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, date, agenda, plan, final_report, risk_verdict, created_at
           FROM sessions ORDER BY created_at DESC LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_summary(session_id: str) -> dict | None:
    """Return full details for a single session."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id=?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Keyword search (fallback when ChromaDB unavailable) ──────

def search_memory_keyword(query: str, limit: int = 5) -> list[str]:
    """SQL LIKE search across sessions + agent_memory."""
    words  = query.split()[:5]
    snippets: list[str] = []
    conn = _get_conn()

    for word in words:
        pattern = f"%{word}%"
        rows = conn.execute(
            """SELECT 'Session ' || date || ': ' || substr(agenda,1,200) AS snippet
               FROM sessions WHERE agenda LIKE ? OR plan LIKE ?
               LIMIT 3""",
            (pattern, pattern),
        ).fetchall()
        snippets.extend(r[0] for r in rows)

        rows = conn.execute(
            """SELECT agent || ' (' || substr(created_at,1,10) || '): ' || substr(task,1,150) AS snippet
               FROM agent_memory WHERE task LIKE ? OR result LIKE ?
               LIMIT 3""",
            (pattern, pattern),
        ).fetchall()
        snippets.extend(r[0] for r in rows)

    conn.close()
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result[:limit]


# ─────────────────────────────────────────────────────────────
# Phase 2 — ChromaDB Vector Index (lazy, graceful fallback)
# ─────────────────────────────────────────────────────────────

def _get_chroma_collection():
    """Lazy singleton — returns ChromaDB collection or raises ImportError."""
    global _chroma_col
    with _chroma_lock:
        if _chroma_col is not None:
            return _chroma_col
        import chromadb  # noqa: PLC0415
        from chromadb.utils.embedding_functions import (  # noqa: PLC0415
            GoogleGenerativeAiEmbeddingFunction,
        )
        ef = GoogleGenerativeAiEmbeddingFunction(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model_name="models/text-embedding-004",   # 768-dim, Gemini API
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_col = client.get_or_create_collection(
            "quantum_memory", embedding_function=ef
        )
        return _chroma_col


def _index_to_chroma(doc_id: str, text: str, metadata: dict) -> None:
    """Upsert a document into ChromaDB. Silently skips if unavailable."""
    try:
        col = _get_chroma_collection()
        col.upsert(ids=[doc_id], documents=[text[:2000]], metadatas=[metadata])
    except Exception:
        pass   # ChromaDB not installed or API key missing — keyword fallback still works


def search_memory_semantic(query: str, top_k: int = 5) -> list[str]:
    """Semantic search via ChromaDB + Gemini embeddings."""
    try:
        col = _get_chroma_collection()
        # ChromaDB needs at least 1 document to query
        if col.count() == 0:
            return search_memory_keyword(query, limit=top_k)
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        docs = results.get("documents", [[]])[0]
        return docs if docs else search_memory_keyword(query, limit=top_k)
    except Exception:
        return search_memory_keyword(query, limit=top_k)


def search_memory(query: str, top_k: int = 5) -> list[str]:
    """
    Unified search: ChromaDB semantic if available, else SQL keyword fallback.
    Always returns list[str] — never raises.
    """
    try:
        return search_memory_semantic(query, top_k)
    except Exception:
        return search_memory_keyword(query, limit=top_k)


# ─────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────

def format_agent_history(records: list[dict]) -> str:
    """Format agent task history as markdown for LLM prompt injection."""
    if not records:
        return ""
    lines = ["## Your Past Tasks (most recent first)\n"]
    for r in records:
        date = (r.get("created_at") or "")[:10]
        task = (r.get("task") or "")[:120]
        res  = (r.get("result") or "")[:200]
        lines.append(f"**[{date}]** {task}")
        if res:
            lines.append(f"Result: {res}")
        lines.append("")
    return "\n".join(lines)


def format_recent_sessions(sessions: list[dict], max_chars: int = 1200) -> str:
    """Format recent session summaries for council context injection."""
    if not sessions:
        return ""
    lines = ["## Past Council Sessions\n"]
    total = 0
    for s in sessions:
        date    = s.get("date") or s.get("created_at", "")[:10]
        agenda  = (s.get("agenda") or "")[:200]
        verdict = s.get("risk_verdict") or ""
        snippet = f"**[{date}]** Agenda: {agenda}" + (f" | Risk verdict: {verdict}" if verdict else "")
        total += len(snippet)
        if total > max_chars:
            break
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Phase 4 — GitHub Sync (push + pull)
# ─────────────────────────────────────────────────────────────

def github_sync_to_db(
    token:     str | None = None,
    repo_name: str | None = None,
) -> int:
    """
    Pull session files from GitHub repo → import missing ones into SQLite.
    Called at orchestrate.py startup so all agents have full cross-session context.
    Returns number of new sessions imported.
    """
    token     = token     or os.environ.get("GITHUB_MEMORY_TOKEN")
    repo_name = repo_name or os.environ.get("GITHUB_MEMORY_REPO")
    if not token or not repo_name:
        return 0

    try:
        from github import Github  # noqa: PLC0415
        g     = Github(token)
        repo  = g.get_repo(repo_name)
        files = repo.get_contents("memory/sessions")
    except Exception:
        return 0

    imported = 0
    for f in files:
        stem  = Path(f.path).stem                        # e.g. "2026-03-23_1230_abcd1234"
        parts = stem.split("_", 2)
        if len(parts) < 3:
            continue
        sid_hint = parts[2].replace("_interrupted", "")  # session_id prefix

        # Skip if already in DB (match by id prefix)
        with _conn_lock:
            conn = _get_conn()
            row  = conn.execute(
                "SELECT id FROM sessions WHERE id=? OR id LIKE ?",
                (sid_hint, f"gh_{stem}%"),
            ).fetchone()
            conn.close()
        if row:
            continue

        # Read file content from GitHub
        try:
            content = f.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            continue

        # Try to parse as JSON checkpoint (interrupted session)
        agenda = plan = disputes = risk_verdict = ""
        final_report = content[:4000]
        sprint_results: dict = {}
        try:
            raw = content
            if raw.startswith("[INTERRUPTED"):
                raw = raw.split("\n\n", 1)[1] if "\n\n" in raw else raw
            data = json.loads(raw)
            agenda       = data.get("agenda",  "")[:1000]
            plan         = data.get("plan",    "")[:2000]
            final_report = json.dumps(data.get("summaries", {}))[:4000]
            sprint_results = data.get("member_results", {})
        except Exception:
            pass  # plain markdown → treat as final_report

        date_str = parts[0]  # "2026-03-23"
        new_id   = f"gh_{stem}"
        save_session(
            session_id    = new_id,
            agenda        = agenda or f"[imported from GitHub: {stem}]",
            plan          = plan,
            final_report  = final_report,
            sprint_results= sprint_results,
            disputes      = disputes,
            risk_verdict  = risk_verdict,
        )
        imported += 1

    return imported


def github_push_async(
    session_id: str,
    content_md: str,
    token:     str | None = None,
    repo_name: str | None = None,
) -> None:
    """
    Push session markdown to a GitHub private repo in a background thread.
    Silent on any failure — GitHub sync is optional.

    Requires:
      GITHUB_MEMORY_TOKEN  — Fine-grained PAT with repo read/write
      GITHUB_MEMORY_REPO   — e.g. "yourname/quantum-memory"
    """
    token     = token     or os.environ.get("GITHUB_MEMORY_TOKEN")
    repo_name = repo_name or os.environ.get("GITHUB_MEMORY_REPO")
    if not token or not repo_name:
        return   # Not configured — skip silently

    def _push() -> None:
        try:
            from github import Github  # noqa: PLC0415
            g    = Github(token)
            repo = g.get_repo(repo_name)
            date = datetime.now().strftime("%Y-%m-%d_%H%M")
            path = f"memory/sessions/{date}_{session_id[:8]}.md"
            repo.create_file(
                path,
                f"session: {session_id[:8]}",
                content_md.encode("utf-8"),
            )
        except Exception:
            pass   # Network down / token expired / repo not found → silent

    threading.Thread(target=_push, daemon=True).start()


# ─────────────────────────────────────────────────────────────
# Auto-init on import
# ─────────────────────────────────────────────────────────────
init_db()


# ─────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Phase 1 test
    print("=== Phase 1: SQLite write/read ===")
    save_session(
        "test-session-001",
        "Test agenda: improve QLSTM win rate",
        "Plan: fix SpectralDecomposer batch stats bug",
        "Final: Gate 0 fix applied, WR improved to 38%",
        {"alpha": "Fixed normalizer", "beta": "Backtest updated", "cto": "Stats validated"},
        disputes="Radi: aggressive entry. Viktor: wait for confirmation.",
        risk_verdict="MAINTAIN",
    )
    save_agent_memory("Darvin", "Fix RollingZScoreNormalizer in backtest loop",
                      "Added window=20 normalizer, backtest WR 19.7 -> 38%", "test-session-001")
    update_task_status("test-session-001", "Darvin",
                       "Fix normalizer", "DONE", "WR improved")

    sessions = get_recent_sessions(1)
    print(f"Sessions saved: {len(sessions)}")
    hist = get_agent_history("Darvin", limit=3)
    print(f"Darvin history: {len(hist)} records")
    kw = search_memory_keyword("normalizer", limit=3)
    print(f"Keyword search 'normalizer': {len(kw)} results")

    # Phase 2 test
    print("\n=== Phase 2: ChromaDB semantic search ===")
    results = search_memory("win rate improvement strategy", top_k=3)
    print(f"Semantic search results: {len(results)}")
    for r in results:
        print(f"  - {r[:80]}")

    print("\n=== Format helpers ===")
    print(format_agent_history(hist))
    print(format_recent_sessions(sessions))
    print("OK")
