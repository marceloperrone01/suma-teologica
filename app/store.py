"""SQLite persistence for conversations, messages, and daily meditations."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "store.sqlite"

_lock = threading.Lock()
_initialized = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    last_at TEXT NOT NULL,
    model TEXT NOT NULL,
    title TEXT,
    dataset TEXT NOT NULL DEFAULT 'suma'
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS meditations (
    date TEXT PRIMARY KEY,
    citacao TEXT NOT NULL,
    parte TEXT NOT NULL,
    questao INTEGER NOT NULL,
    artigo INTEGER NOT NULL,
    paraphrase TEXT
);
"""


@dataclass
class Conversation:
    id: int
    started_at: str
    last_at: str
    model: str
    title: str | None
    dataset: str = "suma"


@dataclass
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    sources_json: str | None
    created_at: str


@contextmanager
def connect():
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if not _initialized:
            conn.executescript(SCHEMA)
            # Migrate existing databases that predate the dataset column.
            try:
                conn.execute("ALTER TABLE conversations ADD COLUMN dataset TEXT NOT NULL DEFAULT 'suma'")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists
            _initialized = True
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def create_conversation(model: str, title: str | None = None, dataset: str = "suma") -> int:
    with _lock, connect() as c:
        now = _now()
        cur = c.execute(
            "INSERT INTO conversations(started_at, last_at, model, title, dataset) VALUES (?, ?, ?, ?, ?)",
            (now, now, model, title, dataset),
        )
        return cur.lastrowid


def update_conversation_title(conversation_id: int, title: str) -> None:
    with _lock, connect() as c:
        c.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))


def update_conversation_model(conversation_id: int, model: str) -> None:
    with _lock, connect() as c:
        c.execute("UPDATE conversations SET model = ? WHERE id = ?", (model, conversation_id))


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> int:
    with _lock, connect() as c:
        now = _now()
        src = json.dumps(sources, ensure_ascii=False) if sources else None
        cur = c.execute(
            "INSERT INTO messages(conversation_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, src, now),
        )
        c.execute(
            "UPDATE conversations SET last_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        return cur.lastrowid


def list_conversations(limit: int = 50) -> list[Conversation]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, started_at, last_at, model, title, dataset FROM conversations "
            "ORDER BY last_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Conversation(**dict(r)) for r in rows]


def get_conversation(conversation_id: int) -> Conversation | None:
    with connect() as c:
        row = c.execute(
            "SELECT id, started_at, last_at, model, title, dataset FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return Conversation(**dict(row)) if row else None


def get_messages(conversation_id: int) -> list[Message]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, conversation_id, role, content, sources_json, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        return [Message(**dict(r)) for r in rows]


def delete_conversation(conversation_id: int) -> None:
    with _lock, connect() as c:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ── Daily meditation ────────────────────────────────────────────────────────


def get_meditation(date_str: str) -> dict | None:
    with connect() as c:
        row = c.execute(
            "SELECT date, citacao, parte, questao, artigo, paraphrase "
            "FROM meditations WHERE date = ?",
            (date_str,),
        ).fetchone()
        return dict(row) if row else None


def set_meditation(date_str: str, citacao: str, parte: str, questao: int, artigo: int) -> None:
    with _lock, connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO meditations(date, citacao, parte, questao, artigo) "
            "VALUES (?, ?, ?, ?, ?)",
            (date_str, citacao, parte, questao, artigo),
        )


def save_paraphrase(date_str: str, text: str) -> None:
    with _lock, connect() as c:
        c.execute(
            "UPDATE meditations SET paraphrase = ? WHERE date = ?",
            (text, date_str),
        )


def recent_meditation_citations(days: int = 60) -> set[str]:
    with connect() as c:
        rows = c.execute(
            "SELECT citacao FROM meditations ORDER BY date DESC LIMIT ?",
            (days,),
        ).fetchall()
        return {r["citacao"] for r in rows}
