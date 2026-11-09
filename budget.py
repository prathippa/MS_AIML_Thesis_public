"""Persistent conservative request cap, not a currency/token cap. No SDK retries."""

import sqlite3, json
from pathlib import Path
from study_io import utcnow, digest


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path, limit):
        self.path = str(path)
        self.limit = int(limit)
        if self.limit < 1:
            raise ValueError("Positive call cap required")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, cap INTEGER)"
            )
            db.execute("INSERT OR IGNORE INTO settings VALUES (1,?)", (self.limit,))
            if (
                db.execute("SELECT cap FROM settings WHERE id=1").fetchone()[0]
                != self.limit
            ):
                raise ValueError("Budget cap frozen")
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY,time TEXT,context TEXT,status TEXT,metadata TEXT)"
            )

    @property
    def used(self):
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def reserve(self, context):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] >= self.limit:
                raise BudgetExceeded("Persistent request cap reached")
            cur = db.execute(
                "INSERT INTO calls(time,context,status,metadata) VALUES (?,?,?,?)",
                (utcnow(), json.dumps(context), "reserved", "{}"),
            )
            return cur.lastrowid

    def finish(self, i, status, meta):
        with sqlite3.connect(self.path) as db:
            db.execute(
                "UPDATE calls SET status=?,metadata=? WHERE id=?",
                (status, json.dumps(meta, default=str), i),
            )


class BudgetedClient:
    def __init__(self, client, budget, context):
        self.client, self.budget, self.context = client, budget, context
        self.model = self.name = client.model
        self.last_metadata = {}

    def complete(self, system, user, decoding):
        self.last_metadata = {}
        if hasattr(self.client, "request_settings"):
            self.client.request_settings(decoding)
        ticket = self.budget.reserve(
            {**self.context, "model": self.model, "prompt_hash": digest([system, user])}
        )
        try:
            result = self.client.complete(system, user, decoding)
            self.last_metadata = {
                **getattr(self.client, "last_metadata", {}),
                "request_ticket": ticket,
            }
            self.budget.finish(ticket, "response", self.last_metadata)
            return result
        except Exception as e:
            self.last_metadata = {
                "request_ticket": ticket,
                "exception": type(e).__name__,
            }
            self.budget.finish(ticket, "error", self.last_metadata)
            raise
