"""Minimal Ollama chat client: temperature 0, fixed seed, JSONL cache.

Cache key is (model, system_date, question_id). Each cache row also stores a hash of the
exact messages + options; a row whose hash no longer matches is ignored and re-queried,
so editing a prompt can never silently reuse a stale answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

from . import date_label

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BASE_SYSTEM = "You are a helpful assistant."
OPTIONS = {"temperature": 0, "seed": 0, "num_predict": 32, "num_ctx": 2048}


def system_prompt(system_date: str | None) -> str:
    if system_date is None:
        return BASE_SYSTEM
    return f"{BASE_SYSTEM} Today's date is {system_date}."


def build_messages(system_date: str | None, question_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(system_date)},
        {"role": "user", "content": question_text},
    ]


def request_hash(model: str, messages: list[dict], options: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "options": options}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: dict[tuple, dict] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        self.rows[(r["model"], r["system_date"], r["question_id"])] = r

    def get(self, model: str, system_date: str | None, qid: str, rhash: str) -> dict | None:
        r = self.rows.get((model, date_label(system_date), qid))
        if r is not None and r.get("request_hash") == rhash:
            return r
        return None

    def put(self, row: dict) -> None:
        self.rows[(row["model"], row["system_date"], row["question_id"])] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def chat(model: str, messages: list[dict], options: dict = OPTIONS, timeout: float = 300) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "options": options, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask(cache: Cache, model: str, system_date: str | None, question: dict, chat_fn=chat) -> dict:
    messages = build_messages(system_date, question["text"])
    rhash = request_hash(model, messages, OPTIONS)
    hit = cache.get(model, system_date, question["id"], rhash)
    if hit is not None:
        return hit
    response = chat_fn(model, messages, OPTIONS)
    row = {
        "model": model,
        "system_date": date_label(system_date),
        "question_id": question["id"],
        "request_hash": rhash,
        "response": response,
    }
    cache.put(row)
    return row
