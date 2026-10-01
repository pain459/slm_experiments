from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fingerprints(prompt: str, response: str) -> dict[str, str]:
    p = normalize_text(prompt)
    r = normalize_text(response)
    return {
        "prompt_sha256": sha(p),
        "response_sha256": sha(r),
        "pair_sha256": sha(p + "\n---\n" + r),
    }


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", normalize_text(value)))


def jaccard(a: str, b: str) -> float:
    x, y = token_set(a), token_set(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


@dataclass
class DedupDecision:
    duplicate: bool
    reason: str | None = None
    matched_id: str | None = None
    similarity: float | None = None


class StreamingDeduper:
    """Exact dedup globally, optional cheap near-dedup within a curriculum bucket."""

    def __init__(self, near_threshold: float | None = 0.94, near_window: int = 200):
        self.prompt_hashes: dict[str, str] = {}
        self.response_hashes: dict[str, str] = {}
        self.pair_hashes: dict[str, str] = {}
        self.bucket_prompts: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.near_threshold = near_threshold
        self.near_window = near_window

    def check(self, row_id: str, prompt: str, response: str, bucket: str) -> tuple[DedupDecision, dict[str, str]]:
        fps = fingerprints(prompt, response)
        for field, store in (
            ("prompt_sha256", self.prompt_hashes),
            ("pair_sha256", self.pair_hashes),
        ):
            if fps[field] in store:
                return DedupDecision(True, field, store[fps[field]], 1.0), fps

        if self.near_threshold is not None:
            for existing_id, old_prompt in self.bucket_prompts[bucket][-self.near_window:]:
                score = jaccard(prompt, old_prompt)
                if score >= self.near_threshold:
                    return DedupDecision(True, "near_prompt", existing_id, score), fps

        self.prompt_hashes[fps["prompt_sha256"]] = row_id
        self.response_hashes[fps["response_sha256"]] = row_id
        self.pair_hashes[fps["pair_sha256"]] = row_id
        self.bucket_prompts[bucket].append((row_id, prompt))
        return DedupDecision(False), fps
