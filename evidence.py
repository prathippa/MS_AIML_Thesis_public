"""Reproducible TF-IDF retrieval. Exact source and text exclusions only.
Near-duplicates and unavailable source metadata still require manual auditing.
"""

import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from study_io import lines, digest


def normalized(s):
    return " ".join(re.findall(r"\w+", s.casefold()))


def corpus(path, heldout):
    rows = lines(path)
    if not rows:
        raise ValueError("Empty evidence corpus")
    ids = set()
    texts = set()
    blocked = {r["source_ruling_id"] for r in heldout if r.get("source_ruling_id")}
    missing = sum(not r.get("source_ruling_id") for r in heldout)
    if missing:
        raise ValueError(
            "Retrieved-evidence protocol requires explicit source_ruling_id on EVERY held-out row. Do not guess IDs or weaken this guard; use no-evidence baseline while repairing provenance."
        )
    for r in rows:
        required = (
            "id",
            "text",
            "source_url",
            "source_ruling_id",
            "accessed_at",
            "version",
        )
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in required):
            raise ValueError(
                "Evidence needs id/text/source_url/source_ruling_id/accessed_at/version strings"
            )
        url = urlparse(r["source_url"])
        if url.scheme not in ("https", "http") or not url.netloc:
            raise ValueError("Traceable source URL required")
        if r["id"] in ids or normalized(r["text"]) in texts:
            raise ValueError("Duplicate evidence IDs/text")
        ids.add(r["id"])
        texts.add(normalized(r["text"]))
        if r["source_ruling_id"] in blocked:
            raise ValueError("Held-out source ruling found in evidence corpus")
        for h in heldout:
            if normalized(h["description"]) in normalized(r["text"]):
                raise ValueError("Held-out description copied into evidence")
            if h.get("reasoning") and normalized(h["reasoning"]) in normalized(
                r["text"]
            ):
                raise ValueError("Held-out gold reasoning copied into evidence")
    return rows


class Retriever:
    def __init__(self, items, k=3):
        if k < 1:
            raise ValueError("Positive top-k required")
        self.items, self.k = items, k
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in items])

    def retrieve(self, description):
        scores = cosine_similarity(
            self.vectorizer.transform([description]), self.matrix
        )[0]
        ix = sorted(
            range(len(scores)), key=lambda i: (-scores[i], self.items[i]["id"])
        )[: self.k]
        return [
            {**self.items[i], "retrieval_score": float(scores[i])}
            for i in ix
            if scores[i] > 0
        ]
