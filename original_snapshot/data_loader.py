"""
data_loader.py
--------------
Stage 1 helpers: load ATLAS/CROSS into the record shape the pipeline expects,
and draw the stratified evidence-support subset.

IMPORTANT (confirmed against the live dataset, Oct 2026): the ATLAS dataset
(flexifyai/cross_rulings_hts_dataset_for_tariffs) is a chat/SFT dataset. Each
row is {"messages": [user, assistant]}, and the gold code(s) and reasoning are
embedded in the ASSISTANT TEXT as:

    HTS US Code -> 8477.10.9015; 8480.71.8045
    Reasoning -> ...

So the gold label is parsed out of text, not read from a column. Many rows carry
MULTIPLE gold codes; `load_atlas_messages` records them all and flags the row,
and uses the first as the primary single-code gold (see `multi_code`).

Standard record: {case_id, description, gold, gold_all, multi_code, reasoning}
"""
from __future__ import annotations
import os, sys, json, csv, re, random
sys.path.insert(0, os.path.dirname(__file__))
from hts_hierarchy import normalize_hts  # noqa: E402

ATLAS_USER_PREFIX = "What is the HTS US Code for "
# A 10-digit HTS code, dotted (8477.10.9015) or bare (8477109015).
CODE_RE = re.compile(r"\d{4}\.\d{2}\.\d{4}|\b\d{10}\b")


def _read_rows(path: str) -> list[dict]:
    if path.endswith((".jsonl", ".ndjson")):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("data", [])
    if path.endswith((".csv", ".tsv")):
        with open(path, newline="") as f:
            return list(csv.DictReader(f, delimiter="\t" if path.endswith(".tsv") else ","))
    raise ValueError(f"Unsupported file type: {path}")


def parse_atlas_assistant(content: str):
    """Extract (codes, reasoning) from an assistant turn. Robust to multiple
    codes and to parenthetical '$10'-style noise in the code line."""
    head, _, reasoning = content.partition("Reasoning ->")
    codes = CODE_RE.findall(head)                 # only code-shaped tokens
    seen, uniq = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq, reasoning.strip()


def load_atlas_messages(path: str) -> list[dict]:
    """Load the real ATLAS chat/SFT format into standard records."""
    out = []
    for i, r in enumerate(_read_rows(path)):
        msgs = r.get("messages", [])
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        desc = user.strip()
        if desc.startswith(ATLAS_USER_PREFIX):
            desc = desc[len(ATLAS_USER_PREFIX):]
        desc = desc.rstrip("?").strip()
        codes, reasoning = parse_atlas_assistant(asst)
        if not codes or not desc:
            continue                              # skip unparseable rows
        gold_all = [normalize_hts(c) for c in codes]
        out.append({
            "case_id": str(r.get("id", i)),
            "description": desc,
            "gold": gold_all[0],                  # primary single-code gold
            "gold_all": gold_all,
            "multi_code": len(gold_all) > 1,
            "reasoning": reasoning,
        })
    return out


def exclude_gold_ruling(evidence: list[dict], gold_ruling_id: str) -> list[dict]:
    """Guarantee the model/judge never sees the case's own answer ruling."""
    return [e for e in evidence if str(e.get("id")) != str(gold_ruling_id)]


def stratified_judge_subset(records, n_per_stratum=15, conf_split=None, seed=7):
    """Four-strata evidence-support subset (Stage 5). Each record needs:
    case_id, exact_10 (bool), severity_tier, confidence."""
    corrects = [r for r in records if r.get("exact_10")]
    if conf_split is None:
        confs = sorted(r["confidence"] for r in corrects)
        conf_split = confs[len(confs) // 2] if confs else 0.5

    near = {"wrong_stat_suffix", "wrong_rate_line"}
    major = {"wrong_heading", "wrong_chapter"}
    strata = {
        "correct_high": [r for r in corrects if r["confidence"] >= conf_split],
        "correct_low":  [r for r in corrects if r["confidence"] < conf_split],
        "near_miss":    [r for r in records if not r.get("exact_10") and r["severity_tier"] in near],
        "major_error":  [r for r in records if r["severity_tier"] in major],
    }
    rng = random.Random(seed)
    picked, seen = [], set()
    for name, pool in strata.items():
        rng.shuffle(pool)
        for r in pool[:n_per_stratum]:
            if r["case_id"] not in seen:
                picked.append({**r, "stratum": name})
                seen.add(r["case_id"])
    return picked
