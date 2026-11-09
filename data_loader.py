"""Audited ATLAS messages parsing. CLI freezes an explicit multi-code policy."""

from pathlib import Path
import json, csv, re, random
from hts_hierarchy import normalize_hts
from study_io import digest

ATLAS_USER_PREFIX = "What is the HTS US Code for "
CODE_RE = re.compile(
    r"(?<![0-9])(?:[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}|[0-9]{4}\.[0-9]{2}\.[0-9]{4}|[0-9]{10})(?![0-9])"
)


def _read_rows(path):
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        if p.suffix in (".jsonl", ".ndjson"):
            return [json.loads(x) for x in f if x.strip()]
        if p.suffix == ".json":
            x = json.load(f)
            return x if isinstance(x, list) else x.get("data", [])
        if p.suffix in (".csv", ".tsv"):
            return list(
                csv.DictReader(f, delimiter="\t" if p.suffix == ".tsv" else ",")
            )
    raise ValueError("Unsupported input format")


def parse_atlas_assistant(content):
    head, _, reason = content.partition("Reasoning ->")
    codes = []
    seen = set()
    for c in CODE_RE.findall(head):
        n = normalize_hts(c)
        if n not in seen:
            seen.add(n)
            codes.append(c)
    return codes, reason.strip()


def load_atlas_messages(path, split=None, multi_policy="first", return_audit=False):
    if multi_policy not in ("first", "exclude", "error"):
        raise ValueError("Invalid multi-code policy")
    raw = _read_rows(path)
    out = []
    excluded = []
    seen = set()
    for i, r in enumerate(raw):
        sid = str(r.get("id", i))
        cid = f"{split}:{sid}" if split else sid
        if cid in seen:
            raise ValueError("Duplicate source row ID")
        seen.add(cid)
        msgs = r.get("messages", [])
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        assistant = next(
            (m.get("content", "") for m in msgs if m.get("role") == "assistant"), ""
        )
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError("Messages must be text")
        desc = user.strip()
        if desc.startswith(ATLAS_USER_PREFIX):
            desc = desc[len(ATLAS_USER_PREFIX) :]
        desc = desc.rstrip("?").strip()
        codes, reason = parse_atlas_assistant(assistant)
        why = (
            "empty_description"
            if not desc
            else ("no_10_digit_gold" if not codes else None)
        )
        if len(codes) > 1:
            if multi_policy == "error":
                raise ValueError(f"Multi-code row {cid}")
            if multi_policy == "exclude":
                why = "multi_code_excluded"
        if why:
            excluded.append({"case_id": cid, "source_index": i, "reason": why})
            continue
        golds = [normalize_hts(x) for x in codes]
        out.append(
            {
                "case_id": cid,
                "source_row_id": sid,
                "source_index": i,
                "split": split,
                "description": desc,
                "description_hash": digest(" ".join(desc.casefold().split())),
                "gold": golds[0],
                "gold_all": golds,
                "multi_code": len(golds) > 1,
                "reasoning": reason,
                "source_ruling_id": (
                    str(r["source_ruling_id"]) if r.get("source_ruling_id") else None
                ),
            }
        )
    audit = {
        "input_count": len(raw),
        "included_count": len(out),
        "excluded_count": len(excluded),
        "multi_policy": multi_policy,
        "exclusions": excluded,
    }
    return (out, audit) if return_audit else out


def exclude_gold_ruling(evidence, gold_ruling_id):
    return [
        e
        for e in evidence
        if gold_ruling_id is None
        or str(e.get("source_ruling_id", e.get("id"))) != str(gold_ruling_id)
    ]


def stratified_judge_subset(records, n_per_stratum=15, conf_split=None, seed=7):
    correct = [r for r in records if r.get("exact_10", r.get("correct", False))]
    cs = sorted(r["confidence"] for r in correct)
    if conf_split is None:
        conf_split = cs[len(cs) // 2] if cs else 0.5
    pools = {
        "correct_high": [r for r in correct if r["confidence"] >= conf_split],
        "correct_low": [r for r in correct if r["confidence"] < conf_split],
        "near_miss": [
            r
            for r in records
            if r.get("severity_tier") in ("wrong_stat_suffix", "wrong_rate_line")
        ],
        "major_error": [
            r
            for r in records
            if r.get("severity_tier")
            in ("wrong_subheading", "wrong_heading", "wrong_chapter")
        ],
    }
    rng = random.Random(seed)
    out = []
    seen = set()
    for name, pool in pools.items():
        pool = sorted(pool, key=lambda r: (str(r["case_id"]), str(r.get("model", ""))))
        rng.shuffle(pool)
        for row in pool[:n_per_stratum]:
            key = (row["case_id"], row.get("model"))
            if key not in seen:
                seen.add(key)
                out.append({**row, "stratum": name})
    return out
