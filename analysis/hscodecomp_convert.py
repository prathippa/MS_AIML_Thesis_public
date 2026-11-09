"""
hscodecomp_convert.py
---------------------
Convert HSCodeComp (Lan et al., 2026; 632 e-commerce products, expert U.S. HTS
10-digit labels) into the ATLAS messages format that run_study.py `prepare`
reads, and make the fixed-seed validation/test split the calibration stage needs
(HSCodeComp ships as a single test set).

Description = product title + marketplace category path + listed attributes +
listed price. Kept noisy on purpose (that is the robustness condition); only
fields that carry no product information are dropped (internal unit IDs, null
brand placeholders, the benchmark's tool-use instructions in `question`).

Split: stratified by top-level category, seed 7, half to validation and half to
test (odd category sizes give the extra case to test).

    python analysis/hscodecomp_convert.py data/hscodecomp/test_data.jsonl data/hscodecomp
"""

from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path

SEED = 7
DROP_ATTRS = {"Measurement unit", "Hign-concerned Chemical", "High-concerned Chemical", "CN"}
NULL_VALUES = {"", "none", "null", "noenname_null", "n/a"}


def describe(r):
    parts = [r["product_name"].strip()]
    cats = [r.get(f"cate_lv{i}_desc") for i in range(1, 6)]
    path = []
    for c in cats:
        if c and c not in path:
            path.append(c)
    if path:
        parts.append("Category: " + " > ".join(path))
    try:
        attrs = json.loads(r.get("product_attributes") or "{}")
    except json.JSONDecodeError:
        attrs = {}
    kept = [f"{k}: {v}" for k, v in attrs.items()
            if k not in DROP_ATTRS and str(v).strip().casefold() not in NULL_VALUES]
    if kept:
        parts.append("Attributes: " + "; ".join(kept))
    if r.get("price"):
        parts.append(f"Listed price: {r['price']} {r.get('currency_code', '')}".strip())
    return ". ".join(parts)


def dotted(code):
    code = str(code)  # stored as int in the source; all 632 are 10 digits (no leading-zero chapters)
    return f"{code[:4]}.{code[4:6]}.{code[6:]}"


def main(src, out_dir):
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r.get("cate_lv1_desc") or "?"].append(r)
    rng = random.Random(SEED)
    split = {"validation": [], "test": []}
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda r: int(r["task_id"]))
        rng.shuffle(group)
        half = len(group) // 2
        split["validation"] += group[:half]
        split["test"] += group[half:]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in split.items():
        part.sort(key=lambda r: int(r["task_id"]))
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({
                    "id": f"hscc-{r['task_id']}",
                    "messages": [
                        {"role": "user", "content": f"What is the HTS US Code for {describe(r)}?"},
                        {"role": "assistant", "content": f"HTS US Code -> {dotted(r['hs_code'])}\nReasoning -> "},
                    ],
                }, ensure_ascii=False) + "\n")
        print(name, len(part))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
