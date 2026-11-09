"""
dry_run_demo.py -- validate the pipeline on REAL ATLAS rows with a mock model.

No API/dataset needed: it loads real ATLAS-format rows from examples/atlas_sample.jsonl
and substitutes a deterministic MockClient for the generative model, so it proves the
load -> generate -> metrics -> calibrate -> route -> judge wiring works on the true
data format. Replace MockClient with the real client (config.MODEL_A) for the sprint.

    python examples/dry_run_demo.py
"""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_atlas_messages
from generation import generate_one
import metrics as M
import calibration as C
import routing as R
from hts_hierarchy import normalize_hts, severity


class ScriptedModel:
    """Deterministic stand-in: returns gold / near-miss / wrong per case, with a
    model-specific confidence pattern, as a structured JSON reply."""
    def __init__(self, name, overconfident=False):
        self.model, self.name, self.over = name, name, overconfident
    def complete(self, system, user, decoding):
        # the demo stashes the gold code for this case on the instance
        g = self._gold
        r = random.Random(hash((self.model, g)) & 0xffffffff)
        roll = r.random()
        if roll < 0.55:
            pred, conf = g, 0.92
        elif roll < 0.80:
            pred, conf = g[:6] + ("00" if g[6:8] != "00" else "11") + g[8:], 0.70  # near-miss (rate line)
        else:
            pred, conf = ("9999" + g[4:]), 0.55                                      # wrong heading/chapter
        if self.over:
            conf = min(0.97, conf + 0.25)
        return json.dumps({"hts_code": pred, "confidence": conf,
                           "rationale": "scripted demo", "evidence_ids": []})


def run(model, records, decoding):
    recs = []
    for rec in records:
        model._gold = rec["gold"]
        g = generate_one(model, {"id": rec["case_id"], "description": rec["description"]},
                         [], decoding, retry_once=True)
        recs.append({"case_id": rec["case_id"], "model": model.name, "gold": rec["gold"],
                     "pred": g.hts_code, "confidence": g.confidence,
                     "correct": int(g.hts_code == rec["gold"])})
    return recs


def main():
    path = os.path.join(os.path.dirname(__file__), "atlas_sample.jsonl")
    records = load_atlas_messages(path)
    dec = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 200}

    print(f"Loaded {len(records)} real ATLAS rows "
          f"({sum(r['multi_code'] for r in records)} multi-code).")
    print("Example gold codes:", [r["gold"] for r in records[:5]])

    a = run(ScriptedModel("Model A"), records, dec)
    b = run(ScriptedModel("Model B", overconfident=True), records, dec)

    print("\n[Metrics] Model A severity distribution:")
    for tier, p in M.severity_distribution(M.eval_records(a)).items():
        if p:
            print(f"   {tier:18s} {p:.0%}")
    print(f"   exact_10  A={M.conventional_baselines(M.eval_records(a))['exact_10']:.2f}  "
          f"B={M.conventional_baselines(M.eval_records(b))['exact_10']:.2f}")

    # calibrate each model (demo splits the sample in half as val/test)
    def split(recs):
        h = len(recs) // 2
        return recs[:h], recs[h:]
    results = {}
    items = []
    for lbl, recs in (("A", a), ("B", b)):
        val, test = split(recs)
        res = C.calibrate_and_eval([r["confidence"] for r in val], [r["correct"] for r in val],
                                   [r["confidence"] for r in test], [r["correct"] for r in test],
                                   method="platt")
        results[lbl] = res
        items += [{"raw": r["confidence"], "cal": float(c), "correct": r["correct"]}
                  for r, c in zip(test, res["calibrated_test_confidence"])]
        print(f"[Calibration] Model {lbl}: test ECE {res['raw']['ece']:.3f} -> {res['calibrated']['ece']:.3f}")

    route = R.compare_pooled_routing(items, coverages=[0.5, 0.7])
    print(f"[Routing] pooled @70%: raw risk={route['selective_risk_raw']:.3f}  "
          f"calibrated={route['selective_risk_calibrated']:.3f}")
    print("\nDry run OK: real ATLAS format flows through the full pipeline.")


if __name__ == "__main__":
    main()
