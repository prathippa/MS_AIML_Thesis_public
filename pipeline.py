"""Sequential real orchestration; offline tests inject mocked clients explicitly."""

from pathlib import Path
import numpy as np
import config, metrics as M, calibration as C, routing as R
from study_io import *
from data_loader import load_atlas_messages, stratified_judge_subset
from hts_hierarchy import evaluate
from generation import get_client, run_generation
from budget import Budget, BudgetedClient


def init(
    root,
    revision,
    evidence_mode="retrieved",
    multi_policy="exclude",
    max_calls=2000,
    human_size=30,
    method="platt",
    n_boot=2000,
):
    root = Path(root)
    if (root / "protocol.json").exists():
        raise ValueError("Use a NEW experiment directory")
    if (
        not str(revision).strip()
        or evidence_mode not in ("none", "retrieved")
        or multi_policy not in ("exclude", "first", "error")
        or method not in ("platt", "isotonic")
        or min(max_calls, human_size, n_boot) < 1
    ):
        raise ValueError("Invalid protocol")
    models = {"A": config.MODEL_A, "B": config.MODEL_B, "J": config.JUDGE_MODEL}
    for label, c in models.items():
        if c["provider"] not in config.DECODING_BY_PROVIDER:
            raise ValueError("Unsupported provider")
        if label != "J" and not c["model"]:
            raise ValueError("Set verified HTS_MODEL_A and HTS_MODEL_B before init")
    protocol = {
        "version": "2.1-rebuilt",
        "created_at": utcnow(),
        "dataset_revision": revision,
        "evidence_mode": evidence_mode,
        "multi_policy": multi_policy,
        "models": models,
        "decoding": config.DECODING_BY_PROVIDER,
        "method": method,
        "max_calls": max_calls,
        "human_size": human_size,
        "n_boot": n_boot,
        "seed": 7,
        "judge_runs": 3,
        "judge_per_stratum": 15,
        "coverage": config.PRIMARY_COVERAGE,
        "coverages": config.COVERAGE_POINTS,
        "target_risk": 0.10,
        "routing_unit": "model_case_prediction",
        "failure_policy": "Failed/missing = wrong in all-case accuracy; always escalate",
        "source_hashes": source_hashes(),
        "ci_scope": "Conditional on frozen calibrators; pooled bootstrap clusters product cases",
    }
    write(root / "protocol.json", protocol)
    write(root / "protocol.sha.json", {"sha256": digest(protocol)})
    write(root / "environment.json", environment())
    return protocol


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.protocol = read(self.root / "protocol.json")
        if digest(self.protocol) != read(self.root / "protocol.sha.json")["sha256"]:
            raise ValueError("Protocol changed")
        if source_hashes() != self.protocol["source_hashes"]:
            raise ValueError("Code changed after freeze; start a new experiment")

    def path(self, split, name):
        if split not in ("validation", "test"):
            raise ValueError("Explicit split required")
        return self.root / split / name

    def seal(self, name, inputs, files):
        write(
            self.root / f"{name}.seal.json",
            {
                "inputs": digest(inputs),
                "files": {f: file_hash(self.root / f) for f in files},
            },
        )

    def verify(self, name, inputs):
        v = read(self.root / f"{name}.seal.json")
        if v["inputs"] != digest(inputs) or any(
            file_hash(self.root / f) != h for f, h in v["files"].items()
        ):
            raise ValueError(f"{name} outputs/inputs changed")

    def prepare(self, validation, test):
        if file_hash(validation) == file_hash(test):
            raise ValueError("Identical validation and test inputs")
        both = {}
        audits = {}
        for s, p in [("validation", validation), ("test", test)]:
            rows, a = load_atlas_messages(p, s, self.protocol["multi_policy"], True)
            if not rows:
                raise ValueError("No retained cases")
            descriptions = [r["description_hash"] for r in rows]
            rids = [r["source_ruling_id"] for r in rows if r.get("source_ruling_id")]
            if len(set(descriptions)) != len(descriptions) or len(set(rids)) != len(
                rids
            ):
                raise ValueError(
                    "Duplicate description/source ruling within split; document policy"
                )
            both[s] = rows
            audits[s] = {**a, "file_hash": file_hash(p), "records_hash": digest(rows)}
        for key in ("description_hash", "source_ruling_id"):
            a = {r[key] for r in both["validation"] if r.get(key)}
            b = {r[key] for r in both["test"] if r.get(key)}
            if a & b:
                raise ValueError(
                    "Validation/test overlap; do not silently resplit benchmark"
                )
        for s in both:
            bind(self.path(s, "input.manifest.json"), audits[s])
            write_lines(self.path(s, "records.jsonl"), both[s])
            write(self.path(s, "load_audit.json"), audits[s])
        return audits

    def records(self, s):
        rows = lines(self.path(s, "records.jsonl"))
        if (
            not rows
            or digest(rows)
            != verify_bound(self.path(s, "input.manifest.json"))["records_hash"]
        ):
            raise ValueError("Prepared records missing or changed")
        return rows

    def evidence(self, path=None, k=3):
        from evidence import corpus, Retriever

        rows = {s: self.records(s) for s in ("validation", "test")}
        retriever = None
        items = []
        if self.protocol["evidence_mode"] == "retrieved":
            if not path:
                raise ValueError("Sourced evidence corpus required")
            items = corpus(path, rows["validation"] + rows["test"])
            retriever = Retriever(items, k)
        elif path:
            raise ValueError("Protocol froze no-evidence baseline")
        bind(
            self.root / "evidence.manifest.json",
            {
                "corpus_hash": digest(items),
                "records": digest(rows),
                "top_k": k,
                "method": "TF-IDF unigram+bigram cosine",
            },
        )
        write_lines(self.root / "corpus_snapshot.jsonl", items)
        for s, rs in rows.items():
            packs = {
                r["case_id"]: retriever.retrieve(r["description"]) if retriever else []
                for r in rs
            }
            bind(self.path(s, "evidence.seal.json"), packs)
            write(self.path(s, "evidence.json"), packs)
        return {"corpus_items": len(items)}

    def packs(self, s):
        packs = read(self.path(s, "evidence.json"))
        if digest(packs) != digest(verify_bound(self.path(s, "evidence.seal.json"))):
            raise ValueError("Evidence changed")
        return packs

    def client(self, label, stage, provided=None):
        c = self.protocol["models"][label]
        if not c["model"]:
            raise ValueError("Configure an independent JUDGE model before init")
        cli = provided if provided is not None else get_client(c)
        return (
            BudgetedClient(
                cli,
                Budget(self.root / "requests.sqlite", self.protocol["max_calls"]),
                {"stage": stage, "label": label},
            ),
            self.protocol["decoding"][c["provider"]],
        )

    def generate(self, s, label, client=None):
        if label not in ("A", "B"):
            raise ValueError("Generator label must be A/B")
        rows = self.records(s)
        packs = self.packs(s)
        c, dec = self.client(label, "generation", client)
        # Never pass gold codes, gold reasoning or source rulings into generation.
        safe = [
            {"case_id": r["case_id"], "description": r["description"]} for r in rows
        ]
        run_generation(
            c,
            safe,
            packs,
            dec,
            str(self.path(s, f"generations_{label}.jsonl")),
            max_calls=self.protocol["max_calls"],
        )
        self.seal(
            f"generation_{s}_{label}",
            {"records": rows, "packs": packs},
            [
                f"{s}/generations_{label}.jsonl",
                f"{s}/generations_{label}.jsonl.manifest.json",
            ],
        )
        return {
            "cached": len(lines(self.path(s, f"generations_{label}.jsonl"))),
            "requests_used": c.budget.used,
        }

    def joined(self, s, label):
        cases = self.records(s)
        gens = lines(self.path(s, f"generations_{label}.jsonl"))
        packs = self.packs(s)
        if self.path(s, f"generations_{label}.jsonl").exists():
            self.verify(f"generation_{s}_{label}", {"records": cases, "packs": packs})
        if len({g["case_id"] for g in gens}) != len(gens):
            raise ValueError("Duplicate generation ID")
        by = {g["case_id"]: g for g in gens}
        ids = {r["case_id"] for r in cases}
        if set(by) - ids:
            raise ValueError("Foreign generation case IDs")
        out = []
        for r in cases:
            g = by.get(r["case_id"])
            status = g.get("status") if g else "missing"
            if status not in ("ok", "failed", "missing"):
                raise ValueError("Unknown generation status")
            row = {
                **r,
                "model": label,
                "prediction_id": f'{r["case_id"]}|{label}',
                "status": status,
                "correct": 0,
                "pred": None,
                "confidence": None,
                "rationale": None,
                "evidence_ids": [],
            }
            if status == "ok":
                from generation import parse_response, build_messages, _prompt_hash

                parsed = parse_response(g["raw"])
                sy, us = build_messages(
                    {"description": r["description"]}, packs[r["case_id"]]
                )
                if _prompt_hash(sy, us) != g["prompt_hash"]:
                    raise ValueError("Cached prompt mismatch")
                if set(parsed["evidence_ids"]) - {e["id"] for e in packs[r["case_id"]]}:
                    raise ValueError("Foreign evidence cited")
                if any(parsed[k] != g[k] for k in parsed):
                    raise ValueError("Cached parsed output changed")
                row.update(
                    pred=parsed["hts_code"],
                    confidence=parsed["confidence"],
                    rationale=parsed["rationale"],
                    evidence_ids=parsed["evidence_ids"],
                )
                row.update(evaluate(row["pred"], row["gold"]))
                row["correct"] = int(row["exact_10"])
            out.append(row)
        return out

    def metrics(self):
        inputs = {k: self.joined("test", k) for k in ("A", "B")}
        summary = {}
        for k, rs in inputs.items():
            valid = [r for r in rs if r["status"] == "ok"]
            n = len(rs)
            summary[k] = {
                "n_all": n,
                "n_valid": len(valid),
                "n_failed": sum(r["status"] == "failed" for r in rs),
                "n_missing": sum(r["status"] == "missing" for r in rs),
                "all_case_exact10_accuracy": sum(r["correct"] for r in rs) / n,
                "valid_output_metrics": M.aggregate(valid) if valid else None,
            }
        write(self.root / "metrics.json", summary)
        a, b = inputs["A"], inputs["B"]
        rng = np.random.default_rng(self.protocol["seed"])
        diff = np.array([x["correct"] - y["correct"] for x, y in zip(a, b)])
        boots = [
            float(diff[rng.integers(0, len(diff), len(diff))].mean())
            for _ in range(self.protocol["n_boot"])
        ]
        write(
            self.root / "paired_accuracy.json",
            {
                "difference_A_minus_B": float(diff.mean()),
                "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                "unit": "paired_product",
            },
        )
        self.seal("metrics", inputs, ["metrics.json", "paired_accuracy.json"])
        return summary

    def calibrate(self):
        inputs = {
            s: {k: self.joined(s, k) for k in ("A", "B")}
            for s in ("validation", "test")
        }
        out = {}
        files = ["calibration.json"]
        for k in ("A", "B"):
            val = [r for r in inputs["validation"][k] if r["status"] == "ok"]
            test = [r for r in inputs["test"][k] if r["status"] == "ok"]
            if not val or not test:
                raise ValueError("Valid validation and test generations needed")
            cv = [r["confidence"] for r in val]
            yv = [r["correct"] for r in val]
            ct = [r["confidence"] for r in test]
            yt = [r["correct"] for r in test]
            cal = C.fit_calibrator(cv, yv, method=self.protocol["method"])
            pt = cal.transform(ct)
            state = (
                cal.state()
                if hasattr(cal, "state")
                else {
                    "method": "isotonic",
                    "x": cal._iso.X_thresholds_.tolist(),
                    "y": cal._iso.y_thresholds_.tolist(),
                }
            )
            write(self.root / f"calibrator_{k}.json", state)
            files.append(f"calibrator_{k}.json")
            out[k] = {
                "n_validation_valid": len(val),
                "n_test_valid": len(test),
                "method": cal.method,
                "raw": {"ece": C.ece(ct, yt), "brier": C.brier(ct, yt)},
                "calibrated": {"ece": C.ece(pt, yt), "brier": C.brier(pt, yt)},
                "state": state,
            }
            for s in ("validation", "test"):
                rs = inputs[s][k]
                vs = [r for r in rs if r["status"] == "ok"]
                pred = iter(cal.transform([r["confidence"] for r in vs]))
                scored = [
                    {
                        **r,
                        "calibrated_confidence": (
                            float(next(pred)) if r["status"] == "ok" else None
                        ),
                    }
                    for r in rs
                ]
                f = f"{s}/scored_{k}.jsonl"
                write_lines(self.root / f, scored)
                files.append(f)
        write(self.root / "calibration.json", out)
        self.seal("calibration", inputs, files)
        return out

    def scored(self, s):
        inputs = {
            sp: {k: self.joined(sp, k) for k in ("A", "B")}
            for sp in ("validation", "test")
        }
        self.verify("calibration", inputs)
        return [r for k in ("A", "B") for r in lines(self.path(s, f"scored_{k}.jsonl"))]

    def route(self):
        both = {s: self.scored(s) for s in ("validation", "test")}

        def pool(rs):
            return [
                {**r, "raw": r["confidence"], "cal": r["calibrated_confidence"]}
                for r in rs
                if r["status"] == "ok"
            ]

        val, test = pool(both["validation"]), pool(both["test"])
        if not val or not test:
            raise ValueError("Nonempty valid pools required")
        out = R.compare_pooled_routing(test, self.protocol["coverage"])
        gap = R.bootstrap_routing_gap(
            test, self.protocol["coverage"], self.protocol["n_boot"]
        )
        out.update(
            ci95_raw_minus_calibrated=list(gap[1:]),
            n_case_clusters=len({r["case_id"] for r in test}),
            ci_scope=self.protocol["ci_scope"],
        )
        ids = [r["prediction_id"] for r in test]
        y = [r["correct"] for r in test]
        for key in ("raw", "cal"):
            curve = R.risk_coverage_curve(
                [r[key] for r in test], y, self.protocol["coverages"], ids
            )
            out["curve_" + key] = [
                {**p, "all_prediction_coverage": p["n_streamlined"] / len(both["test"])}
                for p in curve
            ]
            frozen = R.threshold_for_target_risk(
                [r[key] for r in val],
                [r["correct"] for r in val],
                self.protocol["target_risk"],
            )
            applied = R.apply_threshold([r[key] for r in test], y, frozen["threshold"])
            out["threshold_" + key] = {
                "validation": frozen,
                "test": {
                    **applied,
                    "all_prediction_coverage": applied["n_streamlined"]
                    / len(both["test"]),
                },
                "guaranteed_risk_control": False,
            }
        out["denominators"] = {
            s: {
                "all_predictions": len(rs),
                "valid_predictions": sum(r["status"] == "ok" for r in rs),
                "forced_escalations": sum(r["status"] != "ok" for r in rs),
            }
            for s, rs in both.items()
        }
        out["within_model"] = {
            k: R.compare_pooled_routing(
                [r for r in test if r["model"] == k], self.protocol["coverage"]
            )
            for k in ("A", "B")
        }
        write(self.root / "routing.json", out)
        self.seal("routing", both, ["routing.json"])
        return out

    def prepare_judge(self):
        if self.protocol["evidence_mode"] == "none":
            raise ValueError("RQ3 needs sourced evidence; none mode is baseline-only")
        rows = [
            r for k in ("A", "B") for r in self.joined("test", k) if r["status"] == "ok"
        ]
        packs = self.packs("test")
        chosen = stratified_judge_subset(
            rows, self.protocol["judge_per_stratum"], seed=self.protocol["seed"]
        )
        cases = [
            {
                "case_id": r["case_id"],
                "prediction_id": r["prediction_id"],
                "description": r["description"],
                "predicted_code": r["pred"],
                "rationale": r["rationale"],
                "evidence": packs[r["case_id"]],
                "generator_model": self.protocol["models"][r["model"]]["model"],
                "stratum": r["stratum"],
            }
            for r in chosen
        ]
        bind(self.root / "judge_sample.manifest.json", cases)
        write_lines(self.root / "judge_cases.jsonl", cases)
        return {"n": len(cases)}

    def judge_cases(self):
        rows = lines(self.root / "judge_cases.jsonl")
        if digest(rows) != digest(
            verify_bound(self.root / "judge_sample.manifest.json")
        ):
            raise ValueError("Judge sample changed")
        return rows

    def judge(self, label="J", client=None):
        from judge import judge_one

        rows = self.judge_cases()
        c, dec = self.client(label, "judge", client)
        bind(
            self.root / "judge_run.manifest.json",
            {
                "cases": rows,
                "model": c.model,
                "decoding": dec,
                "runs": self.protocol["judge_runs"],
            },
        )
        previous = lines(self.root / "judge_results.jsonl")
        ids = [r["prediction_id"] for r in previous]
        if len(ids) != len(set(ids)) or set(ids) - {r["prediction_id"] for r in rows}:
            raise ValueError("Invalid judge cache IDs")
        for i, r in enumerate(rows):
            if r["prediction_id"] in ids:
                continue
            if c.budget.used + self.protocol["judge_runs"] > self.protocol["max_calls"]:
                from budget import BudgetExceeded

                raise BudgetExceeded(
                    "Not enough calls for a complete repeated-judge unit"
                )
            result = judge_one(
                c, r, dec, self.protocol["judge_runs"], seed=self.protocol["seed"] + i
            )
            previous.append(
                {**result, "judge_is_generator": c.model == r["generator_model"]}
            )
            write_lines(self.root / "judge_results.jsonl", previous)
        self.seal("judge", rows, ["judge_results.jsonl"])
        return {
            "n": len(previous),
            "abstentions": sum(r["majority"] is None for r in previous),
        }

    def export_human(self):
        from annotations import export

        return export(
            self.judge_cases(),
            self.root / "annotations",
            self.protocol["human_size"],
            self.protocol["seed"],
        )

    def validate_human(self, path, attestation):
        from annotations import validate

        self.verify("judge", self.judge_cases())
        judges = lines(self.root / "judge_results.jsonl")
        result = validate(
            path,
            self.root / "annotations" / "key.json",
            judges,
            read(attestation),
            self.protocol["n_boot"],
        )
        bind(
            self.root / "human_import.manifest.json",
            {"file": file_hash(path), "attestation": file_hash(attestation)},
        )
        write(self.root / "human_validation.json", result)
        self.seal(
            "human",
            {"judge": judges, "key": read(self.root / "annotations" / "key.json")},
            ["human_validation.json"],
        )
        return result
