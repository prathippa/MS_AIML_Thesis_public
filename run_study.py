"""Real split-aware CLI. All root options precede the stage name."""

import argparse, json
from pipeline import Study, init
from study_io import lock, clean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="results/study_v2")
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("init")
    a.add_argument("--dataset-revision", required=True)
    a.add_argument(
        "--evidence-mode", choices=["none", "retrieved"], default="retrieved"
    )
    a.add_argument(
        "--multi-policy", choices=["exclude", "first", "error"], default="exclude"
    )
    a.add_argument("--max-calls", type=int, default=2000)
    a.add_argument("--human-size", type=int, default=30)
    a.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    a.add_argument("--n-boot", type=int, default=2000)
    a = sub.add_parser("prepare")
    a.add_argument("--validation", required=True)
    a.add_argument("--test", required=True)
    a = sub.add_parser("evidence")
    a.add_argument("--corpus")
    a.add_argument("--top-k", type=int, default=3)
    a = sub.add_parser("generate")
    a.add_argument("--model", choices=["A", "B"], required=True)
    a.add_argument("--split", choices=["validation", "test"], required=True)
    for name in ("metrics", "calibrate", "route", "prepare-judge", "export-human"):
        sub.add_parser(name)
    a = sub.add_parser("judge")
    a.add_argument("--model", choices=["J", "A", "B"], default="J")
    a = sub.add_parser("validate-human")
    a.add_argument("--annotations", required=True)
    a.add_argument("--attestation", required=True)
    a = sub.add_parser("report")
    a.add_argument("--out")
    a.add_argument("--require-human", action="store_true")
    a = p.parse_args()
    with lock(a.root):
        if a.stage == "init":
            result = init(
                a.root,
                a.dataset_revision,
                a.evidence_mode,
                a.multi_policy,
                a.max_calls,
                a.human_size,
                a.calibration,
                a.n_boot,
            )
        else:
            s = Study(a.root)
            if a.stage == "prepare":
                result = s.prepare(a.validation, a.test)
            elif a.stage == "evidence":
                result = s.evidence(a.corpus, a.top_k)
            elif a.stage == "generate":
                result = s.generate(a.split, a.model)
            elif a.stage == "metrics":
                result = s.metrics()
            elif a.stage == "calibrate":
                result = s.calibrate()
            elif a.stage == "route":
                result = s.route()
            elif a.stage == "prepare-judge":
                result = s.prepare_judge()
            elif a.stage == "judge":
                result = s.judge(a.model)
            elif a.stage == "export-human":
                result = {"annotation_file": s.export_human()}
            elif a.stage == "validate-human":
                result = s.validate_human(a.annotations, a.attestation)
            elif a.stage == "report":
                from empirical_report import build_report

                result = build_report(a.root, a.out, a.require_human)
    print(json.dumps(clean(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
