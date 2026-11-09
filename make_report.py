"""Persisted empirical outputs only. Synthetic --demo is intentionally unsupported."""

import argparse, json
from study_io import lock
from empirical_report import build_report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out")
    p.add_argument("--require-human", action="store_true")
    a = p.parse_args()
    with lock(a.root):
        print(json.dumps(build_report(a.root, a.out, a.require_human), indent=2))
