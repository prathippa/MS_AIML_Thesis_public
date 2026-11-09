"""Tests for reporting.py: tables build and figures are written."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import reporting as RP

GOLD = "0101210010"
A = [{"gold": GOLD, "pred": "0101210010"}, {"gold": GOLD, "pred": "0101219910"},
     {"gold": GOLD, "pred": "9999210010"}, {"gold": GOLD, "pred": "0101210010"}]
B = [{"gold": GOLD, "pred": "9999999999"}, {"gold": GOLD, "pred": "0101210010"},
     {"gold": GOLD, "pred": "0101210010"}, {"gold": GOLD, "pred": "9999210010"}]


def test_severity_table_md_has_all_tiers():
    md = RP.severity_table_md({"Model A": A, "Model B": B})
    for t in ("correct 10", "wrong chapter", "exact-10 accuracy"):
        assert t in md
    assert "Model A" in md and "Model B" in md


def test_figures_written_non_empty():
    conf = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3]
    correct = [1, 1, 0, 1, 0, 0, 1, 0]
    with tempfile.TemporaryDirectory() as d:
        p1 = RP.plot_severity_distribution({"A": A, "B": B}, os.path.join(d, "sev.png"))
        p2 = RP.plot_reliability(conf, correct, os.path.join(d, "rel.png"), calibrated=conf)
        from routing import risk_coverage_curve
        cov = [0.5, 0.7, 0.9]
        p3 = RP.plot_risk_coverage(risk_coverage_curve(conf, correct, cov),
                                   risk_coverage_curve(conf, correct, cov), os.path.join(d, "rc.png"))
        for p in (p1, p2, p3):
            assert os.path.exists(p) and os.path.getsize(p) > 1000


def test_evidence_support_table():
    agg = {"label_distribution": {"supported": 0.6, "uncertain": 0.2, "unsupported": 0.2},
           "mean_consistency": 0.83}
    md = RP.evidence_support_table_md(agg, kappa=0.71)
    assert "supported" in md and "0.71" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
