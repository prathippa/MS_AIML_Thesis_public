"""
reporting.py
------------
Turn computed results into thesis-ready tables and figures. This is the layer
between the analysis modules and the dissertation: it produces the RQ1 severity
table and chart, the RQ2 reliability diagram and risk-coverage curve, and the
RQ3 evidence-support summary.

Figures are written as PNG (matplotlib, Agg backend) so they can be dropped
straight into the write-up; tables are returned as Markdown.
"""

from __future__ import annotations
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (
    eval_records,
    severity_distribution,
    SEVERITY_TIERS,
    conventional_baselines,
)  # noqa: E402
from calibration import reliability_curve, ece  # noqa: E402


def _ensure(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ---- RQ1 ------------------------------------------------------------------ #
def severity_table_md(records_by_model: dict) -> str:
    models = list(records_by_model)
    dists = {
        m: severity_distribution(eval_records(records_by_model[m])) for m in models
    }
    out = [
        "| Severity tier | " + " | ".join(models) + " |",
        "| " + " | ".join(["---"] * (len(models) + 1)) + " |",
    ]
    for t in SEVERITY_TIERS:
        out.append(
            "| "
            + t.replace("_", " ")
            + " | "
            + " | ".join(f"{dists[m][t]:.0%}" for m in models)
            + " |"
        )
    exacts = {
        m: conventional_baselines(eval_records(records_by_model[m]))["exact_10"]
        for m in models
    }
    out.append(
        "| **exact-10 accuracy** | "
        + " | ".join(f"**{exacts[m]:.0%}**" for m in models)
        + " |"
    )
    return "\n".join(out)


def plot_severity_distribution(records_by_model: dict, path: str):
    _ensure(path)
    models = list(records_by_model)
    x = np.arange(len(SEVERITY_TIERS))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        dist = severity_distribution(eval_records(records_by_model[m]))
        ax.bar(x + i * w, [dist[t] for t in SEVERITY_TIERS], w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(
        [t.replace("_", " ") for t in SEVERITY_TIERS],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Proportion of cases")
    ax.set_title("Error-severity distribution by model (RQ1)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---- RQ2 ------------------------------------------------------------------ #
def plot_reliability(conf, correct, path, calibrated=None, nbins=10):
    _ensure(path)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")

    def pts(c):
        rc = reliability_curve(c, correct, nbins)["bins"]
        xs = [b["mean_confidence"] for b in rc if b["count"]]
        ys = [b["accuracy"] for b in rc if b["count"]]
        return xs, ys

    xr, yr = pts(conf)
    ax.plot(xr, yr, "o-", label=f"raw (ECE={ece(conf, correct):.3f})")
    if calibrated is not None:
        xc, yc = pts(calibrated)
        ax.plot(xc, yc, "s-", label=f"calibrated (ECE={ece(calibrated, correct):.3f})")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (RQ2)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_risk_coverage(curve_raw, curve_cal, path, primary_coverage=0.70):
    _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        [r["coverage"] for r in curve_raw],
        [r["selective_risk"] for r in curve_raw],
        "o-",
        label="raw (pooled)",
    )
    ax.plot(
        [r["coverage"] for r in curve_cal],
        [r["selective_risk"] for r in curve_cal],
        "s-",
        label="calibrated (pooled)",
    )
    ax.axvline(
        primary_coverage,
        color="gray",
        ls=":",
        lw=1,
        label=f"primary coverage {primary_coverage:.0%}",
    )
    ax.set_xlabel("Coverage (fraction streamlined)")
    ax.set_ylabel("Selective risk (error rate among streamlined)")
    ax.set_title("Risk\u2013coverage: raw vs calibrated routing (RQ2)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---- RQ3 ------------------------------------------------------------------ #
def evidence_support_table_md(judge_agg: dict, kappa: float | None = None) -> str:
    dist = judge_agg["label_distribution"]
    out = ["| Evidence-support label | Share |", "| --- | --- |"]
    for lbl in ("supported", "uncertain", "unsupported"):
        out.append(f"| {lbl} | {dist.get(lbl, 0):.0%} |")
    out.append(
        f"| mean consistency across runs | {judge_agg['mean_consistency']:.2f} |"
    )
    if kappa is not None:
        out.append(f"| Cohen's \u03ba vs human | {kappa:.2f} |")
    return "\n".join(out)
