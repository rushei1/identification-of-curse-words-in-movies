"""Regenerate docs/img/*.png from eval/results/*.json (and, if present, a local masked report).

  python scripts/make_figures.py

Colors: validated categorical slots (blue/orange for presets, aqua/violet for diarizers).
All numbers come from the result files; nothing is typed in here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.report import pooled  # noqa: E402

RES, OUT = ROOT / "eval" / "results", ROOT / "docs" / "img"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PRESET = {"paper": "#2a78d6", "improved": "#eb6834"}
DIAR = {"Original diarizer": "#1baf7a", "Diarizer 3.1": "#6250d6"}


def load():
    return {p.stem: json.loads(p.read_text()) for p in RES.glob("*.json")}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, series, colors, fmt="{:.0f}"):
    """Grouped bars; ``series`` = {name: [value per group]}; values in percent."""
    n, w = len(series), 0.34
    for i, (name, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * (w + 0.04) for g in range(len(groups))]
        ax.bar(xs, vals, w, color=colors[name], label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])


def detection(res):
    real = pooled(res)
    sets = [
        ("Real clips\npooled", real["presets"]),
        ("Real clip 1", res["sample-video"]["presets"]),
        ("Real clip 2", res["newyorker"]["presets"]),
        ("Synthetic", res["synthetic"]["presets"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True, facecolor=SURFACE)
    for ax, (metric, title) in zip(
        axes, [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    ):
        style(ax)
        series = {p: [100 * s[1][p][metric] for s in sets] for p in PRESET}
        bars(ax, [s[0] for s in sets], series, PRESET)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    axes[0].set_ylabel("%", color=MUTED, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.90), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK)  # fmt: skip
    fig.suptitle("Token-level profanity detection: paper vs improved matching",
                 x=0.01, y=0.985, ha="left", fontsize=11, color=INK, fontweight="bold")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "detection.png", dpi=160, facecolor=SURFACE)


def diarizers(res):
    clips = [("Real clip 1\n(trailer)", "sample-video", "netflix_d31"),
             ("Real clip 2\n(interview)", "newyorker", "newyorker_d31")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), sharey=True, facecolor=SURFACE)
    for ax, (match, title) in zip(
        axes, [("paper", "paper matching"), ("improved", "improved matching")]
    ):
        style(ax)
        series = {"Original diarizer": [], "Diarizer 3.1": []}
        for _, orig, d31 in clips:
            gold = res[orig]["n_gold_positive"]
            series["Original diarizer"].append(100 * res[orig]["presets"][match]["tp"] / gold)
            series["Diarizer 3.1"].append(100 * res[d31]["presets"][match]["tp"] / gold)
        bars(ax, [c[0] for c in clips], series, DIAR, fmt="{:.1f}")
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    axes[0].set_ylabel("% of gold profane words found", color=MUTED, fontsize=8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.89), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK)  # fmt: skip
    fig.suptitle("End-to-end recall by diarizer (words lost before detection count as misses)",
                 x=0.01, y=0.985, ha="left", fontsize=11, color=INK, fontweight="bold")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(OUT / "diarizers.png", dpi=160, facecolor=SURFACE)


def report_excerpt():
    src = ROOT / "output_improved" / "synthetic.md"
    if not src.exists():
        print("skip report_excerpt: run the synthetic clip with --preset improved first")
        return
    lines = src.read_text(encoding="utf-8").splitlines()
    keep = [
        ln
        for ln in lines
        if ln.startswith(("# ", "Preset", "| Group", "| MALE", "| FEMALE", "| UNKNOWN"))
    ]
    tl = [ln for ln in lines if ln.startswith("- ")][:5]
    text = "\n".join(
        [*keep, "", "## Timeline", *[t[:118] + ("..." if len(t) > 118 else "") for t in tl]]
    )
    fig = plt.figure(figsize=(10, 0.2 * (len(text.splitlines()) + 1.5)), facecolor="#f4f3ef")
    fig.text(
        0.015,
        0.97,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        color=INK,
        linespacing=1.5,
    )
    fig.savefig(OUT / "report_example.png", dpi=160, facecolor="#f4f3ef")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = load()
    detection(res)
    diarizers(res)
    report_excerpt()
    print(sorted(p.name for p in OUT.glob("*.png")))
