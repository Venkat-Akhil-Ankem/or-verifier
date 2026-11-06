"""
report.py
---------
Turn summary metrics into a Markdown table and a grouped bar chart
(accuracy by problem family, one bar per feedback mode).
"""

from __future__ import annotations

from pathlib import Path


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def to_markdown(summary: dict, title: str = "OR-Verifier results") -> str:
    modes = list(summary)
    lines = [f"# {title}", "", "| Mode | Tasks | pass@1 | Final accuracy | Repair gain | Visible-only acc | Mean rounds |",
             "|---|---|---|---|---|---|---|"]
    for m in modes:
        s = summary[m]
        lines.append(f"| {m} | {s['n_tasks']} | {_pct(s['pass@1'])} | {_pct(s['final_acc'])} | "
                     f"{100 * s['repair_gain']:+.0f} pts | {_pct(s['visible_acc'])} | {s['mean_rounds']:.2f} |")

    fams = sorted({f for m in modes for f in summary[m]["by_family"]})
    lines += ["", "## Final accuracy by problem family", "",
              "| Family | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for f in fams:
        lines.append(f"| {f} | " + " | ".join(_pct(summary[m]["by_family"].get(f, float('nan'))) for m in modes) + " |")

    lines += ["", "## Final accuracy by size", "", "| Size | " + " | ".join(modes) + " |",
              "|---|" + "---|" * len(modes)]
    for sc in ("small", "large"):
        lines.append(f"| {sc} | " + " | ".join(_pct(summary[m]["by_scale"].get(sc, float('nan'))) for m in modes) + " |")

    lines += ["", "## Failure categories of the final attempt", ""]
    cats = sorted({c for m in modes for c in summary[m]["failures"]})
    lines += ["| Category | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(str(summary[m]["failures"].get(c, 0)) for m in modes) + " |")
    return "\n".join(lines) + "\n"


def plot_by_family(summary: dict, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(summary)
    fams = sorted({f for m in modes for f in summary[m]["by_family"]})
    colors = ["#9aa5b1", "#2f6f9f", "#d9822b"]          # muted grey / blue / orange
    width = 0.8 / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, m in enumerate(modes):
        xs = [i + (k - (len(modes) - 1) / 2) * width for i in range(len(fams))]
        ax.bar(xs, [100 * summary[m]["by_family"].get(f, 0) for f in fams], width,
               label=m, color=colors[k % len(colors)])
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=8)
    ax.set_ylabel("Tasks solved optimally on all instances (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Text-to-optimisation accuracy by problem family")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
