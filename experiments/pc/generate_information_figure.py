"""Generate the deterministic information--completion frontier figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from information_completion_frontier import run
from matplotlib.text import Text

matplotlib.use("Agg", force=True)


def generate(config_path: Path, output_path: Path, *, input_path: Path | None = None) -> None:
    payload = (
        json.loads(input_path.read_text(encoding="utf-8"))
        if input_path is not None
        else run(config_path)
    )
    structured = payload["structured_frontier"]
    higher_order = payload["higher_order_gap"]
    static_age = payload["static_age_gap"]
    if not all(isinstance(result, list) for result in (structured, higher_order, static_age)):
        raise TypeError("structured results must be lists")

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#4D4D4D")
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), constrained_layout=True)

    deadlines = [row["delay"] for row in structured]
    symbols = [row["minimum_symbols"] for row in structured]
    axes[0].step(deadlines, symbols, where="post", color=colors[0], linewidth=1.7)
    axes[0].scatter(deadlines, symbols, color=colors[0], s=16, zorder=3)
    axes[0].set_xlabel("completion deadline $d$")
    axes[0].set_ylabel("minimum symbols $q^*$")
    axes[0].set_xticks(deadlines)
    axes[0].set_yticks(symbols)
    axes[0].set_ylim(0.8, max(symbols) + 0.25)
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, va="top", weight="bold")

    sizes = [row["uncertain_states"] for row in higher_order]
    gaps = [row["minimum_symbols"] for row in higher_order]
    pairwise = [row["pairwise_graph_estimate"] for row in higher_order]
    axes[1].plot(
        sizes,
        gaps,
        color=colors[1],
        linewidth=1.7,
        marker="o",
        markersize=3.5,
        label="hypergraph",
    )
    axes[1].plot(
        sizes,
        pairwise,
        color=colors[4],
        linewidth=1.2,
        linestyle="--",
        label="pairwise",
    )
    axes[1].set_xlabel("uncertain states $n$")
    axes[1].set_ylabel("immediate $q^*$")
    axes[1].set_xticks((3, 4, 5, 6, 7, 8))
    axes[1].set_yticks((1, 2, 3, 4))
    axes[1].set_xlim(2.8, 8.2)
    axes[1].set_ylim(0.8, 4.2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.01, 0.82), fontsize=6.3, frameon=True)
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, va="top", weight="bold")

    static_sizes = [row["uncertain_states"] for row in static_age]
    static_symbols = [row["static_symbols"] for row in static_age]
    known_age_symbols = [row["known_age_symbols"] for row in static_age]
    axes[2].plot(
        static_sizes,
        static_symbols,
        color=colors[3],
        linewidth=1.7,
        marker="o",
        markersize=3.5,
        label="static map",
    )
    axes[2].plot(
        static_sizes,
        known_age_symbols,
        color=colors[4],
        linewidth=1.2,
        linestyle="--",
        label="age-dependent",
    )
    axes[2].set_xlabel("uncertain states $n$")
    axes[2].set_ylabel("required symbols")
    axes[2].set_xticks(static_sizes)
    axes[2].set_yticks(static_sizes)
    axes[2].set_xlim(min(static_sizes) - 0.2, max(static_sizes) + 0.2)
    axes[2].set_ylim(1.8, max(static_symbols) + 0.2)
    axes[2].legend(loc="upper left", bbox_to_anchor=(0.01, 0.82), fontsize=6.3, frameon=True)
    axes[2].text(0.03, 0.95, "(c)", transform=axes[2].transAxes, va="top", weight="bold")

    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
        axis.grid(True, linewidth=0.35, alpha=0.45)

    # Vector outlines avoid Type 3 and CID font resources in embedded figures.
    # This preserves scalable glyph shapes without adding a TeX dependency.
    for label in figure.findobj(match=Text):
        label.set_path_effects([path_effects.Normal()])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Exact information--completion frontiers",
            "Creator": "safe-compute-envelopes reproducibility package",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pc_information_completion.toml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/information_completion.pdf"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="reuse a previously generated frontier JSON instead of rerunning the experiment",
    )
    args = parser.parse_args()
    generate(args.config, args.output, input_path=args.input)
    print(f"INFORMATION_FIGURE_PASS output={args.output}")


if __name__ == "__main__":
    main()
