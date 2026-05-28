from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl_metrics(metrics_path: Path) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            metrics.append(json.loads(line))
    return metrics


def write_training_report(
    metrics: list[dict[str, float]],
    report_dir: Path,
    run_config: dict[str, Any],
    title: str,
) -> None:
    assert metrics
    report_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_group(
        metrics=metrics,
        keys=["loss", "policy_loss", "value_loss", "ownership_loss"],
        title="Training Loss",
        ylabel="loss",
        output_path=report_dir / "loss.png",
    )
    _plot_metric_group(
        metrics=metrics,
        keys=["validation_candidate_win_rate"],
        title="Validation Win Rate",
        ylabel="win rate",
        output_path=report_dir / "validation_win_rate.png",
        y_min=0.0,
        y_max=1.0,
    )
    _plot_metric_group(
        metrics=metrics,
        keys=[
            "validation_candidate_black_win_rate",
            "validation_candidate_white_win_rate",
        ],
        title="Validation Win Rate By Color",
        ylabel="win rate",
        output_path=report_dir / "validation_color_win_rate.png",
        y_min=0.0,
        y_max=1.0,
    )
    _plot_metric_group(
        metrics=metrics,
        keys=["validation_candidate_avg_score_margin"],
        title="Validation Score Margin",
        ylabel="points",
        output_path=report_dir / "validation_score_margin.png",
    )
    _plot_metric_group(
        metrics=metrics,
        keys=["validation_avg_moves", "validation_terminal_rate"],
        title="Validation Game Dynamics",
        ylabel="value",
        output_path=report_dir / "validation_game_dynamics.png",
    )
    _plot_metric_group(
        metrics=metrics,
        keys=["self_play_samples", "replay_size"],
        title="Self-Play Data",
        ylabel="samples",
        output_path=report_dir / "self_play.png",
    )
    _write_markdown_report(
        metrics=metrics,
        report_dir=report_dir,
        run_config=run_config,
        title=title,
    )


def _plot_metric_group(
    metrics: list[dict[str, float]],
    keys: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    available_keys = [
        key
        for key in keys
        if any(key in row for row in metrics)
    ]
    if not available_keys:
        _plot_empty(title, output_path)
        return

    iterations = [row["iteration"] for row in metrics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key in available_keys:
        values = [
            row[key] if key in row else float("nan")
            for row in metrics
        ]
        ax.plot(iterations, values, marker="o", linewidth=2, label=_label_for_key(key))

    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_empty(title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        "No metric recorded",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_markdown_report(
    metrics: list[dict[str, float]],
    report_dir: Path,
    run_config: dict[str, Any],
    title: str,
) -> None:
    final_metrics = metrics[-1]
    best_metrics = _best_validation_metrics(metrics)
    lines = [
        f"# {title}",
        "",
        "## Final Metrics",
        "",
        _metrics_table(final_metrics),
        "",
        "## Best Validation",
        "",
        _metrics_table(best_metrics),
        "",
        "## Plots",
        "",
        "![Training loss](loss.png)",
        "",
        "![Validation win rate](validation_win_rate.png)",
        "",
        "![Validation win rate by color](validation_color_win_rate.png)",
        "",
        "![Validation score margin](validation_score_margin.png)",
        "",
        "![Validation game dynamics](validation_game_dynamics.png)",
        "",
        "![Self-play data](self_play.png)",
        "",
        "## Run Config",
        "",
        "```json",
        json.dumps(run_config, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _metrics_table(metrics: dict[str, float]) -> str:
    lines = [
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | {metrics[key]} |")
    return "\n".join(lines)


def _best_validation_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    validation_rows = [
        row
        for row in metrics
        if "validation_candidate_win_rate" in row
    ]
    if not validation_rows:
        return metrics[-1]
    return max(
        validation_rows,
        key=lambda row: row["validation_candidate_win_rate"],
    )


def _label_for_key(key: str) -> str:
    return key.replace("_", " ")
