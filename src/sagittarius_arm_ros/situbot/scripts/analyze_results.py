#!/usr/bin/env python3
"""Analyze SituBench results and generate plots.

Usage:
    python analyze_results.py --results-dir results/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Analyze SituBench results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir for plots (defaults to results-dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir

    # Load results
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    all_results_path = os.path.join(args.results_dir, "all_results.json")

    if not os.path.exists(metrics_path):
        print(f"ERROR: {metrics_path} not found. Run run_situbench.py first.")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(all_results_path) as f:
        all_results = json.load(f)

    # Print summary
    overall = metrics["overall"]
    print(f"\nOverall Accuracy: {overall['correct']}/{overall['total']} "
          f"({overall['accuracy']:.1%})")

    print("\nPer-Level Breakdown:")
    for level, lm in metrics.get("by_level", {}).items():
        print(f"  {level:12s}: {lm['correct']}/{lm['total']} "
              f"({lm['accuracy']:.1%}, avg conf={lm['avg_confidence']:.2f})")

    conf = metrics.get("confidence", {})
    print(f"\nConfidence Analysis:")
    print(f"  Correct predictions: avg conf = {conf.get('correct_mean', 0):.2f}")
    print(f"  Wrong predictions:   avg conf = {conf.get('wrong_mean', 0):.2f}")

    # Generate plots
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Accuracy by level (bar chart)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Bar chart: accuracy per level
        levels = list(metrics.get("by_level", {}).keys())
        accuracies = [metrics["by_level"][l]["accuracy"] for l in levels]
        colors = {"functional": "#3498db", "cultural": "#e67e22", "emotional": "#e74c3c"}
        bar_colors = [colors.get(l, "#95a5a6") for l in levels]

        axes[0].bar(levels, accuracies, color=bar_colors, alpha=0.8)
        axes[0].set_ylim(0, 1.0)
        axes[0].set_ylabel("Roundtrip Accuracy")
        axes[0].set_title("Accuracy by Difficulty Level")
        axes[0].axhline(y=overall["accuracy"], color="gray", linestyle="--",
                        label=f"Overall: {overall['accuracy']:.1%}")
        axes[0].legend()
        for i, (l, a) in enumerate(zip(levels, accuracies)):
            axes[0].text(i, a + 0.02, f"{a:.0%}", ha="center", fontweight="bold")

        # 2. Confidence distribution
        correct_confs = [r.get("confidence", 0) for r in all_results if r.get("correct")]
        wrong_confs = [r.get("confidence", 0) for r in all_results if not r.get("correct")]

        if correct_confs:
            axes[1].hist(correct_confs, bins=10, alpha=0.6, color="green", label="Correct")
        if wrong_confs:
            axes[1].hist(wrong_confs, bins=10, alpha=0.6, color="red", label="Wrong")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Confidence Distribution")
        axes[1].legend()

        # 3. Per-scenario heatmap
        scenario_data = metrics.get("per_scenario", [])
        if scenario_data:
            names = [s.get("situation", "")[:30] for s in scenario_data]
            correct_vals = [1 if s.get("correct") else 0 for s in scenario_data]
            y_pos = range(len(names))
            bar_colors_scenario = ["green" if c else "red" for c in correct_vals]
            axes[2].barh(y_pos, [s.get("confidence", 0) for s in scenario_data],
                        color=bar_colors_scenario, alpha=0.7)
            axes[2].set_yticks(y_pos)
            axes[2].set_yticklabels(names, fontsize=6)
            axes[2].set_xlabel("Confidence")
            axes[2].set_title("Per-Scenario Results (green=correct)")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "situbench_analysis.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlots saved to {plot_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping plots")

    # Print failure analysis
    failures = [r for r in all_results if not r.get("correct")]
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURE ANALYSIS ({len(failures)} failures):")
        for r in failures:
            print(f"\n  Scenario: {r.get('scenario_id', '?')} ({r.get('level', '?')})")
            print(f"  Truth:     {r.get('ground_truth', '')[:60]}")
            print(f"  Predicted: {r.get('predicted', '')[:60]}")
            print(f"  Reasoning: {r.get('reasoning', '')[:100]}")


if __name__ == "__main__":
    main()
