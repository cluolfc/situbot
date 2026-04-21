#!/usr/bin/env python3
"""Metrics computation for SituBot evaluation."""

from typing import List, Dict
from collections import defaultdict


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics from evaluation results.

    Args:
        results: List of evaluation result dicts from RoundtripEvaluator.

    Returns:
        Dict with overall and per-level metrics.
    """
    if not results:
        return {"error": "no results"}

    # Overall accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)

    # Per difficulty level
    by_level = defaultdict(list)
    for r in results:
        # Infer level from scenario id if available
        gt = r.get("ground_truth", "")
        level = _infer_level(gt, r)
        by_level[level].append(r)

    level_metrics = {}
    for level, level_results in by_level.items():
        level_correct = sum(1 for r in level_results if r.get("correct", False))
        level_total = len(level_results)
        avg_conf = sum(r.get("confidence", 0) for r in level_results) / level_total
        level_metrics[level] = {
            "accuracy": level_correct / level_total if level_total > 0 else 0.0,
            "correct": level_correct,
            "total": level_total,
            "avg_confidence": avg_conf,
        }

    # Confidence analysis
    correct_confs = [r.get("confidence", 0) for r in results if r.get("correct")]
    wrong_confs = [r.get("confidence", 0) for r in results if not r.get("correct")]

    return {
        "overall": {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
        },
        "by_level": level_metrics,
        "confidence": {
            "correct_mean": sum(correct_confs) / len(correct_confs) if correct_confs else 0,
            "wrong_mean": sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0,
        },
        "per_scenario": [
            {
                "situation": r.get("ground_truth", ""),
                "correct": r.get("correct", False),
                "predicted": r.get("predicted", ""),
                "confidence": r.get("confidence", 0),
            }
            for r in results
        ],
    }


def _infer_level(situation: str, result: Dict) -> str:
    """Infer difficulty level from scenario metadata or situation text."""
    # If scenario_id is available (e.g., "F01", "C05", "E10")
    scenario_id = result.get("scenario_id", "")
    if scenario_id:
        prefix = scenario_id[0]
        return {"F": "functional", "C": "cultural", "E": "emotional"}.get(prefix, "unknown")
    return result.get("level", "unknown")
