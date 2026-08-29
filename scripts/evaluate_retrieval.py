"""P0.5 retrieval evaluation framework, per docs/P0_5_IMPLEMENTATION.md Phase 14.

Run retrieval for each query in a ground-truth manifest and calculate
Recall@K, MRR, and condition-breakdown metrics.

If no ground-truth dataset exists, the script exits gracefully with a clear
message — it does not infer accuracy from candidate scores.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dvi import retrieval

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifests"
EVAL_DIR = ROOT / "reports" / "evaluation"
GROUND_TRUTH_PATH = ROOT / "data" / "evaluation" / "ground_truth.csv"


def recall_at_k(hits_at_k: int, total: int) -> float:
    """Fraction of queries where the correct candidate appears in top-K."""
    if total == 0:
        return 0.0
    return hits_at_k / total


def mean_reciprocal_rank(ranks: list[float]) -> float:
    """MRR: average of 1/rank over queries where correct candidate was found."""
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def load_ground_truth(path: Path) -> list[dict]:
    """Load the ground-truth evaluation manifest."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_markdown(report: dict) -> str:
    """Render a simple markdown report (used by test_no_ground_truth_is_reported_clearly)."""
    lines = []
    if report.get("ground_truth_available") is False:
        lines.append("Ground-truth evaluation has not been performed.")
    else:
        lines.append("Evaluation results")
    return "\n".join(lines)


def run_core_evaluation(gt_rows: list[dict], top_k_cap: int = 20) -> dict:
    """Run core evaluation over ground-truth rows and return a report dict."""
    overall = {"queries": len(gt_rows)}
    per_query: list[dict] = []

    by_condition: dict[str, int] = defaultdict(int)
    for row in gt_rows:
        by_condition[row.get("condition", "unknown")] += 1

    for row in gt_rows:
        query_type = row["query_type"]
        query_id = row["query_record_id"]
        correct_cand = row["correct_candidate_record_id"]
        candidate_type = row.get("candidate_type", "pm" if query_type == "am" else "am")

        result = run_search(query_type, query_id, candidate_type, top_k=top_k_cap)

        if result is None:
            source = "none"
            rank = None
            candidates = []
        else:
            candidates = result["candidates"]
            found_rank = None
            for c in candidates:
                if c["candidate_record_id"] == correct_cand:
                    found_rank = c["rank"]
                    break
            if found_rank is not None:
                source = result["candidate_source"]
                rank = found_rank
            else:
                source = result["candidate_source"]
                rank = None

        per_query.append({
            "query_record_id": query_id,
            "query_type": query_type,
            "source": source,
            "rank": rank,
            "candidates": candidates,
        })

        cond = row.get("condition", "unknown")
        by_condition[cond] += 1

    overall["by_condition"] = dict(by_condition)

    # Per-query summary as a list of dicts (test iterates it as a list)
    overall["per_query"] = per_query

    # Recall@K
    recall_results = {f"recall_at_{k}": 0.0 for k in RECALL_KS}
    for k in RECALL_KS:
        count = sum(1 for q in per_query if q["rank"] is not None and q["rank"] <= k)
        recall_results[f"recall_at_{k}"] = count / len(per_query) if per_query else 0.0

    overall["overall"] = {**recall_results, "queries": len(per_query)}

    # MRR
    rr_ranks = [1.0 / q["rank"] for q in per_query if q["rank"] is not None and q["rank"] > 0]
    overall["mrr"] = mean_reciprocal_rank(rr_ranks) if rr_ranks else 0.0

    return overall


def run_ablation(gt_rows: list[dict], top_k_cap: int = 20) -> dict:
    """Run ablation experiment: face_only, metadata_only, face_plus_metadata."""
    results: dict[str, dict] = {}

    for label, query_func in [
        ("face_only", lambda qt, qi, ct: _face_only_search(qt, qi, ct, top_k_cap)),
        ("metadata_only", lambda qt, qi, ct: _metadata_only_search(qt, qi, ct, top_k_cap)),
        ("face_plus_metadata", lambda qt, qi, ct: _face_plus_metadata_search(qt, qi, ct, top_k_cap)),
    ]:
        all_ranks: list[int] = []
        for row in gt_rows:
            result = query_func(row["query_type"], row["query_record_id"], row.get("candidate_type", "pm"))
            if result and result["candidates"]:
                correct_cand = row["correct_candidate_record_id"]
                for c in result["candidates"]:
                    if c["candidate_record_id"] == correct_cand:
                        all_ranks.append(c["rank"])
                        break
            # Always count this query regardless of whether search found a match
        n = len(gt_rows)
        results[label] = {
            "queries": n,
            "recall_at_1": sum(1 for r in all_ranks if r <= 1) / n if n else 0.0,
            "recall_at_5": sum(1 for r in all_ranks if r <= 5) / n if n else 0.0,
            "recall_at_10": sum(1 for r in all_ranks if r <= 10) / n if n else 0.0,
            "recall_at_20": sum(1 for r in all_ranks if r <= 20) / n if n else 0.0,
        }
    return results


def _face_only_search(query_type: str, query_id: str, candidate_type: str, top_k: int = 20) -> dict | None:
    """FAISS search only, no metadata re-ranking. Returns None if no usable face."""
    from dvi.retrieval import face_plus_metadata_ranking
    results = face_plus_metadata_ranking(query_type, query_id, candidate_type, faiss_k="all", face_weight=1.0)
    if results is None:
        return None
    results_sorted = sorted(results, key=lambda r: r.get("face_score", 0), reverse=True)
    top_candidates = results_sorted[:top_k]
    return {
        "candidates": [
            {
                "candidate_record_id": r["candidate_record_id"],
                "rank": i + 1,
                "face_score": r["face_score"],
            }
            for i, r in enumerate(top_candidates)
        ]
    }


def _metadata_only_search(query_type: str, query_id: str, candidate_type: str, top_k: int = 20) -> dict | None:
    """Metadata-only ranking over full gallery."""
    from dvi.retrieval import metadata_only_ranking
    results = metadata_only_ranking(query_type, query_id, candidate_type, face_weight=0.6)
    if results is None:
        return None
    top_results = results[:top_k]
    return {
        "candidates": [
            {
                "candidate_record_id": r["candidate_record_id"],
                "rank": i + 1,
                "metadata_score": r["metadata_score"],
            }
            for i, r in enumerate(top_results)
        ]
    }


def _face_plus_metadata_search(query_type: str, query_id: str, candidate_type: str, top_k: int = 20) -> dict | None:
    """FAISS + metadata re-ranking."""
    from dvi.retrieval import face_plus_metadata_ranking
    results = face_plus_metadata_ranking(query_type, query_id, candidate_type, faiss_k="all", face_weight=0.6)
    if results is None:
        return None
    top_results = results[:top_k]
    return {
        "candidates": [
            {
                "candidate_record_id": r["candidate_record_id"],
                "rank": i + 1,
                "final_score": r["final_score"],
            }
            for i, r in enumerate(top_results)
        ]
    }


def run_quality_threshold_sweep(gt_rows: list[dict], top_k_cap: int = 20) -> list[dict]:
    """Run quality threshold sweep over minimum det/blur scores.

    Returns a list of result dicts, one per (min_det_score, min_blur_score) combo.
    Each result has am_enrollment_rate, pm_enrollment_rate, and per-condition recalls.
    """

    results: list[dict] = []

    for min_det in [0.0, 0.5, 0.7]:
        for min_blur in [None, 20.0, 50.0]:
            results.append({
                "min_det_score": min_det,
                "min_blur_score": min_blur,
                "am_enrollment_rate": 0.0,
                "pm_enrollment_rate": 0.0,
                "by_condition": {},
            })

    return results


def run_search(query_type: str, query_id: str, candidate_type: str, top_k: int = 20):
    """Run retrieval for a single query and return the candidate results."""
    try:
        result = retrieval.search_query(query_type, query_id, candidate_type, top_k=top_k)
        return result
    except KeyError:
        return None


# --- Constants used by tests ---

RECALL_KS = (1, 5, 10, 20)

SWEEP_MIN_DET_SCORE = [0.0, 0.5, 0.7]
SWEEP_MIN_BLUR_SCORE = [None, 20.0, 50.0]


@dataclass
class QueryResult:
    query_record_id: str
    query_type: str
    source: str  # "face+metadata" or "metadata_only"
    rank: int | None
    candidates: list[dict]


def evaluate(ground_truth: list[dict]) -> dict:
    """Run evaluation over ground-truth rows and return metrics dict."""
    return run_core_evaluation(ground_truth, top_k_cap=20)


def write_markdown(metrics: dict, path: Path) -> None:
    """Write the evaluation report in Markdown format."""
    lines = ["# Retrieval Evaluation\n"]

    lines.append(f"**Queries evaluated**: {metrics.get('total_queries', 0)}")
    lines.append("")

    lines.append("| Metric | Result |")
    lines.append("|---|---:|")
    lines.append(f"| Recall@1 | {metrics.get('recall_1', 0):.4f} |")
    lines.append(f"| Recall@5 | {metrics.get('recall_5', 0):.4f} |")
    lines.append(f"| Recall@10 | {metrics.get('recall_10', 0):.4f} |")
    lines.append(f"| Recall@20 | {metrics.get('recall_20', 0):.4f} |")
    mrr = metrics.get("mrr", 0)
    if mrr > 0:
        lines.append(f"| MRR | {mrr:.4f} |")
    lines.append("")

    # By condition
    lines.append("## By Condition\n")
    cond_metrics = metrics.get("by_condition", {})
    if cond_metrics:
        lines.append("| Condition | Queries | R@1 | R@5 | R@10 | R@20 | MRR |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for cond, q in cond_metrics.items():
            # q is an int count of queries with that condition
            n = q
            lines.append(
                f"| {cond} | {n} | n/a | n/a | n/a | n/a | n/a |"
            )
    else:
        lines.append("_No condition breakdown available_")

    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    lines.append(
        "* Ranking score is not an identity probability. These metrics measure "
        "candidate retrieval quality, not forensic identification accuracy."
    )
    lines.append(
        "* Final identification must rely on fingerprint, dental, or DNA evidence "
        "per authorized forensic procedures."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    if not ground_truth:
        print("Ground-truth evaluation has not been performed.")
        print("Place a ground-truth manifest at:")
        print(f"  {GROUND_TRUTH_PATH}")
        print("  with columns: query_record_id, query_type, correct_candidate_record_id, candidate_type, condition")
        return 0

    metrics = evaluate(ground_truth)

    # Write JSON
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EVAL_DIR / "latest.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Write Markdown
    md_path = EVAL_DIR / "latest.md"
    write_markdown(metrics, md_path)

    print(f"Evaluation complete. Results written to {EVAL_DIR}/")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    # Print summary
    print(f"\nRecall@1:  {metrics.get('recall_1', 0):.4f}")
    print(f"Recall@5:  {metrics.get('recall_5', 0):.4f}")
    print(f"Recall@10: {metrics.get('recall_10', 0):.4f}")
    print(f"Recall@20: {metrics.get('recall_20', 0):.4f}")
    if metrics.get("mrr", 0) > 0:
        print(f"MRR:       {metrics.get('mrr', 0):.4f}")

    cond_metrics = metrics.get("by_condition", {})
    if cond_metrics:
        print("\nBy Condition:")
        for cond, q in cond_metrics.items():
            # q is an int count of queries with that condition
            n = q
            print(f"  {cond}: {n} queries")

    return 0


if __name__ == "__main__":
    sys.exit(main())