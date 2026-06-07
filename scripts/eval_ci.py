#!/usr/bin/env python3
"""
CI/CD Evaluation Gate
═════════════════════
Runs RAGAS evaluation over the golden dataset and fails (exit code 1)
if any metric falls below its configured threshold.

Usage
-----
    # Run full evaluation (all 50 pairs)
    uv run python scripts/eval_ci.py

    # Quick smoke-test with a subset (faster for CI)
    uv run python scripts/eval_ci.py --sample 10

    # Test a specific RAG pattern
    uv run python scripts/eval_ci.py --pattern hybrid --sample 20

    # Write results to JSON for artifact upload
    uv run python scripts/eval_ci.py --output eval_results.json

Exit codes
----------
    0  All metrics pass thresholds
    1  One or more metrics fail thresholds (CI build should fail)
    2  Evaluation infrastructure error (Qdrant/LLM not available)

Environment
-----------
Requires a running Qdrant instance with documents already ingested.
Set LLM credentials via .env or environment variables.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docustra.core import get_settings
from docustra.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("eval_ci")

GOLDEN_DATASET = Path(__file__).parent.parent / "data" / "eval" / "golden_dataset.json"


def load_golden_dataset(sample: int | None = None, domain: str | None = None) -> list[dict]:
    """Load QA pairs from the golden dataset, optionally sampling a subset."""
    with GOLDEN_DATASET.open() as f:
        data = json.load(f)

    pairs = data["pairs"]

    if domain:
        pairs = [p for p in pairs if p.get("domain") == domain]
        logger.info("Filtered by domain", domain=domain, count=len(pairs))

    if sample and sample < len(pairs):
        pairs = random.sample(pairs, sample)
        logger.info("Sampled pairs", sample=sample)

    return pairs


def run_queries(pairs: list[dict], pattern: str) -> tuple[list[str], list[str], list[list[str]]]:
    """
    Execute each QA pair against the RAG system.

    Returns
    -------
    questions, answers, contexts : parallel lists for RAGAS
    """
    from docustra.retrieval import get_strategy

    strategy = get_strategy(pattern)
    questions, answers, contexts = [], [], []

    for i, pair in enumerate(pairs, start=1):
        q = pair["question"]
        logger.info("Running eval query", idx=i, total=len(pairs), question=q[:60])
        try:
            response = strategy.query(q)
            questions.append(q)
            answers.append(response.answer)
            contexts.append([s.get("content", s.get("passage_preview", "")) for s in response.sources])
        except Exception as e:
            logger.error("Query failed", question=q[:60], error=str(e))
            questions.append(q)
            answers.append("ERROR: " + str(e))
            contexts.append([""])

    return questions, answers, contexts


def evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Run RAGAS evaluation and return metric scores."""
    from docustra.evaluation.metrics import evaluate_rag

    result = evaluate_rag(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )
    return result.as_dict()


def check_thresholds(scores: dict[str, float], settings) -> tuple[bool, list[str]]:
    """
    Compare scores against configured thresholds.

    Returns
    -------
    passed : bool
    failures : list of human-readable failure messages
    """
    thresholds = {
        "faithfulness": settings.eval_faithfulness_threshold,
        "answer_relevancy": settings.eval_answer_relevancy_threshold,
        "context_precision": settings.eval_context_precision_threshold,
    }

    failures = []
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0.0)
        if score < threshold:
            failures.append(
                f"  ✗ {metric}: {score:.4f} < threshold {threshold:.2f}"
            )
        else:
            logger.info("Metric passed", metric=metric, score=round(score, 4), threshold=threshold)

    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Docustra CI evaluation gate")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of QA pairs to sample (default: all 50)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="hybrid",
        help="RAG pattern to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["apple_10k", "vector_databases"],
        help="Restrict to a specific document domain",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write full results to JSON file",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override prompt version (e.g. v1, v2)",
    )
    args = parser.parse_args()

    settings = get_settings()

    if args.prompt_version:
        import os
        os.environ["PROMPT_VERSION"] = args.prompt_version
        from docustra.core.prompts import invalidate_cache
        invalidate_cache()

    logger.info(
        "Starting CI evaluation",
        pattern=args.pattern,
        sample=args.sample or "all",
        domain=args.domain or "all",
        prompt_version=settings.prompt_version,
        faithfulness_threshold=settings.eval_faithfulness_threshold,
        relevancy_threshold=settings.eval_answer_relevancy_threshold,
        precision_threshold=settings.eval_context_precision_threshold,
    )

    # ── Load dataset ──────────────────────────────────────────────────────
    try:
        pairs = load_golden_dataset(sample=args.sample, domain=args.domain)
    except FileNotFoundError:
        logger.error("Golden dataset not found", path=str(GOLDEN_DATASET))
        return 2

    if not pairs:
        logger.error("No evaluation pairs after filtering")
        return 2

    ground_truths = [p["ground_truth"] for p in pairs]

    # ── Run queries ───────────────────────────────────────────────────────
    try:
        questions, answers, contexts = run_queries(pairs, args.pattern)
    except Exception as e:
        logger.error("Query execution failed", error=str(e))
        print(f"\n[ERROR] Cannot connect to RAG system: {e}", file=sys.stderr)
        print("Ensure Qdrant is running and documents are ingested.", file=sys.stderr)
        return 2

    # ── Evaluate ──────────────────────────────────────────────────────────
    try:
        scores = evaluate(questions, answers, contexts, ground_truths)
    except Exception as e:
        logger.error("RAGAS evaluation failed", error=str(e))
        print(f"\n[ERROR] RAGAS evaluation failed: {e}", file=sys.stderr)
        return 2

    # ── Check thresholds ──────────────────────────────────────────────────
    passed, failures = check_thresholds(scores, settings)

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DOCUSTRA RAG EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Pattern:        {args.pattern}")
    print(f"  Prompt version: {settings.prompt_version}")
    print(f"  Pairs evaluated:{len(pairs)}")
    print()
    print("  Scores:")
    for metric, score in scores.items():
        threshold = getattr(settings, f"eval_{metric}_threshold", None)
        status = "✓" if threshold is None or score >= threshold else "✗"
        threshold_str = f"(threshold: {threshold:.2f})" if threshold else ""
        print(f"    {status} {metric:<26} {score:.4f}  {threshold_str}")

    if not passed:
        print()
        print("  FAILED — thresholds not met:")
        for f in failures:
            print(f)
        print()
        print("  ⚠ Build should be blocked until scores improve.")
        print("    See docs/eval-improvement.md for guidance.")
        print("=" * 60)

    if passed:
        print()
        print("  ✓ All thresholds passed — build gates cleared.")
        print("=" * 60)

    # ── Write output artifact ──────────────────────────────────────────────
    if args.output:
        output = {
            "pattern": args.pattern,
            "prompt_version": settings.prompt_version,
            "pairs_evaluated": len(pairs),
            "scores": scores,
            "thresholds": {
                "faithfulness": settings.eval_faithfulness_threshold,
                "answer_relevancy": settings.eval_answer_relevancy_threshold,
                "context_precision": settings.eval_context_precision_threshold,
            },
            "passed": passed,
            "failures": failures,
            "sample_responses": [
                {"question": q, "answer": a[:300]}
                for q, a in zip(questions[:5], answers[:5], strict=True)
            ],
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\n  Results written to: {args.output}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
