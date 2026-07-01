"""
tests/evaluation/ragas_evaluator.py

RAGAs evaluation runner for all pipeline stages.

This produces your benchmark table — the ablation study numbers
that go on your resume and in your README.

How it works:
1. Loads test questions from tests/evaluation/test_set_100q.json
2. Runs each question through the specified retriever
3. Collects: question, answer, retrieved contexts, ground truth
4. Sends to RAGAs for scoring: faithfulness, answer_relevancy,
   context_precision, context_recall
5. Saves results to tests/evaluation/results/

Usage:
    python tests/evaluation/ragas_evaluator.py --stage 1
    python tests/evaluation/ragas_evaluator.py --stage 2
    python tests/evaluation/ragas_evaluator.py --compare  # shows all stages
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Literal
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("tests/evaluation/results")
TEST_SET_PATH = Path("tests/evaluation/test_set_100q.json")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_set() -> list[dict]:
    """Load the 100-question test set."""
    if not TEST_SET_PATH.exists():
        print(f"Test set not found at {TEST_SET_PATH}")
        print("Creating a sample test set with 5 questions to get started...")
        sample = create_sample_test_set()
        TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_SET_PATH, "w") as f:
            json.dump(sample, f, indent=2)
        return sample

    with open(TEST_SET_PATH) as f:
        return json.load(f)


def create_sample_test_set() -> list[dict]:
    """
    Sample test set — replace with your 100 hand-crafted questions.
    Each question needs a ground_truth answer for RAGAs context_recall evaluation.
    """
    return [
        {
            "id": "q001",
            "category": "factual",
            "question": "What are the main risk factors mentioned in Zomato's DRHP?",
            "ground_truth": "Zomato's DRHP mentions several risk factors including intense competition from Swiggy and other food delivery platforms, regulatory risks, dependence on restaurant partners, and unit economics challenges including high customer acquisition costs.",
            "company_filter": ["Zomato"],
            "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q002",
            "category": "factual",
            "question": "What is Zomato's primary business model as described in their DRHP?",
            "ground_truth": "Zomato operates a food delivery marketplace connecting consumers with restaurant partners through its app and website, earning commission from restaurants on each order and delivery fees from consumers.",
            "company_filter": ["Zomato"],
            "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q003",
            "category": "factual",
            "question": "What were the objects of Paytm's IPO as mentioned in their DRHP?",
            "ground_truth": "Paytm's IPO proceeds were planned to be used for growing and strengthening the Paytm ecosystem, expanding merchant and consumer base, and general corporate purposes.",
            "company_filter": ["Paytm"],
            "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q004",
            "category": "factual",
            "question": "What are the main legal proceedings mentioned in Zomato's DRHP?",
            "ground_truth": "Zomato's DRHP discloses various pending litigation matters including tax disputes and regulatory proceedings across different jurisdictions where the company operates.",
            "company_filter": ["Zomato"],
            "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q005",
            "category": "factual",
            "question": "Who are the major shareholders of Zomato as disclosed in their DRHP?",
            "ground_truth": "Major shareholders of Zomato include Info Edge (India) Limited, Ant Financial (through subsidiaries), and various institutional investors, with the promoters holding a significant stake before the IPO.",
            "company_filter": ["Zomato"],
            "doc_type_filter": ["DRHP"]
        },
    ]


def run_evaluation(
    stage: Literal[1, 2],
    test_questions: list[dict],
    max_questions: int = None
) -> dict:
    """
    Run RAGAs evaluation for a given stage.

    Args:
        stage:          1 (baseline) or 2 (hybrid + reranker)
        test_questions: Questions from test set
        max_questions:  Limit questions (useful for quick tests)

    Returns:
        RAGAs evaluation scores dict
    """
    if max_questions:
        test_questions = test_questions[:max_questions]

    print(f"\nRunning Stage {stage} evaluation on {len(test_questions)} questions...")

    # Initialize retriever
    if stage == 1:
        retriever = BaseRetriever()
        query_fn = lambda q, meta: retriever.query(q, metadata_filter=meta)
    else:
        retriever = HybridRetriever()
        query_fn = lambda q, companies, years, doc_types: retriever.query(
            q, companies=companies, years=years, doc_types=doc_types
        )

    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    latencies = []

    for i, item in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] {item['question'][:70]}...")

        try:
            start = time.time()

            if stage == 1:
                from src.retrieval.metadata_filter import build_filter
                meta_filter = build_filter(
                    companies=item.get("company_filter"),
                    doc_types=item.get("doc_type_filter")
                )
                result = retriever.query(item["question"], metadata_filter=meta_filter)
            else:
                result = retriever.query(
                    item["question"],
                    companies=item.get("company_filter"),
                    doc_types=item.get("doc_type_filter")
                )

            latency = round((time.time() - start) * 1000)
            latencies.append(latency)

            questions_list.append(item["question"])
            answers_list.append(result["answer"])
            contexts_list.append([c.page_content for c in result["chunks"]])
            ground_truths_list.append(item["ground_truth"])

            # Rate limiting between questions — avoid hitting Groq limits
            time.sleep(2)

        except Exception as e:
            logger.error(f"Question {item['id']} failed: {e}")
            questions_list.append(item["question"])
            answers_list.append("ERROR")
            contexts_list.append([""])
            ground_truths_list.append(item["ground_truth"])
            latencies.append(0)

    # Build RAGAs dataset
    eval_dataset = Dataset.from_dict({
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    })

    print(f"\nRunning RAGAs scoring...")
    scores = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    result_summary = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(test_questions),
        "faithfulness": round(float(scores["faithfulness"]), 4),
        "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
        "context_precision": round(float(scores["context_precision"]), 4),
        "context_recall": round(float(scores["context_recall"]), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "max_latency_ms": max(latencies),
    }

    # Save results
    results_file = RESULTS_DIR / f"stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"STAGE {stage} RESULTS")
    print(f"{'='*50}")
    print(f"Faithfulness:       {result_summary['faithfulness']}")
    print(f"Answer Relevancy:   {result_summary['answer_relevancy']}")
    print(f"Context Precision:  {result_summary['context_precision']}")
    print(f"Context Recall:     {result_summary['context_recall']}")
    print(f"Avg Latency:        {result_summary['avg_latency_ms']}ms")
    print(f"Results saved to:   {results_file}")

    return result_summary


def compare_stages() -> None:
    """Load and compare all saved stage results."""
    result_files = list(RESULTS_DIR.glob("stage*.json"))
    if not result_files:
        print("No evaluation results found. Run evaluation first.")
        return

    results_by_stage = {}
    for f in sorted(result_files):
        with open(f) as fp:
            data = json.load(fp)
        stage = data["stage"]
        # Keep latest result per stage
        if stage not in results_by_stage or data["timestamp"] > results_by_stage[stage]["timestamp"]:
            results_by_stage[stage] = data

    print(f"\n{'='*80}")
    print("ABLATION STUDY — ALL STAGES COMPARISON")
    print(f"{'='*80}")
    print(f"{'Stage':<10} {'Faithfulness':<15} {'Ans Relevancy':<16} {'Ctx Precision':<16} {'Ctx Recall':<12} {'Avg Latency'}")
    print("-" * 80)
    for stage in sorted(results_by_stage.keys()):
        r = results_by_stage[stage]
        print(
            f"Stage {stage:<5} "
            f"{r['faithfulness']:<15} "
            f"{r['answer_relevancy']:<16} "
            f"{r['context_precision']:<16} "
            f"{r['context_recall']:<12} "
            f"{r['avg_latency_ms']}ms"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Stage to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare all saved results")
    parser.add_argument("--quick", action="store_true", help="Run on first 5 questions only")
    args = parser.parse_args()

    if args.compare:
        compare_stages()
    elif args.stage:
        test_set = load_test_set()
        max_q = 5 if args.quick else None
        run_evaluation(args.stage, test_set, max_questions=max_q)
    else:
        print("Usage:")
        print("  python tests/evaluation/ragas_evaluator.py --stage 1")
        print("  python tests/evaluation/ragas_evaluator.py --stage 2")
        print("  python tests/evaluation/ragas_evaluator.py --compare")
        print("  python tests/evaluation/ragas_evaluator.py --stage 1 --quick")