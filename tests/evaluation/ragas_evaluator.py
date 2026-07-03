"""
tests/evaluation/ragas_evaluator.py

Custom RAG evaluator using Groq as judge LLM.
Measures the same metrics as RAGAs but without the dependency hell:
- Faithfulness: does the answer stick to retrieved context only?
- Answer Relevancy: does the answer address the question?
- Context Precision: are retrieved chunks actually relevant?
- Context Recall: does context contain what's needed to answer?

Usage:
    python tests/evaluation/ragas_evaluator.py --stage 1 --quick
    python tests/evaluation/ragas_evaluator.py --stage 1
    python tests/evaluation/ragas_evaluator.py --compare
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from groq import Groq

from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.metadata_filter import build_filter
from src.configuration.config import GROQ_API_KEY
from src.shared.logger import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("tests/evaluation/results")
TEST_SET_PATH = Path("tests/evaluation/test_set_100q.json")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

groq_client = Groq(api_key=GROQ_API_KEY)


# ── Evaluation prompts ────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Task: Does the answer contain ONLY information that can be verified from the context above?
- Score 1.0 if every claim in the answer is supported by the context
- Score 0.5 if some claims are supported but the answer adds unsupported information
- Score 0.0 if the answer contains significant information not in the context

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

RELEVANCY_PROMPT = """You are evaluating whether an AI answer is relevant to the question.

QUESTION:
{question}

ANSWER:
{answer}

Task: Does the answer directly address what was asked?
- Score 1.0 if the answer completely and directly addresses the question
- Score 0.5 if the answer partially addresses the question or goes off-topic
- Score 0.0 if the answer does not address the question at all

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

CONTEXT_PRECISION_PROMPT = """You are evaluating whether retrieved context chunks are relevant to answering a question.

QUESTION:
{question}

RETRIEVED CONTEXT CHUNKS:
{context}

Task: What fraction of the retrieved chunks are actually useful for answering this question?
- Score 1.0 if all chunks are relevant and useful
- Score 0.5 if about half the chunks are relevant
- Score 0.0 if none of the chunks are relevant

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""

CONTEXT_RECALL_PROMPT = """You are evaluating whether retrieved context contains enough information to answer a question.

QUESTION:
{question}

EXPECTED ANSWER (ground truth):
{ground_truth}

RETRIEVED CONTEXT:
{context}

Task: Does the retrieved context contain the information needed to produce the expected answer?
- Score 1.0 if the context fully covers what's needed for the expected answer
- Score 0.5 if the context partially covers what's needed
- Score 0.0 if the context is missing the key information needed

Reply with ONLY a number between 0.0 and 1.0. Nothing else."""


def judge_with_groq(prompt: str) -> float:
    """Ask Groq to score a single metric. Returns float 0.0-1.0."""
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            score = float(text)
            return max(0.0, min(1.0, score))
        except (ValueError, Exception) as e:
            if attempt < 2:
                time.sleep(2)
            else:
                logger.warning(f"Judge failed after 3 attempts: {e}")
                return 0.5
    return 0.5


def evaluate_single(
    question: str,
    answer: str,
    chunks: list,
    ground_truth: str
) -> dict:
    """Evaluate one question-answer pair on all four metrics."""
    context_text = "\n\n---\n\n".join(
        f"[{c.metadata.get('company_name','?')} | Page {c.metadata.get('page_number','?')}]\n{c.page_content[:600]}"
        for c in chunks
    )

    # Small delay between Groq calls to stay under rate limits
    faithfulness = judge_with_groq(
        FAITHFULNESS_PROMPT.format(context=context_text, question=question, answer=answer)
    )
    time.sleep(1.5)

    relevancy = judge_with_groq(
        RELEVANCY_PROMPT.format(question=question, answer=answer)
    )
    time.sleep(1.5)

    precision = judge_with_groq(
        CONTEXT_PRECISION_PROMPT.format(question=question, context=context_text)
    )
    time.sleep(1.5)

    recall = judge_with_groq(
        CONTEXT_RECALL_PROMPT.format(
            question=question, ground_truth=ground_truth, context=context_text
        )
    )
    time.sleep(1.5)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "context_recall": recall,
    }


def load_test_set() -> list[dict]:
    if not TEST_SET_PATH.exists():
        print("Test set not found. Creating sample...")
        sample = create_sample_test_set()
        TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_SET_PATH, "w") as f:
            json.dump(sample, f, indent=2)
        print(f"Sample test set saved to {TEST_SET_PATH}")
        return sample
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def create_sample_test_set() -> list[dict]:
    return [
        {
            "id": "q001", "category": "factual",
            "question": "What are the main risk factors mentioned in Zomato's DRHP?",
            "ground_truth": "Zomato's DRHP mentions risk factors including intense competition from Swiggy, regulatory risks, dependence on restaurant partners, and unit economics challenges.",
            "company_filter": ["Zomato"], "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q002", "category": "factual",
            "question": "What is Zomato's primary business model as described in their DRHP?",
            "ground_truth": "Zomato operates a food delivery marketplace connecting consumers with restaurant partners, earning commission from restaurants and delivery fees from consumers.",
            "company_filter": ["Zomato"], "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q003", "category": "factual",
            "question": "What were the objects of Paytm's IPO as mentioned in their DRHP?",
            "ground_truth": "Paytm's IPO proceeds were planned for growing the Paytm ecosystem, expanding merchant and consumer base, and general corporate purposes.",
            "company_filter": ["Paytm"], "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q004", "category": "factual",
            "question": "What are the main risk factors for Paytm mentioned in their DRHP?",
            "ground_truth": "Paytm's DRHP mentions risks including regulatory uncertainty in payments, competition from banks and fintech companies, and profitability concerns.",
            "company_filter": ["Paytm"], "doc_type_filter": ["DRHP"]
        },
        {
            "id": "q005", "category": "factual",
            "question": "What business does Ola Electric operate as described in their DRHP?",
            "ground_truth": "Ola Electric is an electric vehicle manufacturer focused on electric scooters and motorcycles with vertically integrated manufacturing operations in India.",
            "company_filter": ["Ola Electric"], "doc_type_filter": ["DRHP"]
        },
    ]


def run_evaluation(stage: int, test_questions: list[dict], max_questions: int = None) -> dict:
    if max_questions:
        test_questions = test_questions[:max_questions]

    print(f"\nRunning Stage {stage} evaluation on {len(test_questions)} questions...")
    print("Using Groq (Llama 3.3 70B) as evaluation judge — no external dependencies.\n")

    retriever = BaseRetriever()

    all_scores = []
    latencies = []
    failed = 0

    for i, item in enumerate(test_questions, 1):
        print(f"  [{i}/{len(test_questions)}] {item['question'][:65]}...")

        try:
            meta_filter = build_filter(
                companies=item.get("company_filter"),
                doc_types=item.get("doc_type_filter")
            )
            start = time.time()
            result = retriever.query(item["question"], metadata_filter=meta_filter)
            latency = round((time.time() - start) * 1000)
            latencies.append(latency)

            scores = evaluate_single(
                question=item["question"],
                answer=result["answer"],
                chunks=result["chunks"],
                ground_truth=item["ground_truth"]
            )
            all_scores.append(scores)

            print(f"           F:{scores['faithfulness']:.2f} "
                  f"R:{scores['answer_relevancy']:.2f} "
                  f"P:{scores['context_precision']:.2f} "
                  f"Rc:{scores['context_recall']:.2f} "
                  f"({latency}ms)")

            # Pause between questions to respect Groq rate limits
            time.sleep(3)

        except Exception as e:
            logger.error(f"Question {item['id']} failed: {e}")
            failed += 1
            all_scores.append({
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0
            })
            latencies.append(0)

    # Aggregate scores
    def avg(metric):
        vals = [s[metric] for s in all_scores]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    result_summary = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(test_questions),
        "failed_questions": failed,
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "individual_scores": all_scores,
    }

    results_file = RESULTS_DIR / f"stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(result_summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"STAGE {stage} EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"Questions evaluated:  {len(test_questions)} ({failed} failed)")
    print(f"Faithfulness:         {result_summary['faithfulness']}")
    print(f"Answer Relevancy:     {result_summary['answer_relevancy']}")
    print(f"Context Precision:    {result_summary['context_precision']}")
    print(f"Context Recall:       {result_summary['context_recall']}")
    print(f"Avg Query Latency:    {result_summary['avg_latency_ms']}ms")
    print(f"Results saved to:     {results_file}")

    return result_summary


def compare_stages() -> None:
    result_files = list(RESULTS_DIR.glob("stage*.json"))
    if not result_files:
        print("No evaluation results found. Run: python tests/evaluation/ragas_evaluator.py --stage 1 --quick")
        return

    results_by_stage = {}
    for f in sorted(result_files):
        with open(f) as fp:
            data = json.load(fp)
        stage = data["stage"]
        if stage not in results_by_stage or data["timestamp"] > results_by_stage[stage]["timestamp"]:
            results_by_stage[stage] = data

    print(f"\n{'='*85}")
    print("ABLATION STUDY — STAGE COMPARISON")
    print(f"{'='*85}")
    print(f"{'Stage':<8} {'Faithfulness':<14} {'Ans Relevancy':<15} {'Ctx Precision':<15} {'Ctx Recall':<12} {'Latency'}")
    print("-" * 85)
    for stage in sorted(results_by_stage.keys()):
        r = results_by_stage[stage]
        print(
            f"Stage {stage:<3} "
            f"{r['faithfulness']:<14} "
            f"{r['answer_relevancy']:<15} "
            f"{r['context_precision']:<15} "
            f"{r['context_recall']:<12} "
            f"{r['avg_latency_ms']}ms"
        )
    print(f"{'='*85}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--compare", action="store_true")
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
        print("  python tests/evaluation/ragas_evaluator.py --stage 1 --quick")
        print("  python tests/evaluation/ragas_evaluator.py --stage 1")
        print("  python tests/evaluation/ragas_evaluator.py --compare")