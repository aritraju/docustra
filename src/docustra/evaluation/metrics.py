"""
RAG evaluation using RAGAS metrics:
  - Faithfulness: is the answer grounded in the retrieved context?
  - Answer Relevancy: does the answer address the question?
  - Context Precision: are retrieved chunks actually used?
  - Context Recall: does the retrieved context cover the answer?
"""
from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from docustra.core import EvaluationError, get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
        }


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> EvaluationResult:
    if not (len(questions) == len(answers) == len(contexts)):
        raise EvaluationError("questions, answers, and contexts must have the same length.")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)

    try:
        result = evaluate(dataset, metrics=metrics)
        return EvaluationResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result.get("context_recall", 0.0),
        )
    except Exception as e:
        raise EvaluationError(f"RAGAS evaluation failed: {e}") from e
