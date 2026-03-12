import os
from pathlib import Path

import pytest

from ragas_eval import evaluate_all, load_goldens, summarize, write_report


GOLDENS_PATH = Path(__file__).parent / "goldens.json"
REPORT_PATH = Path("reports") / "ragas_results.json"


@pytest.mark.asyncio
async def test_ragas_quality_gates():
    # Arrange
    if not os.getenv("MISTRAL_API_KEY"):
        if os.getenv("CI", "").lower() == "true":
            pytest.fail("MISTRAL_API_KEY is missing in CI (add it as a GitHub Actions secret).")
        pytest.skip("MISTRAL_API_KEY is not set (needed to run RAGAS judge).")

    goldens = load_goldens(GOLDENS_PATH)

    # Act
    results = await evaluate_all(goldens, concurrency=int(os.getenv("RAGAS_CONCURRENCY", "3")))
    write_report(results, REPORT_PATH)
    s = summarize(results)

    # Assert: quality gates (tune if needed)
    min_faithfulness = float(os.getenv("MIN_FAITHFULNESS", "0.70"))
    min_answer_relevancy = float(os.getenv("MIN_ANSWER_RELEVANCY", "0.40"))
    min_context_recall = float(os.getenv("MIN_CONTEXT_RECALL", "0.70"))

    assert s["faithfulness_avg"] >= min_faithfulness, s
    assert s["answer_relevancy_avg"] >= min_answer_relevancy, s
    assert s["context_recall_avg"] >= min_context_recall, s

