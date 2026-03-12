import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from app import DEFAULT_BASE_URL, DEFAULT_MODEL, answer_question

try:
    # Older RAGAS versions (and our existing notebook) use this name.
    from ragas.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    # Newer RAGAS versions use this name.
    from ragas.embeddings import HuggingfaceEmbeddings as _HuggingFaceEmbeddings


@dataclass(frozen=True)
class Golden:
    id: str
    user_input: str
    retrieved_contexts: List[str]
    reference: str


@dataclass(frozen=True)
class SampleResult:
    id: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    response: str


def load_goldens(path: str | os.PathLike[str]) -> List[Golden]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Golden(
            id=item["id"],
            user_input=item["user_input"],
            retrieved_contexts=list(item["retrieved_contexts"]),
            reference=item["reference"],
        )
        for item in raw
    ]


def _ragas_llm() -> Any:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY env var")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("MISTRAL_BASE_URL", DEFAULT_BASE_URL),
    )
    model = os.getenv("MISTRAL_MODEL", DEFAULT_MODEL)
    # Use OpenAI-compatible client with Mistral base_url.
    # Avoid provider-specific adapters to keep CI setup minimal.
    return llm_factory(model, client=client)


def _embeddings() -> _HuggingFaceEmbeddings:
    return _HuggingFaceEmbeddings(model=os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))


async def evaluate_one(g: Golden) -> SampleResult:
    llm = _ragas_llm()
    embeddings = _embeddings()

    response = await answer_question(g.user_input, g.retrieved_contexts)
    sample = SingleTurnSample(
        user_input=g.user_input,
        response=response,
        retrieved_contexts=g.retrieved_contexts,
        reference=g.reference,
    )

    faith = await Faithfulness(llm=llm).ascore(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=sample.retrieved_contexts,
    )
    ar = await AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
        user_input=sample.user_input,
        response=sample.response,
    )
    cr = await ContextRecall(llm=llm).ascore(
        user_input=sample.user_input,
        retrieved_contexts=sample.retrieved_contexts,
        reference=sample.reference,
    )

    return SampleResult(
        id=g.id,
        faithfulness=float(faith.value),
        answer_relevancy=float(ar.value),
        context_recall=float(cr.value),
        response=response,
    )


async def evaluate_all(goldens: Sequence[Golden], concurrency: int = 3) -> List[SampleResult]:
    sem = asyncio.Semaphore(concurrency)

    async def _run(g: Golden) -> SampleResult:
        async with sem:
            return await evaluate_one(g)

    return list(await asyncio.gather(*[_run(g) for g in goldens]))


def summarize(results: Iterable[SampleResult]) -> dict[str, float]:
    rs = list(results)
    n = max(len(rs), 1)
    return {
        "faithfulness_avg": sum(r.faithfulness for r in rs) / n,
        "answer_relevancy_avg": sum(r.answer_relevancy for r in rs) / n,
        "context_recall_avg": sum(r.context_recall for r in rs) / n,
    }


def write_report(results: Sequence[SampleResult], path: str | os.PathLike[str]) -> None:
    out = {
        "summary": summarize(results),
        "results": [asdict(r) for r in results],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

