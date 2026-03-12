## HW — CI/CD пайплайн с автотестами и проверкой на галлюцинации (RAGAS)

Цель: прогонять **golden-набор** через ваше LLM‑приложение и автоматически ставить **quality gates** по метрикам RAGAS в GitHub Actions.
  
---

## Что внутри

- `app.py` — мини‑приложение (QA по контексту) на Mistral через OpenAI‑совместимый endpoint.
- `tests/goldens.json` — golden-набор (вопрос + контекст + reference/эталон).
- `ragas_eval.py` — оценка goldens метриками:
  - `Faithfulness` (анти‑галлюцинации),
  - `AnswerRelevancy`,
  - `ContextRecall`.
- `tests/test_ragas_quality.py` — pytest‑тест, который:
  - считает метрики,
  - пишет отчёт в `reports/ragas_results.json`,
  - **падает**, если средние значения ниже порогов.
- `.github/workflows/llm-quality.yml` — GitHub Actions workflow.

---

## Локальный запуск

```bash
export MISTRAL_API_KEY="..."
pip install -r requirements.txt
pytest -q
```

Результат будет в `reports/ragas_results.json`.

---

## Запуск в GitHub Actions

1. Запушь репозиторий на GitHub.
2. Добавь секрет: `Settings → Secrets and variables → Actions → New repository secret`:
   - `MISTRAL_API_KEY`
3. Сделай `git push` — workflow стартует сам.

---

## Настройка порогов (quality gates)

В workflow и локально можно менять пороги через переменные окружения:

- `MIN_FAITHFULNESS` (по умолчанию 0.70)
- `MIN_ANSWER_RELEVANCY` (0.40)
- `MIN_CONTEXT_RECALL` (0.70)

Если пайплайн “красный” — смотри артефакт `ragas-reports` и подтягивай:

- промпт в `app.py`,
- качество/полноту `retrieved_contexts` в `tests/goldens.json`,
- модель `MISTRAL_MODEL`.

---

**Теги:** #hw #cicd #tests #ragas #evaluation

