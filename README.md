## HW — CI/CD пайплайн с автотестами и проверкой на галлюцинации (RAGAS)

Цель: прогонять **golden-набор** через ваше LLM‑приложение и автоматически ставить **quality gates** по метрикам RAGAS в GitHub Actions.
  
---



- `app.py` 
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



## Настройка порогов (quality gates)

В workflow и локально можно менять пороги через переменные окружения:

- `MIN_FAITHFULNESS` (по умолчанию 0.70)
- `MIN_ANSWER_RELEVANCY` (0.40)
- `MIN_CONTEXT_RECALL` (0.70)
