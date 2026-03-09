# Bottom Line & Dashboard

## What it is

The **natural-language bottom line** is a short paragraph (2–4 sentences) that summarizes the Snoqualmie Pass forecast in plain language: teleconnection interpretation, ensemble totals vs history, and analog years. It is designed to **add color and weight to a human’s distillation** of other sources and experience—not replace it.

## How it’s generated

- **Template mode** (default): No API key. Built from forecast results, analog years, and current teleconnection state. Run `python forecast.py` and the bottom line is written to `data/bottom_line.json`.
- **LLM mode** (optional): Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment. The same context is sent to the API and the model returns a richer narrative. On failure, the template is used.

## Output file: `data/bottom_line.json`

```json
{
  "bottom_line": "Winter 2026 outlook at Snoqualmie Pass: ...",
  "human_notes": "",
  "context": { "target_year": 2026, "forecast_months": [...], "analog_years": [...], "teleconnection": {...} },
  "generated_at": "2026-02-26T..."
}
```

- **`bottom_line`**: Generated text (template or LLM).
- **`human_notes`**: Empty by default. The dashboard (or a human) can append or replace this with the forecaster’s own distillation and experience-based notes.
- **`context`**: Structured data used to generate the bottom line (for reference or re-generation).

## Dashboard usage

1. **Load** `data/bottom_line.json` when the dashboard starts (or on a “Refresh” action).
2. **Show** the `bottom_line` string prominently (e.g. “Bottom line” or “Summary”).
3. **Provide an editable field** for `human_notes` so the forecaster can:
   - Add their own interpretation from other sources (models, obs, experience).
   - Note caveats or confidence.
   - Combine the generated bottom line with their distillation.
4. **Persist `human_notes`** back to `data/bottom_line.json` when the user saves (so the next run of `forecast.py` does not overwrite human notes—either merge before writing or only update `bottom_line` and `context`/`generated_at` and leave `human_notes` as-is when re-running the pipeline).

Recommended: show “Generated bottom line” and “Your notes” (or “Forecaster comment”) as two blocks so the human’s distillation stays visible alongside the auto-generated text.

## Regenerating the bottom line

- Run `python forecast.py`; it builds context from the latest forecast and analogs, generates the bottom line, and writes `data/bottom_line.json`.
- To force template-only (no LLM): in code call `generate_bottom_line(ctx, use_llm=False)`.
- To force LLM when a key is set: `generate_bottom_line(ctx, use_llm=True)`.

## Lidar (future)

See **`docs/LIDAR_IDEAS.md`** for how high-res lidar could be used later (snow depth/SWE mapping, terrain for cross sections, canopy, operations). No dashboard changes for lidar until that work is prioritized.
