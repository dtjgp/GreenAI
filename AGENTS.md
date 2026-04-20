# GreenAI - Codex Project Rules

## LLM Wiki Integration

This project is connected to the central `llm-wiki` knowledge base:

- Wiki path: `/Users/dtjgp/Library/CloudStorage/GoogleDrive-dtjgp92613@gmail.com/My Drive/Obsidian/llm-wiki`

### Required behavior

- Before answering research, modeling, optimization, scheduling, or paper-writing questions, consult `llm-wiki` first.
- Prefer `qmd` retrieval first, then read canonical wiki files directly when more detail is needed.
- When sandbox or reranking causes issues, fall back to `qmd search` or `qmd query --no-rerank`.
- When producing research conclusions, cite the relevant wiki pages with `[[wikilink]]` style references when appropriate.

### Recommended retrieval flow

1. Run `qmd query "your question"` for hybrid retrieval.
2. If needed, fall back to `qmd search "keywords"` or `qmd query "your question" --no-rerank`.
3. Read canonical pages directly:
   - `Topics/Green_AI.md`
   - `Methods/Modeling/GBR.md`
   - `Methods/Optimization/MILP.md`
   - `Topics/Edge_AI/Overview.md`
4. Check `Literature/Paper_Notes/` for paper-grounded evidence before making strong claims.

### Trigger topics

- GPU power limit, power cap, training energy, stage-wise energy
- GBR, regression, prediction error, cross-hardware comparison
- MILP, battery, solar, grid, hybrid energy scheduling
- carbon, electricity cost, renewable utilization, deadline satisfaction
- measurement protocol, reproducibility, benchmark design, reviewer response

### Write-back rules

- New research insight worth keeping: write to `llm-wiki/Insights/`
- Important new paper: follow the wiki INGEST workflow
- New taxonomy or canonical-path change: update the relevant wiki navigation/docs
