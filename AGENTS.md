# GreenAI - Codex Project Rules

## Purpose

Work as a coding and research assistant for the GreenAI repository. Prioritize
measurement-grounded evidence, reproducible energy analysis, and a publishable
connection between GPU training characterization and hybrid energy scheduling.

## Research Router Auto-Routing

- For any nontrivial research, literature, experiment-design, result-analysis,
  paper-writing, claim-audit, reproducibility, Zotero/Obsidian, MCP, plugin, or
  skill-selection task in this repository, use `$research-router` as the first
  routing layer before choosing specialized skills, MCP servers, or plugins.
- Treat this as the project default for GreenAI work, especially GPU energy
  measurement, power-performance modeling, MILP scheduling, carbon/cost
  analysis, benchmark design, and paper-facing tasks.
- If the current Codex session has not discovered `research-router` yet, follow
  the installed router files directly at
  `/Users/dtjgp/.codex/skills/research-router/` and note that a new Codex
  session is needed for automatic skill discovery.
- If the user explicitly names a skill, MCP server, plugin, or tool, prefer the
  requested route unless it conflicts with this project's measured-data,
  reproducibility, or claim-boundary rules.
- For GreenAI energy, carbon, cost, or scheduling claims, the router must still
  inspect measured repository artifacts first; it does not override the
  authoritative-source rules below.

## Project Charter

### One-Line Goal

Build a measurement-grounded framework for characterizing deep neural network
training energy under GPU power limits, and use those measurements to optimize
data-center energy cost and carbon footprint via hybrid solar-grid-battery
scheduling.

### Two Coupled Research Lines

- **A1. Measurement and Modeling**: empirically characterize how GPU power caps,
  model architecture, and training stages affect execution time, energy, and
  energy efficiency.
- **A2. Scheduling and Optimization**: use measured power-performance profiles
  to optimize training-time dispatch across solar generation, grid power,
  batteries, and selectable GPU power states.

## Authoritative Sources

- Current project overview: `README.md`.
- Current AI-assistant project charter: `CLAUDE.md`.
- Current goals and verifiers: `GOALS.md` when present.
- Measurement and modeling artifacts: `GPU_Performance/`.
- Scheduling and optimization artifacts: `Optimization/`.
- If prose conflicts with measured traces, processed CSVs, scripts, or logs,
  trust the executable/measured artifacts first.

## Key Paths

- GPU measurement and cross-hardware analysis: `GPU_Performance/`.
- Hybrid energy scheduling and optimization: `Optimization/`.
- Repository overview and historical notes: `README.md`.
- Assistant-facing project charter and wiki integration: `CLAUDE.md`.

## Claim Boundaries

- All energy, carbon, and cost conclusions must be grounded in measured hardware
  data.
- Do not use synthetic energy traces as evidence for the main project claim.
- GPU power-state decisions are discrete power-limit options, not continuous
  tuning.
- Battery charge/discharge efficiency must be modeled explicitly when scheduling
  claims involve storage.
- Scheduling objectives must respect training progress, deadline, or completion
  constraints; do not present pure energy minimization as sufficient.
- Cross-hardware comparisons require controlled dataset, batch size, epoch
  count, synchronization policy, sampling procedure, and environment.
- Exploratory architecture studies must be separated from the main publishable
  measurement-scheduling narrative.

## Evaluation Metrics

Measurement and modeling:

- Epoch time.
- Stage-wise execution time.
- Epoch energy.
- Stage-wise energy.
- Energy-delay tradeoff.
- Energy efficiency under power caps.
- Prediction error of power-performance models.

Scheduling and optimization:

- Electricity cost reduction.
- Carbon/emission reduction.
- Renewable energy utilization.
- Battery utilization efficiency.
- Training completion ratio or deadline satisfaction.
- Training time extension under constrained power.

## Baselines and Comparisons

- Full-power training without GPU power capping.
- Uniform fixed power-cap policies across the full training horizon.
- Pure-grid supply versus hybrid supply with solar and battery.
- Different GPU power-limit settings under matched training-budget or
  matched-energy-budget scenarios.
- Model-level and hardware-level comparisons across heterogeneous GPU platforms.
- Scheduling heuristics and fixed-policy baselines before claiming optimization
  advantage.

## Goal and Verification Protocol

- Before substantial implementation, analysis, paper-claim, or documentation
  work, read `GOALS.md` and identify which goal the work advances.
- Treat a goal as complete only when its success criteria and verifier pass, or
  when a blocker is documented with concrete evidence.
- For energy or scheduling claims, always report the data source, hardware,
  power limit, model, dataset, and measurement procedure.
- If a full rerun is too expensive, perform the strongest partial verification
  available and label the result as partial.

## LLM Wiki Integration

This project is connected to the central `llm-wiki` knowledge base:

- Wiki path: `/Users/dtjgp/Library/CloudStorage/GoogleDrive-dtjgp92613@gmail.com/My Drive/Obsidian/llm-wiki`

### Required Behavior

- For current project status and measured-energy facts, inspect repository
  artifacts first.
- For broader research positioning, modeling, optimization, scheduling,
  baseline selection, or paper writing, consult `llm-wiki`.
- Prefer `qmd` retrieval first, then read canonical wiki files directly when
  more detail is needed.
- When sandbox or reranking causes issues, fall back to `qmd search` or
  `qmd query --no-rerank`.
- When producing research conclusions, cite the relevant wiki pages with
  `[[wikilink]]` style references when appropriate.

### Recommended Retrieval Flow

1. Run `qmd query "your question"` for hybrid retrieval.
2. If needed, fall back to `qmd search "keywords"` or
   `qmd query "your question" --no-rerank`.
3. Read canonical pages directly:
   - `Topics/Green_AI.md`
   - `Methods/Modeling/GBR.md`
   - `Methods/Optimization/MILP.md`
   - `Topics/Edge_AI/Overview.md`
4. Check `Literature/Paper_Notes/` for paper-grounded evidence before making
   strong claims.

### Trigger Topics

- GPU power limit, power cap, training energy, stage-wise energy.
- Forward, backward, loss, optimize, and layer-level energy.
- GBR, regression, prediction error, heterogeneous GPUs.
- MILP, Gurobi, battery, solar, grid, hybrid scheduling.
- Carbon, electricity cost, renewable utilization, deadline satisfaction.
- Measurement protocol, reproducibility, benchmark design, reviewer response.

### Write-Back Rules

- New research insight worth keeping: write to `llm-wiki/Insights/`.
- Important new paper: follow the wiki INGEST workflow.
- New taxonomy or canonical-path change: update the relevant wiki navigation or
  docs.

## Coding Expectations

- Prefer Python unless the existing artifact requires notebooks or another
  toolchain.
- Keep edits scoped to the measurement, modeling, or optimization component
  involved.
- Do not overwrite raw measured data.
- When adding analysis code, document expected inputs and generated outputs.
- When changing optimization code, report solver assumptions, constraints, and
  objective terms.

## Verification Expectations

- For data analysis changes, re-run or inspect the affected processed output.
- For scheduling changes, verify feasibility, constraints, and objective values
  on at least one small scenario.
- For paper-facing numbers, trace them to measured data and scripts.
- For cross-hardware comparisons, check that hardware/model/dataset/power-limit
  metadata are present.

## Result Reporting Rules

- Separate measurement results, predictive modeling results, and scheduling
  optimization results.
- Label exploratory results explicitly.
- Do not present optimization gains without stating the baseline policy.
- Do not present carbon or cost reductions without stating energy source,
  price/carbon trace, and time horizon.
