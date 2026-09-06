# GreenAI - Codex Project Rules

## Purpose

Work as a coding and research assistant for the GreenAI repository. Prioritize
measurement-grounded evidence, reproducible energy analysis, and a publishable
connection between GPU training characterization and hybrid energy scheduling.

This file owns the shared project rules. `CLAUDE.md` imports it and retains only
navigation context; current priorities and acceptance contracts belong in
`GOALS.md`, and current facts come from the linked repository artifacts.

## Research Method Review

- For proposed methods, experiment designs, result interpretation, paper plans,
  novelty/rigor assessments, or claim-boundary reviews, use
  `$research-method-review`.
- Before a strong GreenAI judgment, inspect the measured repository artifacts
  first. The review skill does not override this project's measurement,
  reproducibility, or claim-boundary rules.

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
- Shared assistant protocol: this file; `CLAUDE.md` is the Claude adapter.
- Current goals and verifiers: `GOALS.md` when present.
- Measurement and modeling artifacts: `GPU_Performance/`.
- Scheduling and optimization artifacts: `Optimization/`.
- If prose conflicts with measured traces, processed CSVs, scripts, or logs,
  trust the executable/measured artifacts first.

## Key Paths

- GPU measurement and cross-hardware analysis: `GPU_Performance/`.
- Hybrid energy scheduling and optimization: `Optimization/`.
- Repository overview and historical notes: `README.md`.
- Shared claim boundaries and wiki routing: this file.

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

## Measurement Failure and Proportional Robustness

- Treat measurement collection, synchronization, parsing, or time-coverage
  failure as an invalid run, not as a successful result with degraded evidence.
- Never represent missing, corrupt, incomplete, or unverified energy data as
  zero energy, an empty successful table, or another valid measurement.
- Catch exceptions only to perform required cleanup, add actionable context and
  re-raise, or apply a recovery path whose scientific meaning is explicitly
  defined, observable, and verified.
- Keep recovery proportional to the failure. Do not add speculative abstraction,
  compatibility, configurability, retry, or fallback layers to measurement and
  scheduling code without a demonstrated requirement.
- A recovered or partial run may support a paper-facing result only when the
  recovery is recorded in the artifact and the governing measurement protocol
  explicitly permits it.

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
- Treat a goal as complete only when its success criteria and required
  acceptance verifier pass. Evidence discovery, keyword matches, partial
  verification, and a documented execution blocker do not complete a goal.
- A supported negative/no-go result may close a goal whose stated objective is
  to make that scientific decision; retain its evidence and claim boundary.
- For energy or scheduling claims, always report the data source, hardware,
  power limit, model, dataset, and measurement procedure.
- If a full rerun is too expensive, perform the strongest partial verification
  available and label the result as partial.

## LLM Wiki Integration

This project is connected to the central `llm-wiki` knowledge base:

- Wiki path: `/Users/dtjgp/Obsidian/llm-wiki`

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

- When the current task includes durable research writeback, follow the vault
  `AGENTS.md` and `_schema.md` before the first write; use its established
  insight, INGEST, or navigation workflow. Follow any applicable automation
  registration/lease contract without treating an ordinary task as an automation.
- Existing user authorization remains valid. Finish authorized repository work
  while preparing any genuinely undecided writeback as a reviewable proposal;
  do not infer permission to mutate the vault solely from a related keyword.

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
- Match checks to the changed behavior and declared goal. For instruction or
  wording edits, check links, consistency, and affected claims; do not run a
  measurement campaign or add tests that only repeat the prose.
- After relevant checks and required gates pass, broaden/repeat only for a new
  change, failure, or unresolved concern, then complete the authorized handoff.
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

## Manuscript Writing

- Before substantive manuscript drafting, result interpretation, or submission
  checks, read the relevant sections of
  `Docs/writing/ACADEMIC_WRITING_STYLE_GUIDE.md`. For a local wording correction,
  inspect the affected text, its evidence boundary, and the applicable style
  rule without reloading unrelated manuscript context.
- Use that guide for common manuscript structure and style. Current measured
  artifacts, this file's claim boundaries, and target-venue requirements take
  precedence whenever they impose a stricter scientific or formatting rule.
- Present paper-facing reasoning in the order: measured result, applicable
  scope, then interpretation.
- Put general external-validity caveats and future-work discussion in the
  Limitations section instead of scattering repeated caution across the paper.
- Keep hardware, sampling, synchronization, workload, power-limit, scheduling,
  and deadline constraints next to the claim they qualify when removing them
  would change the claim's meaning or validity.
- Do not dilute supported findings with template phrases such as "further
  research is needed" or "this result should be interpreted with caution" in
  every paragraph. State the exact boundary once, at the location where it
  affects the inference.
- Direct writing must remain evidence-calibrated: clearer prose never licenses
  stronger generalization, causal language, or certainty than the measured
  artifacts support.
