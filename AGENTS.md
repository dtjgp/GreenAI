# GreenAI - Codex Project Rules

## Purpose

Work as a coding and research assistant for the GreenAI repository. Prioritize
measurement-grounded evidence, reproducible energy analysis, and a publishable
connection between GPU training characterization and hybrid energy scheduling.

This file owns the shared project rules. `CLAUDE.md` only imports it; current priorities and acceptance contracts belong in
`GOALS.md`, and current facts come from the linked repository artifacts.

## Research Method Review

- Use `$research-method-review` when the task asks to assess a method,
  experimental design, novelty, rigor, or evidence-to-claim fit. Routine code
  maintenance, plotting, and wording corrections follow their affected artifact
  and verifier; a research topic alone does not trigger a full method review.
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
- Repository overview: `README.md`.
- When historical setup/publication context is needed, consult
  `Docs/reference/PROJECT_CONTEXT_20260410.md`; verify current facts separately.
- Evaluation metrics and baseline catalog:
  `Docs/reference/EVALUATION_METRICS_AND_BASELINES.md`.
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

## Evaluation Metrics and Baselines

- For experiment design, metric or baseline selection, and paper-claim
  packaging, read `Docs/reference/EVALUATION_METRICS_AND_BASELINES.md`.
- Before claiming optimization advantage, compare against scheduling
  heuristics and fixed-policy baselines.

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

### Research Retrieval

- For current project status and measured-energy facts, inspect repository
  artifacts first.
- Consult `llm-wiki` when the task needs prior literature, research positioning,
  baseline selection, experiment design, or manuscript sources. A local code
  or wording correction does not require wiki retrieval solely because it
  mentions power, optimization, or another research term.
- Follow the vault's `System/Reference/Operations/Agent_Context_Map.md` to
  choose the relevant canonical pages before searching. Within the vault,
  `Topics/Green_AI.md`, `Methods/Modeling/GBR.md`,
  `Methods/Optimization/MILP.md`, and `Topics/Edge_AI/Overview.md` are topic
  entrypoints; load only those relevant to the question.
- Supplement with `rg` or short-keyword `qmd search`. Use vector/hybrid retrieval
  only when the current runtime meets the vault's retrieval contract; read
  `System/Benchmarks/QMD_Retrieval/README.md` when that capability is needed.
- Verify research claims against the original paper or measured artifact;
  retrieval failure is not evidence that the knowledge is absent. Cite canonical
  wiki pages with `[[wikilink]]` references when appropriate.

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
