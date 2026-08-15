# GreenAI Academic Writing Style Guide

**Project copy:** 2026-08-12
**Purpose:** provide a concise, stable writing and revision standard for
GreenAI manuscripts. This guide is source-derived from the user's
`Writing_style_guide.md` version 1.1 (SHA-256:
`743a41f6e9c4e8444ce4108e099fc7e074c684511beb06680a0f84b77bdfa116`)
and is adapted to this project's measurement-grounded evidence rules.

This guide controls style and manuscript structure. It does not override
`AGENTS.md`, `CLAUDE.md`, `GOALS.md`, measured traces, processed data, analysis
scripts, logs, or a target venue's mandatory instructions. Live measured and
executable artifacts remain authoritative for scientific claims.

## 1. Objective and priority

Write clear, natural engineering papers that connect measured GPU behavior to
predictive modeling and hybrid-energy scheduling without blurring their evidence
types. A reader should understand what was found, under which measurement or
scenario conditions, why it matters, and how it supports the paper's argument.

When revising, prioritize:

1. scientific correctness;
2. measurement traceability and claim alignment;
3. logical flow;
4. readability;
5. conciseness; and
6. venue-level presentation.

Never trade accuracy or reproducibility for stronger prose.

## 2. Preserve the evidence chain

Organize the main scientific story as:

```text
Measured GPU behavior → validated power-performance model →
measurement-calibrated scheduling result
```

Keep these evidence types distinct:

- **Measurement:** hardware observations produced by the declared workload,
  power limit, synchronization, sampling, and aggregation protocol.
- **Predictive modeling:** estimates whose validity is bounded by training data,
  features, validation split, error metrics, and tested hardware/workloads.
- **Scheduling:** optimized or simulated outcomes under declared price, carbon,
  renewable, battery, solver, and completion/deadline assumptions.

Do not present a model estimate as a hardware measurement or an optimization
outcome as an observed operational saving. Synthetic energy traces cannot
support the main measurement claim.

## 3. Claims, scope, and limitations

- Report paper-facing reasoning as **measured result → applicable scope →
  interpretation**.
- For every energy or performance claim, state the hardware, model, dataset,
  discrete power limit, and relevant measurement protocol.
- Cross-hardware comparisons additionally require matched batch size, epoch
  count, synchronization policy, sampling procedure, and environment.
- State the baseline and comparison budget before reporting an energy, latency,
  cost, carbon, or scheduling improvement.
- Keep scheduling conclusions conditional on the declared training-progress,
  completion, deadline, battery-efficiency, and energy-source assumptions.
- Use causal language only when the experiment changes the claimed factor under
  a design that controls the relevant alternatives.
- Treat missing, corrupt, incomplete, or insufficiently covered measurements as
  invalid evidence. A recovered or partial run must be labeled according to the
  governing measurement protocol.

Put assumptions that determine a claim's validity next to that claim. Put broad
external-validity caveats and meaningful future work in Limitations. Do not
scatter generic phrases such as “further research is needed,” “may not be
generalizable,” or “this result should be interpreted with caution” across the
manuscript. Name the exact boundary once where it changes the inference.

## 4. Build an argument, not a technical report

| Section | Primary job |
|---|---|
| Introduction | establish the energy/computing problem, gap, question, approach, and contribution |
| Related work | position the study against measurement, Green AI, and scheduling evidence |
| Measurement protocol | define hardware, workload, power states, synchronization, sampling, and validity rules |
| Modeling / optimization | define features, validation, objective, constraints, scenarios, and baselines |
| Results | report evidence and its immediate interpretation |
| Discussion | explain mechanisms, trade-offs, implications, scope, and limitations |
| Conclusion | answer the research question without introducing new results |

The Introduction should normally progress from **problem → knowledge gap →
research question → approach → contribution**. Give each paragraph one purpose
and each sentence one main message. Prefer concrete subjects and active verbs;
use passive voice when the actor is irrelevant or conventional methods wording
is clearer.

Use one stable term for each metric, stage, power state, hardware platform, and
scheduling quantity. Define abbreviations, symbols, units, aggregation windows,
time bases, system boundaries, and carbon or price sources before relying on
them.

## 5. Results and Discussion

Do not write Results as a sequence of figure descriptions. Use:

```text
Finding → evidence → interpretation → importance / transition
```

Lead with the finding; cite the figure or table as evidence. Report only the
numbers needed for the claim. Let tables provide complete values and figures
show patterns, trade-offs, and mechanisms.

Separate measurement, prediction, and scheduling subsections unless the paper
explicitly connects them through a traceable calibrated input. Explain model
error before using predictions downstream. Report feasibility, constraint
satisfaction, and deadline or completion outcomes alongside scheduling
objectives.

The Discussion should interpret rather than repeat. Explain plausible physical,
computational, or scheduling mechanisms and distinguish them from speculation.
Do not treat a high objective value or reduced energy alone as sufficient when
latency, training progress, completion, cost, carbon, or battery behavior changes
the practical meaning.

## 6. Language and claim tone

- Prefer clear, common technical English to ornamental vocabulary.
- State supported findings directly without empty emphasis such as “it is worth
  noting that” or “the results clearly demonstrate.”
- Avoid promotional terms such as *groundbreaking*, *transformative*,
  *cutting-edge*, or *state-of-the-art* without a defined comparison.
- Use *suggests*, *indicates*, *is consistent with*, *may reflect*, or *appears
  to* when the evidence is observational, narrow, or uncertain.
- Prefer direct descriptions of code and model behavior to vague abstractions.
- Do not vary terminology merely to create stylistic variety.

Direct, confident narrative remains evidence-calibrated. It never licenses
unsupported generalization, causality, cross-hardware transfer, or real-world
deployment claims.

## 7. Figures, tables, equations, and citations

Every figure must answer an analytical question. Use tables for exact values,
benchmark definitions, assumptions, and complete results; use figures for
patterns, contrasts, and trade-offs. Captions should identify hardware or
scenario, workload, power limit, units, aggregation basis, and uncertainty when
needed for independent interpretation.

Keep colors, panel labels, scales, terminology, units, symbols, number formats,
and uncertainty summaries consistent. Do not hide invalid runs, excluded data,
or unmatched comparison conditions in a footnote when they affect the main
claim.

Use citations as evidence: prefer original papers for methods, official sources
for price/carbon or policy inputs, and manufacturer or primary documentation for
hardware specifications. Place citations close to the supported statement. A
citation cannot substitute for repository measurements or repair a comparison
whose experimental controls are missing.

## 8. Revision workflow and final checks

Revise in this order:

1. **Evidence-chain pass:** verify the measurement → model → scheduling link.
2. **Claim pass:** trace each number, qualifier, comparison, and implication to a
   trusted artifact or source.
3. **Structure pass:** repair section roles, paragraph purpose, transitions,
   repetition, and figure/table ordering.
4. **Language pass:** simplify wording, normalize terminology, and remove empty
   emphasis and generic caveats.
5. **Rendered-output pass:** inspect the PDF for overflow, clipped labels,
   unreadable fonts, inconsistent typography, and malformed references.

Before accepting a draft, verify that:

- every result identifies its evidence type and applicable scope;
- measurement, prediction, and scheduling claims remain distinguishable;
- comparisons use declared baselines and matched conditions;
- invalid or partial evidence has not entered a main result silently;
- every figure and table supports the argument;
- terminology, units, citations, and metric definitions are consistent; and
- the manuscript reads as a coherent journal article rather than a process log,
  technical report, or reviewer-response document.
