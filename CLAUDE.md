# CLAUDE.md - AI Assistant Configuration for GreenAI Project

**Last Updated**: 2026-04-10
**Project**: GreenAI - Measurement-Grounded GPU Training Energy Characterization and Hybrid Energy Scheduling

This file provides working guidance for AI assistants when operating in this repository.

---

## Project Charter

### One-Line Goal
Build a measurement-grounded framework for characterizing deep neural network training energy under GPU power limits, and use those measurements to optimize data-center energy cost and carbon footprint via hybrid solar-grid-battery scheduling.

### Two Coupled Research Lines
- **A1. Measurement & Modeling** — empirically characterize how GPU power caps, model architecture, and training stages affect execution time, energy, and energy efficiency
- **A2. Scheduling & Optimization** — use measured power-performance profiles to optimize training-time dispatch across solar generation, grid power, batteries, and selectable GPU power states

### Scenarios & Objects
- Heterogeneous AI training environments: local (`RTX 3060`, `RTX 4070`, `Apple M1`, `Apple M3`) and cloud (`RTX 3080`, `RTX 4090`)
- Workloads spanning CNN families: `AlexNet`, `VGG`, `ResNet`, `GoogLeNet` variants, `MobileNet`, plus selected `ViT` tests
- Hybrid-powered AI data center with solar generation, grid purchase/sell-back, and battery storage

### Core Methods
- Fine-grained stage-level GPU energy measurement (`to_device`, `forward`, `loss`, `backward`, `optimize`) plus layer-level profiling for representative models
- GPU power-performance modeling under discrete power caps via **GBR (Gradient Boosting Regression)** and related regression
- Cross-hardware comparison linking architecture descriptors → time, energy, efficiency
- **MILP / Gurobi**-based energy scheduling with discrete GPU power-state selection, solar/grid/battery dispatch, battery efficiency (~0.9), optional grid sell-back
- PVWatts-based solar generation modeling

### Key Assumptions & Constraints (Red Lines)
- **All energy conclusions must be grounded in real measured hardware data — no synthetic traces**
- GPU power is controlled through **discrete** power-limit options, not continuous tuning
- Battery charge/discharge efficiency modeled explicitly
- Scheduling is constrained by training-progress / completion requirements, not pure energy minimization
- Cross-hardware comparability requires controlled dataset, batch size, epoch count, synchronization, sampling

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

## Manuscript Writing

- Before drafting, revising, polishing, interpreting results, writing captions,
  or performing submission checks, read
  `Docs/writing/ACADEMIC_WRITING_STYLE_GUIDE.md`.
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

### Evaluation Metrics
- **Measurement / Modeling**: epoch time, stage-wise time, epoch energy, stage-wise energy, energy-delay tradeoff, efficiency under power caps, model prediction error
- **Scheduling / Optimization**: electricity cost reduction, carbon reduction, renewable utilization, battery utilization, training completion ratio / deadline satisfaction, training-time extension under constrained power

### Baselines & Comparisons
- Full-power training (no power cap)
- Uniform fixed power-cap policies across the full horizon
- Pure-grid supply vs. hybrid supply (solar + battery)
- Different power-limit settings under matched training-budget or matched-energy-budget scenarios
- Model-level and hardware-level cross-platform comparisons

### Current Progress
- ✅ Multi-platform measurement repository across local + cloud hardware
- ✅ Large-scale energy/performance artifacts including epoch-level and labeled stage-level traces
- ✅ Reusable energy-labeling and stage-decomposition utilities
- ✅ Gurobi-based scheduling formulations with GPU power-state decisions and hybrid dispatch
- ✅ PVWatts renewable supply modeling workflow
- ✅ Published: *Energy Sustainability Analysis of Deep Neural Network*, **MSWiM 2025**, DOI `10.1109/MSWiM67937.2025.11309062`

### Current Bottlenecks
- End-to-end narrative connecting measurement/modeling → optimization gains needs tightening
- Power-performance prediction model needs validation across more workloads and hardware
- Scheduling lacks systematic heuristic / fixed-policy baselines
- Reproducibility layer needs upgrade: experiment registry, data schema, configuration logs, seed/environment docs
- Need clearer separation between exploratory studies and the main publishable claim

### Near-Term Milestones (2–6 weeks)
1. Consolidate canonical benchmark subset (models × datasets × GPUs × power levels) for the main paper
2. Finalize per-GPU power-performance curves; rigorously quantify prediction error
3. Add stronger scheduling baselines; report cost / carbon / deadline tradeoffs under matched scenarios
4. Reorganize results into paper-ready structure: empirical characterization → predictive modeling → optimization outcomes
5. Package measurement + analysis pipeline into a reproducible open-source workflow

### Paper / System Goals
- **Primary publishable angle**: measurement-grounded green AI training under controllable GPU power limits
- **Secondary angle**: hybrid-energy-aware scheduling calibrated by empirically measured GPU power-performance models
- **Target venues**: ACM eEnergy, IEEE INFOCOM workshops, sustainable computing / green AI venues
- **Contribution framing**:
  1. Heterogeneous GPU measurement study of training energy under power caps
  2. Fine-grained stage-level energy characterization linking model structure ↔ training behavior
  3. Predictive power-performance model usable by downstream schedulers
  4. Hybrid-energy scheduling framework calibrated by real measured GPU behavior
- **Open-source plan**: release measurement-analysis-scheduling toolchain with reproducible configs, raw/processed data schema, benchmark instructions

---

## LLM Wiki 集成

本项目连接到统一的 `llm-wiki` 知识库，用于研究辅助、知识校对、baseline 检索与论文写作支撑。

**Wiki 路径**: `/Users/dtjgp/Obsidian/llm-wiki`

**使用规则**：
1. 讨论学术问题、实验设计、建模、优化或论文写作时，先查 `llm-wiki`
2. 优先使用 `qmd` 检索；需要更多细节时再直接读取 canonical wiki 页面
3. 如果 sandbox / rerank 路径不稳定，回退到 `qmd search` 或 `qmd query --no-rerank`
4. 输出研究结论时，尽量附上对应 wiki 页面引用
5. 新研究洞察、新论文笔记、taxonomy 变化应回写 `llm-wiki`

**推荐检索命令**：
- 关键词搜索：`/opt/homebrew/bin/qmd search "gpu power limit training energy"`
- 混合查询：`/opt/homebrew/bin/qmd query "hybrid energy scheduling for gpu training"`
- 回退查询：`/opt/homebrew/bin/qmd query "gpu training energy model" --no-rerank`

**优先读取的 canonical 页面**：
- `Topics/Green_AI.md`
- `Methods/Modeling/GBR.md`
- `Methods/Optimization/MILP.md`
- `Topics/Edge_AI/Overview.md`

**触发关键词**（出现时优先查 wiki）：
GPU power limit, power cap, training energy, stage-wise energy, forward/backward energy,
GBR, regression, prediction error, heterogenous GPUs, MILP, battery, solar, grid,
hybrid scheduling, renewable utilization, carbon, electricity cost, reproducibility,
benchmark design, reviewer response, 实验设计, 方法对比, 能耗建模, 优化调度

---

## Project Focus

- Measurement-grounded GPU energy characterization under discrete power caps
- Cross-hardware modeling of performance, energy, and efficiency
- Hybrid solar-grid-battery scheduling for AI training workloads
- Reproducible evaluation of cost, carbon, and training-completion tradeoffs
