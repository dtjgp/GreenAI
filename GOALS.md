# GreenAI Goals

This file turns the GreenAI project plan into Codex-executable goals. Use
`README.md`, `CLAUDE.md`, measured data directories, and current scripts as the
source of truth for project facts.

## Definition of Done

A goal is complete only when:

- The required measured-data or code artifacts exist.
- The verifier, or the strongest feasible subset, has been run.
- The result is traceable to hardware, model, dataset, power limit, and
  measurement procedure.
- Paper-facing claims are updated conservatively and do not rely on synthetic
  energy traces.

## Goal 1: Canonical Benchmark Subset

Status: active
Priority: P0
Project line: A1 Measurement and Modeling

### Objective

Consolidate a paper-ready benchmark subset of models, datasets, GPUs, and
discrete power levels for the main measurement study.

### Success Criteria

- [ ] A benchmark registry exists with model, dataset, hardware, power limit,
      synchronization policy, batch size, epoch count, and data path.
- [ ] Local and cloud GPU measurements are clearly separated.
- [ ] Exploratory architecture variants are labeled as exploratory rather than
      main evidence.
- [ ] Missing combinations are listed explicitly.

### Trusted Inputs

- `README.md`
- `CLAUDE.md`
- `GPU_Performance/`

### Verifier

```bash
find GPU_Performance -maxdepth 3 -type f | sort | sed -n '1,160p'
rg -n "RTX|M1|M3|power|Power|AlexNet|VGG|ResNet|GoogLeNet|MobileNet|ViT" README.md CLAUDE.md GPU_Performance
```

## Goal 2: Power-Performance Model Validation

Status: active
Priority: P0
Project line: A1 Measurement and Modeling

### Objective

Finalize per-GPU power-performance curves and quantify prediction error for
GBR (Gradient Boosting Regression) and related regression models.

### Success Criteria

- [ ] Training speed and energy curves are available for each canonical GPU and
      power limit.
- [ ] Prediction error is reported using a consistent metric such as MAE, RMSE,
      MAPE, or R2.
- [ ] Train/test split or cross-validation policy is documented.
- [ ] Model inputs and outputs are reproducible from measured artifacts.

### Trusted Inputs

- `GPU_Performance/`
- power-limit analysis notebooks or scripts referenced in `README.md`

### Verifier

```bash
rg -n "GBR|Gradient|Regression|TrainSpeed|PowerLimit|EnergySaving|prediction|MAE|RMSE|MAPE|R2" GPU_Performance README.md CLAUDE.md
```

## Goal 3: Scheduling Baselines

Status: active
Priority: P1
Project line: A2 Scheduling and Optimization

### Objective

Add systematic scheduling baselines for the hybrid solar-grid-battery training
scheduler.

### Success Criteria

- [ ] Full-power pure-grid baseline exists.
- [ ] Uniform fixed power-cap baseline exists.
- [ ] Hybrid solar-grid-battery optimized policy exists.
- [ ] At least one simple heuristic baseline exists, such as solar-first,
      deadline-greedy, or cost-threshold policy.
- [ ] Cost, carbon, renewable utilization, battery utilization, and deadline
      satisfaction are reported under matched scenarios.

### Trusted Inputs

- `Optimization/`
- `CLAUDE.md`
- PVWatts and scheduling code referenced by the project notes

### Verifier

```bash
find Optimization -maxdepth 4 -type f | sort | sed -n '1,160p'
rg -n "Gurobi|MILP|battery|solar|grid|PVWatts|deadline|cost|carbon|baseline|heuristic" Optimization README.md CLAUDE.md
```

## Goal 4: Reproducibility Layer

Status: active
Priority: P1
Project line: shared infrastructure

### Objective

Create a reproducibility layer that makes measurement, modeling, and scheduling
results auditable.

### Success Criteria

- [ ] Data schema or registry documents raw and processed artifacts.
- [ ] Environment, package versions, hardware, power limits, and sampling
      policy are recorded.
- [ ] Canonical commands are documented for measurement analysis and scheduling.
- [ ] Generated tables/figures can be regenerated from scripts or notebooks.
- [ ] Raw data are never overwritten by analysis scripts.

### Trusted Inputs

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `GPU_Performance/`
- `Optimization/`

### Verifier

```bash
rg -n "schema|registry|environment|seed|version|command|sampling|nvidia-smi|powermetrics|CodeCarbon|csv|database" README.md CLAUDE.md AGENTS.md GPU_Performance Optimization
```

## Goal 5: Paper Narrative Packaging

Status: active
Priority: P1
Project line: paper/system

### Objective

Reorganize results into a paper-ready narrative that connects empirical
characterization, predictive modeling, and optimization outcomes.

### Success Criteria

- [ ] Contributions are separated into measurement, modeling, and scheduling.
- [ ] Baselines are explicit for both measurement and scheduling claims.
- [ ] Limitations are stated, especially hardware coverage and measurement
      comparability.
- [ ] Open-source release plan includes data schema, configs, and reproduction
      instructions.

### Trusted Inputs

- `README.md`
- `CLAUDE.md`
- `GPU_Performance/`
- `Optimization/`

### Verifier

```bash
rg -n "publication|MSWiM|DOI|contribution|baseline|limitation|reproducible|open-source|paper|figure|table" README.md CLAUDE.md AGENTS.md GPU_Performance Optimization
```
