# GreenAI Goals

This file turns the GreenAI project plan into Codex-executable goals. Use
`README.md`, measured data directories, and current scripts as the source of
truth for project facts. `AGENTS.md` owns the shared evidence and execution rules.

## Definition of Done

A goal is complete only when:

- The required measured-data or code artifacts exist.
- The required acceptance checks have passed and their artifact paths, exact
  commands or review records, and outcomes are recorded.
- Discovery searches and a strongest-feasible subset alone remain partial.
  An unavailable input, environment, or verifier remains blocked/open.
- The result is traceable to hardware, model, dataset, power limit, and
  measurement procedure.
- Paper-facing claims are updated conservatively and do not rely on synthetic
  energy traces.

## Acceptance Status And Stop Conditions

The discovery commands below locate candidate evidence; their exit codes are
not scientific acceptance. The acceptance sections define the required checks,
not a claim that an automated verifier already exists or has passed. During
authorized goal work, bind each check to an actual artifact and executable
command or inspectable review record before declaring completion. Record any
missing validator as an open implementation need rather than inventing a pass.
Stop successfully only when the stated goal criteria pass. A supported no-go
may close an explicit decision objective; an execution blocker does not.

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

### Evidence Discovery (Not Acceptance)

```bash
rg --files GPU_Performance -g '*.csv' -g '*.json' -g '*.py' -g '*.ipynb'
rg -l -g '*.md' -g '*.py' "RTX|M1|M3|power|Power|AlexNet|VGG|ResNet|GoogLeNet|MobileNet|ViT" README.md CLAUDE.md GPU_Performance
```

### Acceptance Verifier And Stop Condition

- Validate a nonempty registry with a unique run identity, model, dataset,
  hardware/location, discrete power limit, batch/epoch count, synchronization,
  sampling policy, environment, and existing measured-data path for every row.
- Compare the registry against the declared benchmark Cartesian set; list
  missing combinations and distinguish main from exploratory runs explicitly.
- Reject missing required fields, missing raw inputs, and duplicate run keys.
  Record the registry path and validation receipt; complete only when all
  success criteria above are satisfied.

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

### Evidence Discovery (Not Acceptance)

```bash
rg -l -g '*.md' -g '*.py' "GBR|Gradient|Regression|TrainSpeed|PowerLimit|EnergySaving|prediction|MAE|RMSE|MAPE|R2" GPU_Performance README.md CLAUDE.md
```

### Acceptance Verifier And Stop Condition

- Freeze the metric definition, train/test or cross-validation split, and the
  measured source identifiers; ensure evaluation data do not enter training.
- Recompute each reported prediction-error value from stored predictions and
  measured targets with the declared units, aggregation, and tolerance.
- Reject missing/non-finite inputs, split overlap, and metric disagreement.
  Record the input/prediction/split paths and verifier outcome; curve or keyword
  presence alone does not complete this goal.

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

### Evidence Discovery (Not Acceptance)

```bash
rg --files Optimization -g '*.py' -g '*.ipynb' -g '*.csv' -g '*.json'
rg -l -g '*.md' -g '*.py' "Gurobi|MILP|battery|solar|grid|PVWatts|deadline|cost|carbon|baseline|heuristic" Optimization README.md CLAUDE.md
```

### Acceptance Verifier And Stop Condition

- Evaluate the declared baseline policies and optimized policy on the same
  measured power-performance inputs, workloads, time horizon, price/carbon
  traces, initial state, and completion/deadline requirements.
- Check power balance, discrete GPU states, charge/discharge efficiencies,
  battery state bounds, training progress, and required completion/deadlines.
- Recompute cost/carbon/utilization metrics and compare the admitted policies;
  retain infeasible cases with explicit status instead of silently dropping them.
- Reject a deliberately mismatched scenario or violated constraint in the
  chosen verifier. Record scenario, result, and verification paths before
  completing the goal; solver success alone does not establish these criteria.

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

### Evidence Discovery (Not Acceptance)

```bash
rg -l -g '*.md' -g '*.py' "schema|registry|environment|seed|version|command|sampling|nvidia-smi|powermetrics|CodeCarbon|csv|database" README.md CLAUDE.md AGENTS.md GPU_Performance Optimization
```

### Acceptance Verifier And Stop Condition

- Trace every admitted result through its raw input, configuration, environment,
  exact command, generator, and output identity; validate that referenced paths
  exist and distinguish raw from derived files.
- Reproduce an affected table/figure in the declared output location and compare
  it with the recorded result using an explicit equality/tolerance rule.
- Check raw-input hashes before/after to establish that analysis did not
  overwrite them. Existing sampling-contract tests can support collection
  invariants but do not replace end-to-end result provenance.
- Record each completed check and unresolved reproduction gap; an incomplete
  path/command chain leaves this goal partial.

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

### Evidence Discovery (Not Acceptance)

```bash
rg -l -g '*.md' -g '*.py' "publication|MSWiM|DOI|contribution|baseline|limitation|reproducible|open-source|paper|figure|table" README.md CLAUDE.md AGENTS.md GPU_Performance Optimization
```

### Acceptance Verifier And Stop Condition

- Review an explicit claim-to-evidence map for measurement, modeling, and
  scheduling contributions, including baseline, scope, metric, generator, and
  source artifact for every numerical claim.
- Verify that main claims use measured evidence, that relevant hardware and
  comparison limits remain next to the inference, and that exploratory work is
  visibly separated. Check release-plan paths and reproduction instructions.
- Record the inspected manuscript/map version and unresolved evidence markers.
  Close only when the declared packaging criteria are met; this review does not
  close an unfinished measurement, model-validation, or scheduling goal.
