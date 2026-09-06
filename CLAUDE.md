@AGENTS.md

# GreenAI Claude Adapter

Use `AGENTS.md` as the shared project charter, evidence hierarchy, measurement
failure policy, manuscript rules, and wiki routing contract. Do not duplicate
those rules or maintain a second current status/backlog here.

For substantial work, read the relevant goal in `GOALS.md`, then inspect its
live inputs and nearest scripts. `README.md` and measured artifacts determine
current project facts; instructions, a directory listing, or a keyword hit do
not establish completed measurement, modeling, or scheduling evidence.

## Repository Navigation Context

The following inventory comes from the 2026-04-10 project note. Confirm current
coverage and parameters from code/data before using them in a result or paper.

- Measurement areas include local RTX 3060/4070 and Apple M1/M3 work and cloud
  RTX 3080/4090 work under `GPU_Performance/`; their protocols and physical
  energy boundaries require the comparison checks in `AGENTS.md`.
- Workloads include AlexNet, VGG, ResNet, GoogLeNet variants, MobileNet, and
  selected ViT studies. Stage labels include `to_device`, `forward`, `loss`,
  `backward`, and `optimize`, with representative layer-level profiling.
- Modeling areas include GBR (Gradient Boosting Regression) and architecture
  descriptors linked to time, energy, and efficiency.
- `Optimization/` contains MILP/Gurobi and PVWatts-related work for discrete GPU
  states and solar/grid/battery dispatch. The earlier note mentioned battery
  efficiency around 0.9 and optional grid sell-back; read the actual scenario
  configuration instead of inheriting these as fixed defaults.
- Historical publication pointer: *Energy Sustainability Analysis of Deep
  Neural Network*, MSWiM 2025, DOI `10.1109/MSWiM67937.2025.11309062`.
  Verify the repository publication record and primary source before citation.
- Earlier candidate venues included ACM eEnergy, IEEE INFOCOM workshops, and
  sustainable-computing/green-AI venues. This list does not freeze a submission
  target, deadline, acceptance, or current venue eligibility.

Benchmark consolidation, power-performance validation, scheduling baselines,
reproducibility, and paper packaging are maintained in `GOALS.md`. Preserve
its open criteria; do not infer completion from this historical inventory.
