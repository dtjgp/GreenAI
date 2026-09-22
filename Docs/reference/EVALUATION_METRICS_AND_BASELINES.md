# GreenAI Evaluation Metrics and Baselines

Moved from `AGENTS.md` on 2026-09-22. Read this when designing experiments,
choosing metrics or baselines, or packaging paper claims (GOALS.md Goal 5).
Claim boundaries and reporting rules remain in `AGENTS.md`.

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
