# CLAUDE.md - AI Assistant Configuration for GreenAI Project

**Last Updated**: 2026-04-10
**Project**: GreenAI - Measurement-Grounded GPU Training Energy Characterization and Hybrid Energy Scheduling

This file provides working guidance for AI assistants when operating in this repository.

---

## LLM Wiki 集成

本项目连接到统一的 `llm-wiki` 知识库，用于研究辅助、知识校对、baseline 检索与论文写作支撑。

**Wiki 路径**: `/Users/dtjgp/Library/CloudStorage/GoogleDrive-dtjgp92613@gmail.com/My Drive/Obsidian/llm-wiki`

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
