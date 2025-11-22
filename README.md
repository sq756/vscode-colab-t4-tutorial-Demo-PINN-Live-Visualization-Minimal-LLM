# Dual Demo: PINN Live Visualization + Minimal LLM (RAG/Agent)
# 双语指南：PINN 实时可视化与轻量级 LLM（RAG/Agent）

This repository hosts a single Jupyter notebook, **Dual_Demo_PINN_LLM_RAG.ipynb**, that combines two demos:

- **A. Physics-Informed Neural Network (PINN) live training** for a 2D Poisson equation with dynamic matplotlib visualizations.
- **B. Minimal LLM with retrieval-augmented generation (RAG) and an agent-like chat loop**, exposed through a simple Gradio UI.

本仓库仅包含一个 Jupyter 笔记本 **Dual_Demo_PINN_LLM_RAG.ipynb**，同时提供两个示例：

- **A. 基于物理约束神经网络（PINN）的 2D Poisson 方程实时训练**，附带动态 matplotlib 可视化。
- **B. 轻量级 LLM + 检索增强（RAG）的问答/Agent 流程**，通过 Gradio UI 体验。

## Quick start · 快速上手

1. Launch VS Code or Colab and open `Dual_Demo_PINN_LLM_RAG.ipynb`.
2. Follow the notebook’s suggested order: **0 → A → B**. Run the setup cell to print Python/CUDA info, then install dependencies (idempotent pip cell).【F:Dual_Demo_PINN_LLM_RAG.ipynb†L70-L135】
3. Proceed to the PINN section (A) to train and visualize; then move to the minimal LLM + RAG section (B) to launch the chat UI.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L144-L279】【F:Dual_Demo_PINN_LLM_RAG.ipynb†L499-L708】

推荐步骤：

1. 在 VS Code 或 Colab 中打开 `Dual_Demo_PINN_LLM_RAG.ipynb`。
2. 按笔记本提示的顺序 **0 → A → B** 依次运行：先查看环境信息，再执行依赖安装（幂等）。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L70-L135】
3. 进入 A 部分进行 PINN 训练与可视化，再切换到 B 部分启动轻量级 LLM + RAG 聊天界面。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L144-L279】【F:Dual_Demo_PINN_LLM_RAG.ipynb†L499-L708】

## Tutorials · 教程

For detailed, bilingual walkthroughs—including parameters, expected outputs, and how to start the Gradio app—see [`TUTORIAL.md`](TUTORIAL.md).

更详细的双语操作说明（参数解释、可视化产出、Gradio 启动方法）请查看 [`TUTORIAL.md`](TUTORIAL.md)。
