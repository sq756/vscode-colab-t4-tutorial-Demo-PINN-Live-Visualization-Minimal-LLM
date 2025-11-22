# Tutorial / 教程

This guide walks through the two sections of `Dual_Demo_PINN_LLM_RAG.ipynb` with bilingual instructions.

本教程双语介绍 `Dual_Demo_PINN_LLM_RAG.ipynb` 的两大模块。

---

## 0. Setup & dependency check / 环境检查与依赖

1. Run the **setup cell** to print Python, CUDA, and GPU info using `torch` and `nvidia-smi` (if available).【F:Dual_Demo_PINN_LLM_RAG.ipynb†L98-L113】
2. Execute the **idempotent pip install cell** to fetch `transformers`, `accelerate`, `bitsandbytes`, `sentencepiece`, `gradio`, `faiss-cpu`, and `sentence-transformers`. You can safely re-run it on both Colab and VS Code.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L114-L135】

运行步骤：

1. 执行 **环境检查单元**，打印 Python/CUDA/GPU 信息（若有 GPU 则会调用 `nvidia-smi`）。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L98-L113】
2. 运行 **依赖安装单元**，获取 `transformers`、`accelerate`、`bitsandbytes`、`sentencepiece`、`gradio`、`faiss-cpu`、`sentence-transformers` 等包；重复执行也是安全的。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L114-L135】

---

## A. PINN live visualization / PINN 实时可视化

- **Goal**: train a small fully-connected PINN for the 2D Poisson equation with Dirichlet boundary conditions, and stream matplotlib plots of prediction, analytic solution, and PDE residuals.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L144-L279】
- **Key parameters**: `n_int`, `n_bc_per_edge` (sampling density), `width`/`depth` (network size), `refresh` (plot update frequency), `use_amp` (mixed precision on GPU).【F:Dual_Demo_PINN_LLM_RAG.ipynb†L144-L279】
- **Outputs**: live plots during training and checkpoints saved to `runs/pinn_ep*.pth`; final figures saved as `runs/pinn2d_live.pdf/svg`.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L228-L279】

操作提示：

1. 调整 `epochs`、`n_int`、`n_bc_per_edge` 等参数后运行 `run_pinn2d_poisson(...)`；脚本会自动在 GPU 上启用 AMP（若可用）。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L144-L223】
2. 训练期间会动态展示预测、解析解与残差热图，并定期保存检查点与最终图像到 `runs/` 目录。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L228-L279】

---

## B. Minimal LLM + RAG agent / 轻量级 LLM + RAG Agent

- **Model loading**: chooses TinyLlama 1.1B chat; if CUDA + `bitsandbytes` are available it loads 4-bit quantization, otherwise falls back to float16/float32 automatically.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L512-L551】
- **Embeddings & store**: uses `sentence-transformers/all-MiniLM-L6-v2` to build an in-memory vector store (`VectorDB`) for chunked text with cosine similarity search.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L553-L599】
- **Chat logic**: `rag_answer` augments prompts with top-k retrieved chunks and generates responses with temperature/max token controls.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L600-L652】
- **UI**: Gradio app with two tabs—Chat (message box + sliders) and Upload & Index (upload `.txt`, set chunk size/overlap, build index). Launch with `app.launch(share=True)`.【F:Dual_Demo_PINN_LLM_RAG.ipynb†L671-L708】

操作说明：

1. 直接运行 B 部分单元格完成模型加载和向量库初始化；如未安装 `bitsandbytes`，代码会自动提示并使用 float16/float32。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L512-L551】
2. 上传纯文本文件，在 “Upload & Index” 页签中调整 `chunk_size`/`overlap` 后点击 **Build Index**，即可建立检索向量。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L699-L705】
3. 在 “Chat” 页签输入问题，必要时调整温度与 `max_new_tokens`，对话会附带来源标记（Sources）。【F:Dual_Demo_PINN_LLM_RAG.ipynb†L671-L708】
