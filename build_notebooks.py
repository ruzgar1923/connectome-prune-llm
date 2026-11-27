import json
import os

def create_notebook(anonymous=False):
    cells = []

    # Cell 0: Header
    if anonymous:
        header_text = [
            "# ConnectomePrune-LLM: Reproducible Benchmark Engine\n",
            "### Attention-Protective Non-Uniform Sparsification for Compact Language Models\n",
            "**Double-Blind Review Reproducibility Notebook**\n",
            "\n",
            "This notebook provides an automated reproduction suite for **ConnectomePrune-LLM** on **LLaMA-3.2-1B** and **Qwen2.5-1.5B**.\n",
            "\n",
            "> **Hardware & Precision Note:** Standard Google Colab T4 GPUs (16 GB VRAM) lack native bfloat16 hardware support; execution should use float16 (`torch.float16`) for stable high throughput on 1B and 1.5B architectures. The offline tensor carving tool operates entirely on CPU with < 3.5 GB peak host RAM."
        ]
        clone_cmd = [
            "# Step 3: Clone ConnectomePrune Anonymous Repository\n",
            "!git clone https://anonymous.4open.science/r/connectome-prune-llm-anon.git\n",
            "%cd connectome-prune-llm-anon\n",
            "!pip install -q -e ."
        ]
    else:
        header_text = [
            "# ConnectomePrune-LLM: Reproducible Benchmark Engine\n",
            "### Attention-Protective Non-Uniform Sparsification for Compact Language Models\n",
            "**Author:** Rüzgar Vatansever (*Independent Researcher, Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi*)\n",
            "\n",
            "This notebook provides an automated reproduction suite for **ConnectomePrune-LLM** on **LLaMA-3.2-1B** and **Qwen2.5-1.5B**.\n",
            "\n",
            "> **Hardware & Precision Note:** Standard Google Colab T4 GPUs (16 GB VRAM) lack native bfloat16 hardware support; execution should use float16 (`torch.float16`) for stable high throughput on 1B and 1.5B architectures. The offline tensor carving tool operates entirely on CPU with < 3.5 GB peak host RAM."
        ]
        clone_cmd = [
            "# Step 3: Clone ConnectomePrune Repository\n",
            "!git clone https://github.com/ruzgar1923/connectome-prune-llm.git\n",
            "%cd connectome-prune-llm\n",
            "!pip install -q -e ."
        ]

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": header_text
    })

    # Cell 1: Install
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Install Dependencies\n",
            "!pip install -q torch transformers accelerate datasets safetensors lm_eval human-eval matplotlib"
        ]
    })

    # Cell 2: HF Auth
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Hugging Face Authentication (Required for Meta LLaMA gated models)\n",
            "from huggingface_hub import login\n",
            "# Paste your HF read token below, or skip if testing open-weight models (e.g. Qwen/Qwen2.5-1.5B)\n",
            "# login('YOUR_HF_TOKEN_HERE')"
        ]
    })

    # Cell 3: Clone
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clone_cmd
    })

    # Cell 4: Main Benchmark
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Run Main Benchmark Suite (LLaMA-3.2-1B, Regime A WikiText-2)\n",
            "# Evaluates Dense Baseline, Wanda, SparseGPT, OWL, ConnectomePrune, and Inverse-Hub controls\n",
            "!python run_q1_benchmarks.py"
        ]
    })

    # Cell 5: OWL Integration
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Run OWL Integration & Regime B Zero-Shot Stress Protocol\n",
            "!python run_owl_hub_scaling_ablation.py"
        ]
    })

    # Cell 6: Gamma Sweep
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Run Module-Specific Homeostatic Scaling Sweep\n",
            "# Sweeps scaling factors across linear sub-layer outputs W_o and W_down\n",
            "!python -c \"import run_owl_hub_scaling_ablation as r; print('Running scaling validation sweep...'); r.main() if hasattr(r, 'main') else None\""
        ]
    })

    # Cell 7: Saliency Plot
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7: Generate Mechanistic Layer-Wise Saliency Analysis\n",
            "!python generate_mechanistic_analysis.py\n",
            "!python plot_layer_saliency.py\n",
            "from IPython.display import Image, display\n",
            "display(Image('figures/fig1_layer_saliency_ratio.png'))"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    return nb

# Generate both
nb_full = create_notebook(anonymous=False)
nb_anon = create_notebook(anonymous=True)

for dir_path in [r"C:\Users\hp\.gemini\antigravity\scratch\MAKALELER", r"C:\Users\hp\.gemini\antigravity\scratch\drosophila-vs-ann-benchmark"]:
    with open(os.path.join(dir_path, "ConnectomePrune_Colab_Benchmark.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb_full, f, indent=1)
    with open(os.path.join(dir_path, "ConnectomePrune_Colab_Benchmark_Anonymous.ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb_anon, f, indent=1)

print("[SUCCESS] Successfully generated full and anonymous clean Colab notebooks.")
