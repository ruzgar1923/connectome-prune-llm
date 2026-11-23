# 🪰 ConnectomeAI: Whole-Brain *Drosophila* Connectome vs. Scale-Matched ANNs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24375969.svg)](https://doi.org/10.5281/zenodo.24375969)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Google Scholar](https://img.shields.io/badge/Google-Scholar-blue.svg)](https://scholar.google.com/citations?user=iqSiuPwAAAAJ)

> **Official Open-Source Repository** for the research paper:  
> *"Biomimetic Sparsity vs. Dense Computation: Benchmarking the Whole-Brain Drosophila Connectome Against Scale-Matched Artificial Neural Networks Under Strict Resource Constraints"*  
> **Author:** Rüzgar Vatansever (*Şehit Öğretmen Necmeddin Kuyucu Anadolu Lisesi, Turkey*)  
> **Correspondence:** `rzvatansevertr@gmail.com`

---

## 📌 Executive Summary

While contemporary deep learning models require gigawatts of computational infrastructure, biological insect nervous systems solve real-time aerodynamic navigation, turbulent odor tracking, and obstacle avoidance within sub-milliwatt power budgets.

Leveraging the recent whole-brain connectome of the adult fruit fly (*Drosophila melanogaster*) resolved by the **FlyWire consortium** (139,255 neurons, ~54.5M synapses; *Nature* 2024), this repository introduces a standardized, reproducible benchmark directly comparing an event-driven **Bio-mimetic Spiking Neural Network (Bio-SNN)** against an identically sized **Dense Artificial Neural Network (Dense-ANN)** in a turbulent advection-diffusion wind tunnel arena.

### 📊 Key Empirical Findings

| Metric | Bio-SNN (Connectome) | Dense ANN Baseline | Quantitative Advantage |
| :--- | :--- | :--- | :--- |
| **Arithmetic Operations / Episode** | **565,244 SynOps** | 1,129,180,416 FLOPs | **1,997.7× Fewer Ops** |
| **Thermodynamic Energy (7nm ASIC)** | **0.509 μJ** | 5,194.230 μJ | **10,210.4× Energy Reduction** |
| **Unablated Final Distance** | **113.2 units** | 126.9 units | **1.12× Closer to Source** |
| **Fault Tolerance (20% Ablation)** | **145.0 units** | 157.6 units | **Superior Structural Resilience** |

---

## 🗺️ Visual Results

### 1. Operations and Energy Disparity
![Compute & Energy Benchmark](results/fig1_compute_energy_benchmark.png)

### 2. Topological Fault Tolerance Under Node Ablation
![Ablation Resilience](results/fig2_ablation_resilience.png)

### 3. Closed-Loop Chemotaxis Navigation in Turbulent Plumes
![Chemotaxis Trajectories](results/fig3_chemotaxis_trajectories.png)

---

## ⚡ Quickstart & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/RuzgarVatansever/drosophila-connectome-benchmark.git
cd drosophila-connectome-benchmark
pip install -r requirements.txt
```

### 1. Run the Interactive Web Dashboard (Streamlit)
Launch the real-time interactive visualization arena with live telemetry meters:
```bash
streamlit run app.py
```

### 2. Run the Full Headless Benchmark
Execute 25 navigational trials, ablation experiments, and generate high-resolution 300 DPI publication plots:
```bash
python benchmark_simulation.py
```

### 3. Use ConnectomeAI in Your Own Models
```python
from connectome_ai import ConnectomeNetwork

# Create an insect-inspired neuromorphic agent
agent = ConnectomeNetwork(
    n_in=6,          # 6 sensory inputs (bilateral antennae + rangefinder)
    n_hidden=1200,   # Hidden connectome size
    n_out=2,         # Differential steering motor output
    sparsity=0.985   # 98.5% biological sparsity
)

# Step inference
sensory_input = [1.2, 0.4, 0.8, 0.1, 0.9, -0.4]
motor_command = agent.forward_step(sensory_input)
print(f"Motor Command: {motor_command} | Synaptic Operations: {agent.total_synops}")
```

---

## 📄 Citation

If you use this benchmark, architecture, or empirical data in your research, please cite our paper:

```bibtex
@article{vatansever2026biomimetic,
  title={Biomimetic Sparsity vs. Dense Computation: Benchmarking the Whole-Brain Drosophila Connectome Against Scale-Matched Artificial Neural Networks Under Strict Resource Constraints},
  author={Vatansever, R{\"u}zgar},
  journal={Zenodo Preprint},
  year={2026},
  doi={10.5281/zenodo.24375969},
  url={https://doi.org/10.5281/zenodo.24375969}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
