"""
Makale 1: Comprehensive Neuromorphic vs ANN Energy Calculator
=============================================================
Calculates exact energy dissipation under standardized 45 nm CMOS arithmetic costs (Horowitz, 2014)
across 400 simulation steps:
- E_MAC = 4.6 pJ (32-bit float multiply-accumulate)
- E_AC  = 0.9 pJ (32-bit addition / spike accumulation)
"""

def main():
    steps = 400
    N = 1200
    in_dim = 6
    out_dim = 2
    sparsity = 0.985

    # 1. Dense ANN Baseline
    # Recurrent: 1200 * 1200 = 1,440,000 MACs
    # Input: 6 * 1200 = 7,200 MACs
    # Output: 1200 * 2 = 2,400 MACs
    # Total MACs per step = 1,449,600 MACs
    dense_macs_step = (in_dim * N) + (N * N) + (N * out_dim)
    dense_macs_episode = dense_macs_step * steps
    dense_energy_uj = dense_macs_episode * 4.6e-12 * 1e6 # 2667 uJ (~2.67 mJ)

    # 2. Sparse ANN Control (98.5% sparsity on recurrent matrix)
    sparse_macs_step = (in_dim * N) + int(N * N * (1 - sparsity)) + (N * out_dim)
    sparse_macs_episode = sparse_macs_step * steps
    sparse_energy_uj = sparse_macs_episode * 4.6e-12 * 1e6 # ~57.4 uJ

    # If input and output are also sparse:
    sparse_all_macs_step = int(dense_macs_step * (1 - sparsity))
    sparse_all_macs_episode = sparse_all_macs_step * steps
    sparse_all_energy_uj = sparse_all_macs_episode * 4.6e-12 * 1e6 # ~40.0 uJ

    # 3. Bio-SNN Components
    # Synaptic Operations (SynOps) measured empirically across 400 steps:
    synops = 565244
    synop_energy_uj = synops * 0.9e-12 * 1e6 # 0.509 uJ

    # Membrane leak updates (beta * V) for N neurons over 400 steps:
    leak_updates = N * steps # 480,000 updates

    # Input projections (in_dim * N = 7,200 projections per step * 400 steps = 2,880,000)
    input_ops = in_dim * N * steps

    # Scenarios:
    # Scenario A: Conservative (Leak and Input computed as full 32-bit MACs: 4.6 pJ)
    snn_energy_scenario_a = synop_energy_uj + (leak_updates * 4.6e-12 * 1e6) + (input_ops * 4.6e-12 * 1e6)

    # Scenario B: Neuromorphic Spike-Coded (Leak is MAC 4.6 pJ, Input is spike-coded AC 0.9 pJ)
    snn_energy_scenario_b = synop_energy_uj + (leak_updates * 4.6e-12 * 1e6) + (input_ops * 0.9e-12 * 1e6)

    # Scenario C: Dedicated Neuromorphic Hardware (Loihi / SpiNNaker / Analog: Bit-shift decay 0.9 pJ + Spike Input 0.9 pJ)
    snn_energy_scenario_c = synop_energy_uj + (leak_updates * 0.9e-12 * 1e6) + (input_ops * 0.9e-12 * 1e6)

    print("=" * 75)
    print("COMPREHENSIVE MULTI-SCENARIO ENERGY BENCHMARK (400 STEPS)")
    print("=" * 75)
    print(f"Dense ANN Baseline:               {dense_energy_uj:.2f} uJ ({dense_energy_uj/1000:.2f} mJ)")
    print(f"Sparse ANN Control (Uniform):      {sparse_all_energy_uj:.2f} uJ")
    print(f"Bio-SNN Synaptic Transmission:    {synop_energy_uj:.4f} uJ")
    print("-" * 75)
    print(f"Scenario A (Conservative FP32):   {snn_energy_scenario_a:.2f} uJ | vs Sparse: {sparse_all_energy_uj/snn_energy_scenario_a:.1f}x | vs Dense: {dense_energy_uj/snn_energy_scenario_a:.0f}x")
    print(f"Scenario B (Spike-Coded Input):   {snn_energy_scenario_b:.2f} uJ  | vs Sparse: {sparse_all_energy_uj/snn_energy_scenario_b:.1f}x | vs Dense: {dense_energy_uj/snn_energy_scenario_b:.0f}x")
    print(f"Scenario C (Hardware Bit-Shift):  {snn_energy_scenario_c:.2f} uJ  | vs Sparse: {sparse_all_energy_uj/snn_energy_scenario_c:.1f}x | vs Dense: {dense_energy_uj/snn_energy_scenario_c:.0f}x")
    print("=" * 75)

if __name__ == "__main__":
    main()
