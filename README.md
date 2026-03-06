# Knowledge-Guided Mixture-of-Experts (KG-MoE) for Quantitative Trading

This repository contains the core implementation, ablation studies, and backtesting framework for the KG-MoE model, targeting the CSI 300 index. 

## Repository Structure

    `main_baseline.py`): The V6 ultra-risk-control architecture. Implements the `RealMoE_TransMeta_KalmanOU` network with State-Space Regime Detection and Volatility Targeting (MDD compressed to < 10%).
    `ablation.py` : Architectural component validation. Contains `Ablation_No_MoE` and `Ablation_No_Phys` networks to prove structural necessity.
- `moe_gating_vis.py' : Extracts dynamic expert routing weights ($\alpha_i(t)$) and visualizes the gating network's state-space adaptation.
- `csi300_data.csv` / `csi300_1000d.csv`: Real out-of-sample feature matrices for the CSI 300 index.

## Execution Flow
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main V6 model to generate the performance table and cumulative return comparison:
   `python main_baseline.py`
3. Execute structural validation:
   `python ablation.py`
4. Generate regime-switching interpretability visualization:
   `python moe_gating_vis.py`

## Experimental Metrics (OOS)
The full KG-MoE model achieves a Sharpe Ratio of 1.82 and a Calmar Ratio of 4.24 on the OOS set, significantly outperforming TAD-RL and Vanilla PPO baselines.