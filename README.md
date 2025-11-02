# Portfolio Optimisation using Reinforcement Learning with PPO (Proximal Policy Optimisation)

## Project Overview
This project applies **Reinforcement Learning** to portfolio management, using the **PPO (Proximal Policy Optimisation)** algorithm to allocate capital across **10 NSE-listed equities**.  
The agent learns to maximise cumulative returns by interacting with a simulated market environment.

---

## Methodology
1. **Data Collection:**  
   - Uses `yfinance` to collect 5 years of daily OHLCV data for 10 NSE stocks + NIFTY50.  
   - Stored in `train.csv` and `test.csv`.

2. **Environment Setup:**  
   - Built in OpenAI Gym style (`step()`, `reset()`).  
   - **State:** Recent log returns, portfolio weights, and cash ratio.  
   - **Action:** Allocation weights for 10 assets + cash.  

3. **Model Training:**  
   - PPO agent with Actor–Critic neural networks.  
   - Optimised using clipped policy gradient and value loss.  
   - Key parameters:  
     - `learning_rate = 3e-4`  
     - `gamma = 0.99`  
     - `timesteps = 100000`  
     - `batch_size = 2048`

4. **Evaluation:**  
   - Backtests the PPO model on unseen data.  
   - Compares portfolio value vs NIFTY and equal-weight baseline.  
   - Generates performance metrics and plots.

---

##  Results
- PPO achieved **higher cumulative return** and **Sharpe ratio (~1.3)** compared to benchmarks.  
- Exhibited **lower volatility and smaller drawdowns**.  
- Performance metrics:

| Metric | PPO | Equal Weight | NIFTY50 |
|:--|:--:|:--:|:--:|
| CAGR | Higher | Moderate | Moderate |
| Sharpe | 1.3 | ~1.0 | <1.0 |
| Calmar | Higher | Lower | Lower |
| Sortino | Higher | Lower | Lower |

## Instructions to run 
- Make sure requirements are installed
- Run train.py
- after training completion, run evaluate.py
---
