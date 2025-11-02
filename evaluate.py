import torch, matplotlib.pyplot as plt, pandas as pd
from data_interface import load_prices, get_cfg
from trading_env import PortfolioEnv
from ppo_agent import Actor, Critic
import numpy as np
import pandas as pd

def equal_weight_pv(prices, init):
    rets = prices.pct_change().fillna(0)
    ew = (rets.mean(axis=1) + 1).cumprod()
    return pd.Series(init * ew.values, index=prices.index)

def index_pv(series, init):
    return pd.Series(init * (series / series.iloc[0]), index=series.index)

def main():
    prices, nifty = load_prices(); cfg = get_cfg()
    env = PortfolioEnv(prices, cfg["LOOKBACK_WINDOW"],
                       cfg["TRANSACTION_FEE_PERCENT"], cfg["INITIAL_BALANCE"],
                       train=False, test_split_days=cfg["TEST_SPLIT_DAYS"])
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    actor, critic = Actor(obs_dim, act_dim), Critic(obs_dim)
    actor.load_state_dict(torch.load("actor.pth", map_location="cpu"))
    actor.eval()

    obs, _ = env.reset(); pv, dates = [], []
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mu, std = actor(obs_t); act = mu.squeeze(0).detach().numpy()
        obs, rew, term, trunc, info = env.step(act)
        pv.append(info["portfolio_value"]); dates.append(pd.to_datetime(info["date"]))
        if term or trunc: break

    test_prices = prices.iloc[-cfg["TEST_SPLIT_DAYS"]:]
    test_nifty = nifty.reindex(test_prices.index).ffill()
    rl = pd.Series(pv, index=dates)
    ix = index_pv(test_nifty.loc[rl.index], cfg["INITIAL_BALANCE"])

    plt.plot(rl, label="RL (PPO)")
    plt.plot(ix, label="NIFTY-50")
    plt.title("Portfolio Value Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("backtest.png", dpi=150)
    print("Saved backtest.png")
    rl_ret = rl.pct_change().dropna().values
    sharpe_rl = (rl_ret.mean() / (rl_ret.std() + 1e-9)) * np.sqrt(252)
    print(f"Sharpe (RL, annualized): {sharpe_rl:.3f}")
    ix_ret = ix.pct_change().dropna().values
    sharpe_ix = (ix_ret.mean() / (ix_ret.std() + 1e-9)) * np.sqrt(252)
    print(f"Sharpe (NIFTY-50, annualized): {sharpe_ix:.3f}")
    rl_ret = rl.pct_change().dropna()
    A = 252.0

    sharpe_rl  = (rl_ret.mean()*A) / (rl_ret.std(ddof=1)*np.sqrt(A) + 1e-12)
    sortino_rl = (rl_ret.mean()*A) / (rl_ret[rl_ret<0].std(ddof=1)*np.sqrt(A) + 1e-12)
    rl_dd = rl/rl.cummax() - 1.0
    maxdd_rl = float(rl_dd.min())
    years_rl = (rl.index[-1] - rl.index[0]).days/365.25
    cagr_rl  = (rl.iloc[-1]/rl.iloc[0])**(1/max(years_rl,1e-9)) - 1.0 if years_rl>0 else np.nan
    vol_rl   = rl_ret.std(ddof=1)*np.sqrt(A)
    calmar_rl = (cagr_rl / abs(maxdd_rl)) if maxdd_rl<0 else np.nan
    hitrate_rl = float((rl_ret > 0).sum()/max(len(rl_ret),1))

    # RL metrics row
    rl_row = {
        "Metric": "Your RL (PPO)",
        "Sharpe": round(sharpe_rl, 3),
        "Sortino": round(sortino_rl, 3),
        "MaxDD %": round(100*maxdd_rl, 2),
        "CAGR %": round(100*cagr_rl, 2),
        "Vol (ann %)": round(100*vol_rl, 2),
        "Calmar": round(calmar_rl, 3),
        "Hit-rate %": round(100*hitrate_rl, 2),
    }

    # Benchmark interpretation rows
    benchmarks = [
        {"Metric": "Weak",        "Sharpe": "<1.0",   "Sortino": "<1.0", "MaxDD %": "<-40%", "CAGR %": "<5%",  "Vol (ann %)": ">30%", "Calmar": "<0.5", "Hit-rate %": "<48%"},
        {"Metric": "Good",        "Sharpe": "1.0–1.5","Sortino": "1.0–1.5","MaxDD %": "-30%–-20%", "CAGR %": "5–12%", "Vol (ann %)": "20–30%", "Calmar": "0.5–1.0", "Hit-rate %": "48–55%"},
        {"Metric": "Excellent",   "Sharpe": "1.5–2.0","Sortino": "1.5–2.0","MaxDD %": "-20%–-10%", "CAGR %": "12–20%", "Vol (ann %)": "12–20%", "Calmar": "1.0–2.0", "Hit-rate %": "55–60%"},
        {"Metric": "Exceptional", "Sharpe": ">2.0",   "Sortino": ">2.0", "MaxDD %": ">-10%", "CAGR %": ">20%", "Vol (ann %)": "<12%", "Calmar": ">2.0", "Hit-rate %": ">60%"},
    ]

    # Combine into one file
    combined_df = pd.DataFrame([rl_row])._append(pd.DataFrame(benchmarks), ignore_index=True)
    combined_df.to_csv("metrics_comparison.csv", index=False)

    print("\nSaved metrics_comparison.csv (RL metrics + benchmark table together)")
# Sharpe ratio (annualized)
    

if __name__ == "__main__":
    main()