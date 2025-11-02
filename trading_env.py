import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """Simple multi-asset allocation environment."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices: pd.DataFrame, window: int = 60,
                 fee_rate: float = 0.001, initial_balance: float = 1_000_000,
                 train: bool = True, test_split_days: int = 500):
        super().__init__()
        prices = prices.sort_index()
        if test_split_days > 0 and test_split_days < len(prices.index):
            split_idx = -int(test_split_days)
        else:
            split_idx = len(prices.index)
        self.prices_train = prices.iloc[:split_idx] if train else prices.iloc[split_idx:]
        self.returns_full = prices.pct_change().fillna(0.0)
        self.returns = self.returns_full.loc[self.prices_train.index].values

        self.dates = self.prices_train.index
        self.T, self.N = self.returns.shape
        self.window = int(window)
        self.fee = float(fee_rate)
        self.initial_balance = float(initial_balance)

        obs_dim = self.N * self.window + self.N
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-6.0, high=6.0, shape=(self.N,), dtype=np.float32)

        self._t = None
        self._w = None
        self._pv = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
        self._w = np.ones(self.N, dtype=np.float32) / self.N
        self._pv = self.initial_balance
        return self._obs(), {}

    def step(self, action):
        logits = np.clip(np.nan_to_num(action, nan=0.0), -6, 6)
        ex = np.exp(logits - logits.max())
        w_new = ex / ex.sum()
        turnover = np.abs(w_new - self._w).sum()
        cost = self.fee * turnover

        r_vec = self.returns[self._t]
        port_ret = float((w_new * r_vec).sum() - cost)
        reward = float(np.log1p(port_ret))
        self._pv *= (1.0 + port_ret)
        self._w = w_new
        self._t += 1

        terminated = self._t >= self.T - 1
        info = {"date": str(self.dates[self._t - 1]), "portfolio_value": self._pv}
        return self._obs(), reward, terminated, False, info

    def _obs(self):
        win = self.returns[self._t - self.window:self._t]
        return np.concatenate([win.flatten(), self._w]).astype(np.float32)