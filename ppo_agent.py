import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


def mlp(in_dim, out_dim, hidden=(128, 128), act=nn.Tanh):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu = self.net(obs)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v = mlp(obs_dim, 1)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 6
    seed: int = 42


class PPO:
    def __init__(self, obs_dim, act_dim, cfg: PPOConfig):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        self.pi_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.v_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.cfg = cfg

    def policy(self, obs):
        mu, std = self.actor(obs)
        return torch.distributions.Normal(mu, std)

    def value(self, obs):
        return self.critic(obs)

    def update(self, buf):
        obs = torch.as_tensor(buf['obs'], dtype=torch.float32)
        act = torch.as_tensor(buf['act'], dtype=torch.float32)
        adv = torch.as_tensor(buf['adv'], dtype=torch.float32)
        ret = torch.as_tensor(buf['ret'], dtype=torch.float32)
        logp_old = torch.as_tensor(buf['logp'], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.cfg.update_epochs):
            dist = self.policy(obs)
            logp = dist.log_prob(act).sum(-1)
            ratio = torch.exp(logp - logp_old)
            pi_loss = -(torch.min(ratio * adv,
                                  torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv)).mean()
            v = self.value(obs)
            v_loss = ((v - ret) ** 2).mean()
            self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
            self.v_opt.zero_grad(); v_loss.backward(); self.v_opt.step()