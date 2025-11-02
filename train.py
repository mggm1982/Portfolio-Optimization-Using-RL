import torch
from data_interface import load_prices, get_cfg
from trading_env import PortfolioEnv
from ppo_agent import PPO, PPOConfig
from utils import set_seed, GAEBuffer

def main():
    prices, _ = load_prices()
    cfg = get_cfg()
    env = PortfolioEnv(prices, cfg["LOOKBACK_WINDOW"],
                       cfg["TRANSACTION_FEE_PERCENT"], cfg["INITIAL_BALANCE"],
                       train=True, test_split_days=cfg["TEST_SPLIT_DAYS"])
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = PPO(obs_dim, act_dim, PPOConfig())
    set_seed(42)

    obs, _ = env.reset(); buf = GAEBuffer(0.99, 0.95)
    steps, total_steps = 0, int(cfg["TOTAL_TIMESTEPS"])
    while steps < total_steps:
        done = False
        while not done and steps < total_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = agent.policy(obs_t)
            act = dist.sample(); logp = dist.log_prob(act).sum(-1).item()
            act_np = act.squeeze(0).detach().numpy()
            next_obs, rew, term, trunc, info = env.step(act_np)
            val = agent.value(obs_t).item()
            buf.store(obs=obs, act=act_np, rew=rew, val=val, logp=logp)
            obs = next_obs; steps += 1; done = term or trunc
        data = buf.finish(); agent.update(data); buf = GAEBuffer(0.99, 0.95); obs, _ = env.reset()
        print(f"Episode done | steps={steps} | PV={info.get('portfolio_value', 0):.2f}")

    torch.save(agent.actor.state_dict(), "actor.pth")
    torch.save(agent.critic.state_dict(), "critic.pth")
    print("Saved actor.pth and critic.pth")

if __name__ == "__main__":
    main()