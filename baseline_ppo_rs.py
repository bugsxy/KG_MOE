"""
Baseline 3: PPO-RS (PPO + Risk Shaping) for CSI 300 Trading
- 在标准 PPO 基础上加入 drawdown 惩罚 + 波动率惩罚（软约束风控）
- 与 Deep KG-MoE 的硬约束 V6-CRC 形成直接对比
- 使用相同数据和划分
- 输出 CR, MDD, Sharpe, Calmar, Sortino
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
CSV_PATH = "csi300_1000d.csv"
LOOKBACK = 15
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
TRANSACTION_COST = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 超参
EPISODES = 200
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH = 64

# Risk Shaping 参数（软约束）
DD_PENALTY = 5.0        # drawdown 惩罚系数
VOL_PENALTY = 2.0       # 波动率惩罚系数
DD_THRESHOLD = 0.07     # 7% drawdown 开始惩罚


# ====================== 1. 数据加载 ======================
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(20).std()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'return', 'log_return', 'volatility', 'ma5', 'ma20', 'rsi']
    for col in feature_cols:
        rolling_mean = df[col].rolling(min(252, len(df) // 2), min_periods=20).mean()
        rolling_std = df[col].rolling(min(252, len(df) // 2), min_periods=20).std()
        df[col + '_norm'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

    df = df.dropna().reset_index(drop=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def split_data(df):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ====================== 2. 交易环境（带 Risk Shaping Reward）======================
class TradingEnvRS:
    def __init__(self, df, lookback=15):
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.feature_cols = [c for c in df.columns if c.endswith('_norm')]
        self.n_features = len(self.feature_cols)
        self.reset()

    def reset(self):
        self.step_idx = self.lookback
        self.position = 0.0
        self.nav = 1.0
        self.nav_peak = 1.0
        self.nav_history = [1.0]
        self.return_history = []
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        start = self.step_idx - self.lookback
        end = self.step_idx
        window = self.df[self.feature_cols].iloc[start:end].values
        obs = window.flatten().astype(np.float32)
        dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        extra = np.array([self.position, dd], dtype=np.float32)
        return np.concatenate([obs, extra])

    @property
    def obs_dim(self):
        return self.lookback * self.n_features + 2

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # 交易成本
        trade_size = abs(action - self.position)
        cost = trade_size * TRANSACTION_COST

        # 收益
        ret = self.df['return'].iloc[self.step_idx]
        pnl = self.position * ret - cost
        self.nav *= (1 + pnl)

        self.position = action
        self.nav_peak = max(self.nav_peak, self.nav)
        self.nav_history.append(self.nav)
        self.return_history.append(pnl)
        self.step_idx += 1

        if self.step_idx >= len(self.df) - 1:
            self.done = True

        # ========== Risk-Shaped Reward ==========
        # 基础 reward: differential Sharpe ratio 近似
        if len(self.return_history) > 2:
            recent = np.array(self.return_history[-min(20, len(self.return_history)):])
            if np.std(recent) > 1e-8:
                base_reward = np.mean(recent) / (np.std(recent) + 1e-8)
            else:
                base_reward = pnl * 10
        else:
            base_reward = pnl * 10

        # 惩罚 1: Drawdown 惩罚（超过 7% 开始惩罚）
        current_dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        if current_dd > DD_THRESHOLD:
            dd_penalty = -DD_PENALTY * (current_dd - DD_THRESHOLD)
        else:
            dd_penalty = 0.0

        # 惩罚 2: 波动率惩罚（最近收益波动太大则惩罚）
        if len(self.return_history) > 5:
            recent_vol = np.std(self.return_history[-20:])
            vol_penalty = -VOL_PENALTY * max(recent_vol - 0.02, 0)
        else:
            vol_penalty = 0.0

        reward = base_reward + dd_penalty + vol_penalty

        return self._get_obs(), reward, self.done


# ====================== 3. PPO 网络 ======================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        # Actor
        self.actor_fc1 = nn.Linear(obs_dim, hidden)
        self.actor_fc2 = nn.Linear(hidden, hidden)
        self.actor_mean = nn.Linear(hidden, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic
        self.critic_fc1 = nn.Linear(obs_dim, hidden)
        self.critic_fc2 = nn.Linear(hidden, hidden)
        self.critic_out = nn.Linear(hidden, 1)

    def actor(self, obs):
        x = F.relu(self.actor_fc1(obs))
        x = F.relu(self.actor_fc2(x))
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp().expand_as(mean)
        return mean, std

    def critic(self, obs):
        x = F.relu(self.critic_fc1(obs))
        x = F.relu(self.critic_fc2(x))
        return self.critic_out(x)

    def get_action(self, obs, eval_mode=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        mean, std = self.actor(obs_t)
        if eval_mode:
            action = torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            z = dist.sample()
            action = torch.tanh(z)
        return action.cpu().detach().numpy().flatten()[0]

    def evaluate(self, obs_batch, action_batch):
        mean, std = self.actor(obs_batch)
        dist = Normal(mean, std)
        # 反 tanh 来算 log_prob
        z = torch.atanh(action_batch.clamp(-0.999, 0.999))
        log_prob = dist.log_prob(z) - torch.log(1 - action_batch.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        value = self.critic(obs_batch)
        return log_prob, value, entropy


# ====================== 4. PPO Trajectory Buffer ======================
class TrajectoryBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def push(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, last_value):
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_batches(self, last_value):
        advantages, returns = self.compute_gae(last_value)

        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)

        # 归一化 advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.FloatTensor(states).to(DEVICE),
            torch.FloatTensor(actions).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(old_log_probs).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(returns).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(advantages).unsqueeze(1).to(DEVICE),
        )

    def clear(self):
        self.__init__()


# ====================== 5. PPO 训练 ======================
def train_ppo_rs(model, optimizer, train_df):
    env = TradingEnvRS(train_df, lookback=LOOKBACK)
    best_nav = 0

    for episode in range(EPISODES):
        state = env.reset()
        buffer = TrajectoryBuffer()
        total_reward = 0

        while not env.done:
            obs_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

            # 获取动作和 value
            mean, std = model.actor(obs_t)
            dist = Normal(mean, std)
            z = dist.sample()
            action = torch.tanh(z)
            log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum().item()
            value = model.critic(obs_t).item()

            action_np = action.cpu().detach().numpy().flatten()[0]
            next_state, reward, done = env.step(action_np)

            buffer.push(state, action_np, reward, float(done), log_prob, value)
            state = next_state
            total_reward += reward

        # 计算最后一个 value
        last_obs = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        last_value = model.critic(last_obs).item()

        # PPO 更新
        states_t, actions_t, old_lp_t, returns_t, advs_t = buffer.get_batches(last_value)

        for _ in range(PPO_EPOCHS):
            # Mini-batch（如果数据量小就全量更新）
            n = states_t.shape[0]
            indices = np.random.permutation(n)

            for start in range(0, n, MINI_BATCH):
                end = min(start + MINI_BATCH, n)
                idx = indices[start:end]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_lp = old_lp_t[idx]
                mb_returns = returns_t[idx]
                mb_advs = advs_t[idx]

                log_prob, value, entropy = model.evaluate(mb_states, mb_actions)

                # PPO Clipped Loss
                ratio = torch.exp(log_prob - mb_old_lp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(value, mb_returns)

                entropy_bonus = -0.01 * entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        final_nav = env.nav_history[-1]
        if final_nav > best_nav:
            best_nav = final_nav
            torch.save(model.state_dict(), "ppo_rs_best.pth")

        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}/{EPISODES} | "
                  f"Reward: {total_reward:.2f} | NAV: {final_nav:.4f}")


# ====================== 6. 评估 ======================
def compute_metrics(nav_history):
    nav = np.array(nav_history)
    returns = np.diff(nav) / nav[:-1]

    cr = (nav[-1] / nav[0] - 1) * 100

    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    mdd = np.max(drawdown) * 100

    n_days = len(returns)
    ann_return = (nav[-1] / nav[0]) ** (252 / max(n_days, 1)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)

    sharpe = ann_return / (ann_vol + 1e-8)
    calmar = ann_return / (mdd / 100 + 1e-8)

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = ann_return / (downside_vol + 1e-8)

    return {
        'CR(%)': round(cr, 2),
        'MDD(%)': round(mdd, 2),
        'Sharpe': round(sharpe, 2),
        'Calmar': round(calmar, 2),
        'Sortino': round(sortino, 2)
    }


def evaluate_model(model, test_df, label="Test"):
    env = TradingEnvRS(test_df, lookback=LOOKBACK)
    state = env.reset()

    while not env.done:
        action = model.get_action(state, eval_mode=True)
        state, _, _ = env.step(action)

    metrics = compute_metrics(env.nav_history)
    print(f"\n{'='*50}")
    print(f"  PPO-RS {label} Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")
    return metrics


# ====================== 7. 主函数 ======================
def main():
    print("🚀 PPO-RS (PPO + Risk Shaping) Baseline for CSI 300")
    print(f"   Device: {DEVICE}")
    print(f"   Risk Shaping: DD_penalty={DD_PENALTY}, Vol_penalty={VOL_PENALTY}, DD_threshold={DD_THRESHOLD}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ 未找到数据文件: {CSV_PATH}")

    df = load_and_preprocess(CSV_PATH)
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    print(f"   日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    train_df, val_df, test_df = split_data(df)
    print(f"✅ 数据划分: 训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)}")

    # 初始化
    env_tmp = TradingEnvRS(train_df, lookback=LOOKBACK)
    model = ActorCritic(obs_dim=env_tmp.obs_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_ACTOR)
    print(f"✅ PPO-RS 初始化完成 (obs_dim={env_tmp.obs_dim})")

    # 训练
    print(f"\n📈 开始训练 ({EPISODES} episodes)...")
    train_ppo_rs(model, optimizer, train_df)

    # 加载最优
    if os.path.exists("ppo_rs_best.pth"):
        model.load_state_dict(torch.load("ppo_rs_best.pth", map_location=DEVICE))
        print("✅ 已加载最优策略权重")

    # OOS 测试
    metrics = evaluate_model(model, test_df, label="OOS Test")

    # 保存
    result_df = pd.DataFrame([metrics], index=["PPO-RS"])
    result_df.to_csv("ppo_rs_baseline_results.csv")
    print(f"\n✅ 结果已保存到 ppo_rs_baseline_results.csv")


if __name__ == "__main__":
    main()
