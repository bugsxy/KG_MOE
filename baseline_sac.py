"""
Baseline 1: SAC (Soft Actor-Critic) for CSI 300 Trading
- 使用与 Deep KG-MoE 相同的数据和划分
- 输出 CR, MDD, Sharpe, Calmar, Sortino 五个指标
- 自包含，不依赖外部训练脚本
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置 ======================
CSV_PATH = "csi300_1000d.csv"  # 你的数据文件
LOOKBACK = 15          # 观测窗口，和论文一致
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
INITIAL_CASH = 1_000_000
TRANSACTION_COST = 0.001  # 10 bps
EPISODES = 200
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2            # SAC 熵系数
BUFFER_SIZE = 50000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== 1. 数据加载与预处理 ======================
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 计算特征
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(20).std()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df = df.dropna().reset_index(drop=True)

    # 滚动 Z-score 归一化（252日窗口，和论文一致）
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


# ====================== 2. 交易环境 ======================
class TradingEnv:
    def __init__(self, df, lookback=15):
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.feature_cols = [c for c in df.columns if c.endswith('_norm')]
        self.n_features = len(self.feature_cols)
        self.reset()

    def reset(self):
        self.step_idx = self.lookback
        self.position = 0.0       # 当前仓位 [-1, 1]
        self.nav = 1.0            # 净值
        self.nav_peak = 1.0
        self.nav_history = [1.0]
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        start = self.step_idx - self.lookback
        end = self.step_idx
        window = self.df[self.feature_cols].iloc[start:end].values
        # 展平为一维向量
        obs = window.flatten().astype(np.float32)
        # 追加当前仓位和 drawdown
        dd = (self.nav_peak - self.nav) / (self.nav_peak + 1e-8)
        extra = np.array([self.position, dd], dtype=np.float32)
        return np.concatenate([obs, extra])

    @property
    def obs_dim(self):
        return self.lookback * self.n_features + 2

    def step(self, action):
        """action: float in [-1, 1]，代表目标仓位"""
        action = np.clip(action, -1.0, 1.0)

        # 交易成本
        trade_size = abs(action - self.position)
        cost = trade_size * TRANSACTION_COST

        # 收益
        ret = self.df['return'].iloc[self.step_idx]
        pnl = self.position * ret - cost
        self.nav *= (1 + pnl)

        # 更新
        self.position = action
        self.nav_peak = max(self.nav_peak, self.nav)
        self.nav_history.append(self.nav)
        self.step_idx += 1

        if self.step_idx >= len(self.df) - 1:
            self.done = True

        # reward: differential Sharpe ratio 近似
        if len(self.nav_history) > 2:
            recent_returns = np.diff(self.nav_history[-min(20, len(self.nav_history)):])
            if len(recent_returns) > 1 and np.std(recent_returns) > 1e-8:
                reward = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
            else:
                reward = pnl * 10
        else:
            reward = pnl * 10

        return self._get_obs(), reward, self.done


# ====================== 3. SAC 网络 ======================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        # log_prob with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# ====================== 4. Replay Buffer ======================
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ====================== 5. SAC Agent ======================
class SACAgent:
    def __init__(self, obs_dim):
        self.policy = GaussianPolicy(obs_dim).to(DEVICE)
        self.q1 = SoftQNetwork(obs_dim).to(DEVICE)
        self.q2 = SoftQNetwork(obs_dim).to(DEVICE)
        self.q1_target = SoftQNetwork(obs_dim).to(DEVICE)
        self.q2_target = SoftQNetwork(obs_dim).to(DEVICE)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=LR)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=LR)

        self.buffer = ReplayBuffer()
        self.alpha = ALPHA

    def select_action(self, state, eval_mode=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        if eval_mode:
            mean, _ = self.policy(state_t)
            action = torch.tanh(mean)
        else:
            action, _ = self.policy.sample(state_t)
        return action.cpu().detach().numpy().flatten()[0]

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # --- Q update ---
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = rewards + GAMMA * (1 - dones) * q_next

        q1_loss = F.mse_loss(self.q1(states, actions), q_target)
        q2_loss = F.mse_loss(self.q2(states, actions), q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # --- Policy update ---
        new_action, log_prob = self.policy.sample(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # --- Soft update target ---
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# ====================== 6. 评估指标 ======================
def compute_metrics(nav_history):
    """计算 CR, MDD, Sharpe, Calmar, Sortino"""
    nav = np.array(nav_history)
    returns = np.diff(nav) / nav[:-1]

    # Cumulative Return
    cr = (nav[-1] / nav[0] - 1) * 100

    # Maximum Drawdown
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    mdd = np.max(drawdown) * 100

    # Annualized return & vol (假设 252 交易日)
    n_days = len(returns)
    ann_return = (nav[-1] / nav[0]) ** (252 / max(n_days, 1)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)

    # Sharpe (无风险利率 0)
    sharpe = ann_return / (ann_vol + 1e-8)

    # Calmar
    calmar = ann_return / (mdd / 100 + 1e-8)

    # Sortino
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


# ====================== 7. 训练与测试 ======================
def train_sac(agent, train_df):
    env = TradingEnv(train_df, lookback=LOOKBACK)
    best_nav = 0

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state
            total_reward += reward

        final_nav = env.nav_history[-1]
        if final_nav > best_nav:
            best_nav = final_nav
            torch.save(agent.policy.state_dict(), "sac_best_policy.pth")

        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}/{EPISODES} | "
                  f"Reward: {total_reward:.2f} | NAV: {final_nav:.4f}")


def evaluate_sac(agent, test_df, label="Test"):
    env = TradingEnv(test_df, lookback=LOOKBACK)
    state = env.reset()

    while not env.done:
        action = agent.select_action(state, eval_mode=True)
        state, _, _ = env.step(action)

    metrics = compute_metrics(env.nav_history)
    print(f"\n{'='*50}")
    print(f"  SAC {label} Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")
    return metrics


# ====================== 8. 主函数 ======================
def main():
    print("🚀 SAC Baseline for CSI 300 Trading")
    print(f"   Device: {DEVICE}")

    # 加载数据
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ 未找到数据文件: {CSV_PATH}")

    df = load_and_preprocess(CSV_PATH)
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    print(f"   日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    # 划分数据
    train_df, val_df, test_df = split_data(df)
    print(f"✅ 数据划分: 训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)}")

    # 初始化 Agent
    env_tmp = TradingEnv(train_df, lookback=LOOKBACK)
    agent = SACAgent(obs_dim=env_tmp.obs_dim)
    print(f"✅ SAC Agent 初始化完成 (obs_dim={env_tmp.obs_dim})")

    # 训练
    print(f"\n📈 开始训练 ({EPISODES} episodes)...")
    train_sac(agent, train_df)

    # 加载最优策略
    if os.path.exists("sac_best_policy.pth"):
        agent.policy.load_state_dict(torch.load("sac_best_policy.pth", map_location=DEVICE))
        print("✅ 已加载最优策略权重")

    # OOS 测试
    metrics = evaluate_sac(agent, test_df, label="OOS Test")

    # 保存结果
    result_df = pd.DataFrame([metrics], index=["SAC"])
    result_df.to_csv("sac_baseline_results.csv")
    print(f"\n✅ 结果已保存到 sac_baseline_results.csv")


if __name__ == "__main__":
    main()
