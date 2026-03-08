import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
from torch.optim.lr_scheduler import OneCycleLR

warnings.filterwarnings("ignore")
torch.manual_seed(42)

# ==========================================
# 0. 设备 + 数据路径（本地）
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

csv_path = "sp500_1000d.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ 未找到 {csv_path}，请先运行 download_us_data.py")

df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]
raw_rets = df['close'].pct_change().fillna(0).values
print(f"📂 数据加载完成: {csv_path} ({len(df)} 条)")

def calc_metrics(rets, name):
    rets = np.array(rets)
    if len(rets) == 0 or np.all(rets == 0):
        return [name, 0.0, 0.0, 0.0, 0.0, 0.0], np.ones(1)
    net_val = np.cumprod(1 + rets)
    cr = (net_val[-1] - 1) * 100
    roll_max = np.maximum.accumulate(net_val)
    mdd_decimal = np.max((roll_max - net_val) / (roll_max + 1e-9))
    sharpe = np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252)
    ann_ret = (net_val[-1] ** (252 / len(rets))) - 1 if net_val[-1] > 0 else -0.01
    calmar = ann_ret / (mdd_decimal + 1e-9)
    downside_rets = rets[rets < 0]
    sortino = (np.mean(rets) * 252) / (np.std(downside_rets) * np.sqrt(252) + 1e-6) if len(downside_rets) > 0 else 0
    return [name, round(cr, 2), round(mdd_decimal * 100, 4), round(sharpe, 2), round(calmar, 2), round(sortino, 2)], net_val

# ==========================================
# 1. 物理引擎（不变）
# ==========================================
def generate_advanced_physics(rets):
    kf_trend = pd.Series(rets).ewm(span=5).mean().values
    ou_vol = pd.Series(rets).rolling(15).std().fillna(0.01).values
    ou_dev = rets - pd.Series(rets).rolling(20).mean().fillna(0).values
    ema_fast = pd.Series(rets).ewm(span=5).mean().values
    ema_slow = pd.Series(rets).ewm(span=20).mean().values
    momentum = ema_fast - ema_slow
    return kf_trend, ou_vol, ou_dev, momentum

# ==========================================
# 2. 核心模型（完全不变）
# ==========================================
class RealMoE_TransMeta_KalmanOU(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dropout=0.1)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)) for _ in range(3)])
        self.gate = nn.Linear(64, 3)

        self.phys_expert = nn.Sequential(nn.Linear(4, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64))
        self.meta_net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 3))
        self.head = nn.Linear(64, 1)

    def forward(self, x, phys):
        lstm_out, _ = self.lstm(x)
        trans_out = self.trans(lstm_out)[:, -1, :]
        gate_scores = self.gate(trans_out)
        weights = F.softmax(gate_scores, dim=-1)
        moe_out = sum(w.unsqueeze(1) * expert(trans_out) for w, expert in zip(weights.unbind(1), self.experts))
        f_p = self.phys_expert(phys)
        meta_out = self.meta_net(phys)
        fused = 0.7 * moe_out + 0.3 * f_p
        return self.head(fused), torch.sigmoid(meta_out[:, -1:]), weights

# ==========================================
# 3. 训练 + 回测（只有自己的模型）
# ==========================================
def main():
    kf_trend, ou_vol, ou_dev, momentum = generate_advanced_physics(raw_rets)
    split_idx = int(len(raw_rets) * 0.7)

    model_ours = RealMoE_TransMeta_KalmanOU().to(device)

    window, epochs, batch_size, SCALE = 15, 200, 64, 100.0
    opt_ours = torch.optim.AdamW(model_ours.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = OneCycleLR(opt_ours, max_lr=0.005, total_steps=epochs * 100, pct_start=0.1)

    print(f"🔥 Training Deep KG-MoE on S&P 500 ({epochs} epochs)...")
    for epoch in range(epochs):
        model_ours.train()
        for i in range(window, split_idx - batch_size, batch_size):
            obs = torch.tensor([raw_rets[j - window:j] * SCALE for j in range(i, i + batch_size)],
                               dtype=torch.float32).view(batch_size, window, 1).to(device)
            phys = torch.tensor(
                [[kf_trend[j] * SCALE, ou_vol[j] * SCALE, ou_dev[j] * SCALE, momentum[j] * SCALE] for j in
                 range(i, i + batch_size)], dtype=torch.float32).to(device)
            target_val = torch.tensor([[raw_rets[j + 1] * SCALE] for j in range(i, i + batch_size)],
                                      dtype=torch.float32).to(device)
            target_dir = torch.tensor([[1.0 if raw_rets[j + 1] > 0 else -1.0] for j in range(i, i + batch_size)],
                                      dtype=torch.float32).to(device)

            pred_v, conf, weights = model_ours(obs, phys)
            loss_mse = F.huber_loss(pred_v, target_val)
            loss_dir = F.mse_loss(pred_v, target_dir)
            rolling_sharpe = (pred_v.mean() / (pred_v.std() + 1e-8)) * torch.sqrt(
                torch.tensor(252.0, device=device))
            loss_sharpe = -rolling_sharpe * 0.05
            entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=1).mean()
            loss_ours = loss_mse + 2.0 * loss_dir + loss_sharpe - 0.08 * entropy

            opt_ours.zero_grad()
            loss_ours.backward()
            torch.nn.utils.clip_grad_norm_(model_ours.parameters(), max_norm=1.0)
            opt_ours.step()

        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} completed")

    # ==========================================
    # 4. OOS 回测：V6 风控（完全不变）
    # ==========================================
    print("⚔️ Starting OOS Backtest V6 (Ultra Risk Control) on S&P 500...")
    model_ours.eval()
    test_market_rets = raw_rets[split_idx:]
    test_len = len(test_market_rets)
    rets_ours = np.zeros(test_len)
    last_pos_ours = 0.0
    COST = 0.0005
    TARGET_VOL = 0.12
    MAX_LEVERAGE = 2.0
    MAX_DD_THRESHOLD = 0.07

    current_nav = 1.0
    running_max_nav = 1.0

    with torch.no_grad():
        for idx_test, i in enumerate(range(split_idx, len(raw_rets))):
            obs = torch.tensor(raw_rets[i - window:i] * SCALE, dtype=torch.float32).view(1, window, 1).to(device)
            phys = torch.tensor(
                [[kf_trend[i - 1] * SCALE, ou_vol[i - 1] * SCALE, ou_dev[i - 1] * SCALE,
                  momentum[i - 1] * SCALE]], dtype=torch.float32).to(device)
            today_mkt = test_market_rets[idx_test]

            pred_v, conf, _ = model_ours(obs, phys)
            raw_sig = torch.tanh(pred_v).item()
            leverage = min(conf.item() * 3.5 + (1.0 if momentum[i - 1] > 0 else 0), MAX_LEVERAGE)

            current_vol = ou_vol[i - 1] * np.sqrt(252)
            vol_scalar = TARGET_VOL / (current_vol + 1e-8)
            leverage = leverage * np.clip(vol_scalar, 0.5, 2.0)

            if abs(ou_dev[i - 1]) > 2.8 * ou_vol[i - 1]:
                curr_pos = raw_sig * leverage * 0.35
            elif kf_trend[i - 1] < -0.02:
                curr_pos = raw_sig * leverage * 0.6
            else:
                curr_pos = raw_sig * leverage

            current_nav *= (1 + curr_pos * today_mkt)
            running_max_nav = max(running_max_nav, current_nav)
            current_dd = (running_max_nav - current_nav) / running_max_nav
            if current_dd > MAX_DD_THRESHOLD:
                curr_pos *= 0.3

            curr_pos = np.clip(curr_pos, -2.0, 2.0)
            rets_ours[idx_test] = curr_pos * today_mkt - abs(curr_pos - last_pos_ours) * COST
            last_pos_ours = curr_pos

    # ==========================================
    # 5. 输出结果
    # ==========================================
    metrics, nav_curve = calc_metrics(rets_ours, "Deep KG-MoE (Ours)")
    market_metrics, _ = calc_metrics(test_market_rets, "S&P 500 (Market)")

    print(f"\n{'=' * 60}")
    print(f"  Deep KG-MoE on S&P 500 — OOS Results")
    print(f"{'=' * 60}")
    print(f"  {'Model':<30} {'CR(%)':<10} {'MDD(%)':<10} {'Sharpe':<10} {'Calmar':<10} {'Sortino':<10}")
    print(f"  {'-' * 70}")
    print(f"  {metrics[0]:<30} {metrics[1]:<10} {metrics[2]:<10} {metrics[3]:<10} {metrics[4]:<10} {metrics[5]:<10}")
    print(f"  {market_metrics[0]:<30} {market_metrics[1]:<10} {market_metrics[2]:<10} {market_metrics[3]:<10} {market_metrics[4]:<10} {market_metrics[5]:<10}")
    print(f"{'=' * 60}")

    # 保存结果
    result = pd.DataFrame({
        'Model': [metrics[0], market_metrics[0]],
        'CR(%)': [metrics[1], market_metrics[1]],
        'MDD(%)': [metrics[2], market_metrics[2]],
        'Sharpe': [metrics[3], market_metrics[3]],
        'Calmar': [metrics[4], market_metrics[4]],
        'Sortino': [metrics[5], market_metrics[5]],
    })
    result.to_csv("sp500_kgmoe_results.csv", index=False)
    print(f"\n✅ 结果已保存到 sp500_kgmoe_results.csv")


if __name__ == "__main__":
    main()
