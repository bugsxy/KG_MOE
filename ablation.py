import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import os
from torch.optim.lr_scheduler import OneCycleLR

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 严格固定全局种子，确保基线结果不变
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. 硬件与数据通道 (强制GPU)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise SystemError("GPU未就绪，请检查CUDA环境。")
print(f"🚀 Detected GPU count: {torch.cuda.device_count()}")

data_dir = '/kaggle/input/datasets/shayulovejin/111111'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
csv_path = os.path.join(data_dir, csv_files[0])
df = pd.read_csv(csv_path, encoding='utf-8-sig')
raw_rets = df['close'].pct_change().fillna(0).values

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
# 1. 物理引擎
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
# 2. 模型架构 (基线与消融变体)
# ==========================================
# [基线] 原始完全体，保证参数与权重初始化与原版一致
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

# [消融变体 1] w/o MoE (剥离专家系统，退化为单MLP)
class Ablation_NoMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dropout=0.1)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.single_expert = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        
        self.phys_expert = nn.Sequential(nn.Linear(4, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64))
        self.meta_net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 3))
        self.head = nn.Linear(64, 1)

    def forward(self, x, phys):
        lstm_out, _ = self.lstm(x)
        trans_out = self.trans(lstm_out)[:, -1, :]
        main_out = self.single_expert(trans_out)
        
        f_p = self.phys_expert(phys)
        meta_out = self.meta_net(phys)
        fused = 0.7 * main_out + 0.3 * f_p
        # 返回均匀的dummy weight以兼容原代码接口
        dummy_weights = torch.ones(x.size(0), 3, device=x.device) / 3.0
        return self.head(fused), torch.sigmoid(meta_out[:, -1:]), dummy_weights

# [消融变体 2] w/o Phys (剥离物理先验融合)
class Ablation_NoPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dropout=0.1)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)) for _ in range(3)])
        self.gate = nn.Linear(64, 3)
        
        self.head = nn.Linear(64, 1)

    def forward(self, x, phys):
        lstm_out, _ = self.lstm(x)
        trans_out = self.trans(lstm_out)[:, -1, :]
        gate_scores = self.gate(trans_out)
        weights = F.softmax(gate_scores, dim=-1)
        moe_out = sum(w.unsqueeze(1) * expert(trans_out) for w, expert in zip(weights.unbind(1), self.experts))
        
        # 截断物理特征融合，完全信任数据驱动
        dummy_conf = torch.ones(x.size(0), 1, device=x.device) * 0.5 
        return self.head(moe_out), dummy_conf, weights

# ==========================================
# 3. 训练流程 (严格分离)
# ==========================================
def train_model(model, raw_rets, kf_trend, ou_vol, ou_dev, momentum, split_idx, is_nomoe=False):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    window, epochs, batch_size, SCALE = 15, 200, 64, 100.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=0.005, total_steps=epochs*100, pct_start=0.1)
    
    model.train()
    for epoch in range(epochs):
        for i in range(window, split_idx - batch_size, batch_size):
            obs = torch.tensor([raw_rets[j-window:j] * SCALE for j in range(i, i+batch_size)], dtype=torch.float32).view(batch_size, window, 1).to(device)
            phys = torch.tensor([[kf_trend[j]*SCALE, ou_vol[j]*SCALE, ou_dev[j]*SCALE, momentum[j]*SCALE] for j in range(i, i+batch_size)], dtype=torch.float32).to(device)
            target_val = torch.tensor([[raw_rets[j+1]*SCALE] for j in range(i, i+batch_size)], dtype=torch.float32).to(device)
            target_dir = torch.tensor([[1.0 if raw_rets[j+1] > 0 else -1.0] for j in range(i, i+batch_size)], dtype=torch.float32).to(device)

            pred_v, conf, weights = model(obs, phys)
            loss_mse = F.huber_loss(pred_v, target_val)
            loss_dir = F.mse_loss(pred_v, target_dir)
            rolling_sharpe = (pred_v.mean() / (pred_v.std() + 1e-8)) * torch.sqrt(torch.tensor(252.0, device=device))
            loss_sharpe = -rolling_sharpe * 0.05
            
            if is_nomoe:
                entropy = torch.tensor(0.0, device=device)
            else:
                entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=1).mean()
                
            loss_ours = loss_mse + 2.0 * loss_dir + loss_sharpe - 0.08 * entropy
            
            optimizer.zero_grad()
            loss_ours.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
    return model

def main():
    kf_trend, ou_vol, ou_dev, momentum = generate_advanced_physics(raw_rets)
    split_idx = int(len(raw_rets) * 0.7)
    
    print("🔥 Training Baseline (Deep KG-MoE V6)...")
    # 强制重置种子以确保基线初始化和原版一模一样
    torch.manual_seed(42)
    model_baseline = train_model(RealMoE_TransMeta_KalmanOU(), raw_rets, kf_trend, ou_vol, ou_dev, momentum, split_idx)
    
    print("🔥 Training Ablation: w/o MoE...")
    model_nomoe = train_model(Ablation_NoMoE(), raw_rets, kf_trend, ou_vol, ou_dev, momentum, split_idx, is_nomoe=True)
    
    print("🔥 Training Ablation: w/o Phys...")
    model_nophys = train_model(Ablation_NoPhys(), raw_rets, kf_trend, ou_vol, ou_dev, momentum, split_idx)

    # ==========================================
    # 4. 闭卷回测：V6版超强风控 (所有变体统一规则)
    # ==========================================
    print("⚔️ Starting OOS Backtest with Ablation Variants...")
    model_baseline.eval(); model_nomoe.eval(); model_nophys.eval()
    test_market_rets = raw_rets[split_idx:]
    test_len = len(test_market_rets)
    
    # 存储回测结果
    results = {
        "Deep KG-MoE (Baseline V6)": np.zeros(test_len),
        "Ablation: w/o MoE": np.zeros(test_len),
        "Ablation: w/o Phys": np.zeros(test_len),
        "CSI 300 (Market)": test_market_rets
    }
    
    models_dict = {
        "Deep KG-MoE (Baseline V6)": model_baseline,
        "Ablation: w/o MoE": model_nomoe,
        "Ablation: w/o Phys": model_nophys
    }

    SCALE = 100.0
    COST = 0.0005
    TARGET_VOL = 0.12          
    MAX_LEVERAGE = 2.0         
    MAX_DD_THRESHOLD = 0.07    

    with torch.no_grad():
        for name, model in models_dict.items():
            current_nav = 1.0
            running_max_nav = 1.0
            last_pos = 0.0
            
            for idx_test, i in enumerate(range(split_idx, len(raw_rets))):
                obs = torch.tensor(raw_rets[i-15:i]*SCALE, dtype=torch.float32).view(1, 15, 1).to(device)
                phys = torch.tensor([[kf_trend[i-1]*SCALE, ou_vol[i-1]*SCALE, ou_dev[i-1]*SCALE, momentum[i-1]*SCALE]], dtype=torch.float32).to(device)
                today_mkt = test_market_rets[idx_test]
                
                # 预测输出
                pred_v, conf, _ = model(obs, phys)
                raw_sig = torch.tanh(pred_v).item()
                leverage = min(conf.item() * 3.5 + (1.0 if momentum[i-1] > 0 else 0), MAX_LEVERAGE)
                
                # 风控机制 (各变体一致)
                current_vol = ou_vol[i-1] * np.sqrt(252)
                vol_scalar = TARGET_VOL / (current_vol + 1e-8)
                leverage = leverage * np.clip(vol_scalar, 0.5, 2.0)
                
                if abs(ou_dev[i-1]) > 2.8 * ou_vol[i-1]:
                    curr_pos = raw_sig * leverage * 0.35
                elif kf_trend[i-1] < -0.02:
                    curr_pos = raw_sig * leverage * 0.6
                else:
                    curr_pos = raw_sig * leverage
                
                current_nav *= (1 + curr_pos * today_mkt)
                running_max_nav = max(running_max_nav, current_nav)
                current_dd = (running_max_nav - current_nav) / running_max_nav
                if current_dd > MAX_DD_THRESHOLD:
                    curr_pos *= 0.3   
                
                curr_pos = np.clip(curr_pos, -2.0, 2.0)
                results[name][idx_test] = curr_pos * today_mkt - abs(curr_pos - last_pos) * COST
                last_pos = curr_pos

    # ==========================================
    # 5. 战报与可视化
    # ==========================================
    print("\n" + "🛡️" * 35)
    print(f"{'Model Name':<35} | {'CR(%)':>8} | {'MDD(%)':>8} | {'Sharpe':>7} | {'Calmar':>7} | {'Sortino':>7}")
    print("-" * 95)
    curves = {}
    for name, rets in results.items():
        metrics, curve = calc_metrics(rets, name)
        curves[name] = curve
        print(f"{metrics[0]:<35} | {metrics[1]:>8.2f} | {metrics[2]:>8.4f} | {metrics[3]:>7.2f} | {metrics[4]:>7.2f} | {metrics[5]:>7.2f}")
    print("🛡️" * 35)

    plt.figure(figsize=(12, 6), dpi=150)
    colors = ['#D73027', '#4575B4', '#F46D43', '#999999']
    for idx, (name, curve) in enumerate(curves.items()):
        plt.plot(curve, label=name, color=colors[idx], lw=2.5 if 'Baseline' in name else 1.5, alpha=1 if 'Baseline' in name else 0.8)
        
    plt.title("Ablation Study: Architectural Components Validation (OOS)", fontsize=14)
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Return")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()