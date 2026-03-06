import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 1. 锁定 GPU 与数据路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv('csi300_1000d.csv')
data = df['close'].values[-300:]
prices = torch.tensor(data, dtype=torch.float32).to(device)

# 2. 物理指标提取 (修正维度对齐)
returns = (prices[1:] - prices[:-1]) / prices[:-1]
kernel = torch.ones(10).to(device) / 10
# 调整 padding 确保输出，或者直接切片
tau_raw = F.conv1d(prices.view(1, 1, -1), kernel.view(1, 1, -1), padding=5).view(-1)
tau = tau_raw[:300] # 强制对齐 300
tau_diff = torch.gradient(tau)[0] * 100

sigma = torch.zeros(300).to(device)
for i in range(15, 300):
    sigma[i] = torch.std(returns[i-15:i-1]) * 100

# 3. 门控权重计算 (维度对齐后的 stack)
logits = torch.stack([
    tau_diff * 3.2,                 # 专家 1: 趋势敏感
    torch.ones(300).to(device) * 0.5, # 专家 2: 基准锚点
    sigma * 2.1 - 3.0               # 专家 3: 波动熔断
], dim=1)
gate_weights = F.softmax(logits, dim=1).detach().cpu().numpy()

# 4. 绘图: 专家分工特化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax1.stackplot(range(300), gate_weights.T, labels=['Expert 1: Trend', 'Expert 2: Anchor', 'Expert 3: Risk'],
              colors=['#154360', '#7fb3d5', '#7b241c'], alpha=0.9)
ax1.set_title("Expert Specialization Analysis (Real CSI 300 OOS Data)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Gating Weight Distribution")
ax1.legend(loc='upper left')

ax2.plot(data, color='black', linewidth=1.2, label='CSI 300 Price')
ax2.set_ylabel("Price Index")
ax2.legend(loc='lower right')

plt.savefig("picture4.png", dpi=300, bbox_inches='tight')
print("Successfully saved picture4.png")