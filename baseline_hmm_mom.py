"""
Baseline 2: HMM-MoM (Hidden Markov Model + Momentum Strategy) for CSI 300
- HMM 检测市场 regime（牛市/震荡/熊市）
- 根据 regime 执行动量策略：牛市做多、熊市做空、震荡轻仓
- 使用与 Deep KG-MoE 相同的数据和划分
- 输出 CR, MDD, Sharpe, Calmar, Sortino
- 需要安装: pip install hmmlearn
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    print("❌ 请先安装 hmmlearn: pip install hmmlearn")
    exit(1)

# ====================== 配置 ======================
CSV_PATH = "csi300_1000d.csv"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
N_REGIMES = 3          # 三个 regime: 牛市/震荡/熊市
TRANSACTION_COST = 0.001  # 10 bps
MOMENTUM_WINDOW = 20    # 动量计算窗口


# ====================== 1. 数据加载 ======================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 基础特征
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['return'].rolling(20).std()
    df['momentum'] = df['close'] / df['close'].shift(MOMENTUM_WINDOW) - 1

    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


# ====================== 2. HMM 训练 ======================
def train_hmm(train_df):
    """用训练集的 return + volatility 拟合 3-state Gaussian HMM"""
    features = train_df[['return', 'volatility']].values

    best_model = None
    best_score = -np.inf

    # 多次初始化取最优（HMM 对初始值敏感）
    for seed in range(10):
        try:
            model = GaussianHMM(
                n_components=N_REGIMES,
                covariance_type='full',
                n_iter=200,
                random_state=seed,
                tol=1e-4
            )
            model.fit(features)
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("❌ HMM 训练失败")

    # 识别每个 regime 的含义（按 return 均值排序）
    means = best_model.means_[:, 0]  # return 维度的均值
    regime_order = np.argsort(means)  # 从小到大: 熊市, 震荡, 牛市
    regime_map = {}
    regime_map[regime_order[0]] = 'bear'
    regime_map[regime_order[1]] = 'neutral'
    regime_map[regime_order[2]] = 'bull'

    print(f"✅ HMM 训练完成 (best log-likelihood: {best_score:.2f})")
    for state_id, label in regime_map.items():
        print(f"   Regime {state_id} ({label}): "
              f"mean_return={best_model.means_[state_id, 0]:.6f}, "
              f"mean_vol={best_model.means_[state_id, 1]:.6f}")

    return best_model, regime_map


# ====================== 3. Regime 预测 ======================
def predict_regimes(model, df):
    """对整段数据做 regime 预测"""
    features = df[['return', 'volatility']].values
    regimes = model.predict(features)
    return regimes


# ====================== 4. 动量交易策略 ======================
def momentum_strategy(df, regimes, regime_map):
    """
    根据 HMM regime + 动量信号决定仓位:
    - 牛市 (bull): 仓位 = +0.8 * sign(momentum)，顺势做多
    - 熊市 (bear): 仓位 = -0.6 * sign(momentum)，顺势做空
    - 震荡 (neutral): 仓位 = +0.2 * momentum_signal，轻仓
    """
    positions = np.zeros(len(df))
    nav = np.ones(len(df))
    nav_peak = 1.0

    # 反转 regime_map: label -> list of state_ids
    label_to_ids = {}
    for state_id, label in regime_map.items():
        if label not in label_to_ids:
            label_to_ids[label] = []
        label_to_ids[label].append(state_id)

    prev_position = 0.0

    for i in range(1, len(df)):
        regime_id = regimes[i]
        regime_label = regime_map.get(regime_id, 'neutral')
        momentum = df['momentum'].iloc[i]

        # 动量信号
        mom_signal = np.clip(momentum * 5, -1, 1)  # 缩放到 [-1, 1]

        # 根据 regime 决定仓位
        if regime_label == 'bull':
            target_pos = 0.8 * (1.0 if mom_signal > 0 else 0.3)
        elif regime_label == 'bear':
            target_pos = -0.6 * (1.0 if mom_signal < 0 else 0.3)
        else:  # neutral
            target_pos = 0.2 * mom_signal

        target_pos = np.clip(target_pos, -1.0, 1.0)

        # 交易成本
        trade_cost = abs(target_pos - prev_position) * TRANSACTION_COST

        # PnL
        ret = df['return'].iloc[i]
        pnl = prev_position * ret - trade_cost

        nav[i] = nav[i - 1] * (1 + pnl)
        nav_peak = max(nav_peak, nav[i])
        positions[i] = target_pos
        prev_position = target_pos

    return nav, positions


# ====================== 5. 评估指标 ======================
def compute_metrics(nav_array):
    """计算 CR, MDD, Sharpe, Calmar, Sortino"""
    nav = np.array(nav_array)
    returns = np.diff(nav) / nav[:-1]

    # CR
    cr = (nav[-1] / nav[0] - 1) * 100

    # MDD
    peak = np.maximum.accumulate(nav)
    drawdown = (peak - nav) / peak
    mdd = np.max(drawdown) * 100

    # 年化
    n_days = len(returns)
    ann_return = (nav[-1] / nav[0]) ** (252 / max(n_days, 1)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)

    # Sharpe
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


# ====================== 6. 主函数 ======================
def main():
    print("🚀 HMM-MoM Baseline for CSI 300 Trading")

    # 加载数据
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ 未找到数据文件: {CSV_PATH}")

    df = load_data(CSV_PATH)
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    print(f"   日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")

    # 划分
    train_df, val_df, test_df = split_data(df)
    print(f"✅ 数据划分: 训练 {len(train_df)} | 验证 {len(val_df)} | 测试 {len(test_df)}")

    # 训练 HMM
    print(f"\n📊 训练 HMM ({N_REGIMES} regimes)...")
    hmm_model, regime_map = train_hmm(train_df)

    # 在测试集上预测 regime
    print(f"\n🔍 OOS 测试集 Regime 预测...")
    test_regimes = predict_regimes(hmm_model, test_df)

    # 统计 regime 分布
    for state_id, label in regime_map.items():
        count = np.sum(test_regimes == state_id)
        pct = count / len(test_regimes) * 100
        print(f"   {label} (state {state_id}): {count} days ({pct:.1f}%)")

    # 执行动量策略
    print(f"\n📈 执行 HMM + Momentum 策略...")
    nav, positions = momentum_strategy(test_df, test_regimes, regime_map)

    # 计算指标
    metrics = compute_metrics(nav)
    print(f"\n{'='*50}")
    print(f"  HMM-MoM OOS Test Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")

    # 额外信息
    avg_pos = np.mean(np.abs(positions))
    turnover = np.mean(np.abs(np.diff(positions)))
    print(f"\n  平均持仓: {avg_pos:.3f}")
    print(f"  日均换手: {turnover:.4f}")

    # 保存
    result_df = pd.DataFrame([metrics], index=["HMM-MoM"])
    result_df.to_csv("hmm_mom_baseline_results.csv")
    print(f"\n✅ 结果已保存到 hmm_mom_baseline_results.csv")

    # 保存 regime 和 NAV 序列（方便画图）
    test_df_out = test_df.copy()
    test_df_out['regime'] = test_regimes
    test_df_out['nav'] = nav
    test_df_out['position'] = positions
    test_df_out[['date', 'close', 'return', 'regime', 'nav', 'position']].to_csv(
        "hmm_mom_trajectory.csv", index=False
    )
    print(f"✅ 交易轨迹已保存到 hmm_mom_trajectory.csv")


if __name__ == "__main__":
    main()
