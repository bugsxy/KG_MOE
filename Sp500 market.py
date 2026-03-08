"""
S&P 500 对比实验 3: Buy & Hold（市场基准）
"""
import numpy as np
import pandas as pd
import os

CSV_PATH = "sp500_1000d.csv"

def main():
    print("🚀 Buy & Hold (S&P 500 Market Benchmark)")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 取 OOS 测试集（后 20%）
    n = len(df)
    test_df = df.iloc[int(n * 0.8):].reset_index(drop=True)

    prices = test_df['close'].values
    nav = prices / prices[0]

    returns = np.diff(nav) / nav[:-1]
    cr = (nav[-1] / nav[0] - 1) * 100
    peak = np.maximum.accumulate(nav)
    mdd = np.max((peak - nav) / peak) * 100
    n_days = len(returns)
    ann_ret = (nav[-1] / nav[0]) ** (252 / max(n_days, 1)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-8)
    calmar = ann_ret / (mdd / 100 + 1e-8)
    ds = returns[returns < 0]
    ds_vol = np.std(ds) * np.sqrt(252) if len(ds) > 0 else 1e-8
    sortino = ann_ret / (ds_vol + 1e-8)

    metrics = {
        'CR(%)': round(cr, 2),
        'MDD(%)': round(mdd, 2),
        'Sharpe': round(sharpe, 2),
        'Calmar': round(calmar, 2),
        'Sortino': round(sortino, 2)
    }

    print(f"\n{'='*50}")
    print(f"  S&P 500 Buy & Hold OOS Results")
    print(f"{'='*50}")
    print(f"  日期: {test_df['date'].iloc[0]} ~ {test_df['date'].iloc[-1]}")
    print(f"  交易日: {len(test_df)}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")

    pd.DataFrame([metrics], index=["S&P 500 (Market)"]).to_csv("sp500_market_results.csv")
    print("✅ 保存到 sp500_market_results.csv")


if __name__ == "__main__":
    main()
