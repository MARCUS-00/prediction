import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import LABEL_MAP_INV


def run_backtest(test_df, pred_col="Predicted",
                 label_col="label", ret_col="Return_1d") -> dict:
    df = test_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Stock"])

    if pd.api.types.is_numeric_dtype(df[pred_col]):
        df["pred_str"] = df[pred_col].astype(int).map(LABEL_MAP_INV)
    else:
        df["pred_str"] = df[pred_col].astype(str)

    if ret_col not in df.columns:
        raise KeyError(f"Return column '{ret_col}' not in dataframe")

    df["strat_ret"] = np.where(df["pred_str"] == "UP", df[ret_col].fillna(0), 0.0)
    df["bh_ret"]    = df[ret_col].fillna(0)

    daily = df.groupby("Date")[["strat_ret", "bh_ret"]].mean()
    daily["strat_cum"] = (1 + daily["strat_ret"]).cumprod()
    daily["bh_cum"]    = (1 + daily["bh_ret"]).cumprod()

    if daily.empty:
        return {"strategy_return": 0.0, "bh_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "daily": daily}

    strat_ret = float(daily["strat_cum"].iloc[-1] - 1)
    bh_ret    = float(daily["bh_cum"].iloc[-1]   - 1)
    std       = float(daily["strat_ret"].std())
    sharpe    = float(daily["strat_ret"].mean() / std * np.sqrt(252)) if std > 0 else 0.0
    cummax    = daily["strat_cum"].cummax()
    max_dd    = float(((daily["strat_cum"] - cummax) / cummax.replace(0, np.nan)).min())
    if np.isnan(max_dd):
        max_dd = 0.0
    active    = df[df["pred_str"] == "UP"]
    win_rate  = float((active["strat_ret"] > 0).mean()) if len(active) > 0 else 0.0

    print(f"\n{'-' * 45}")
    print("  BACKTEST RESULTS")
    print(f"{'-' * 45}")
    print(f"  Strategy Return : {strat_ret * 100:.2f}%")
    print(f"  Buy & Hold      : {bh_ret * 100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd * 100:.2f}%")
    print(f"  Win Rate        : {win_rate * 100:.1f}%")
    print(f"  UP Trades       : {len(active)}")
    print(f"{'-' * 45}\n")

    return {"strategy_return": strat_ret, "bh_return": bh_ret, "sharpe": sharpe,
            "max_drawdown": max_dd, "win_rate": win_rate, "daily": daily}
