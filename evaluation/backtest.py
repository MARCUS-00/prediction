import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from config.settings import LABEL_MAP_INV


def run_backtest(test_df, pred_col="Predicted",
                 label_col="Direction", ret_col="Return_1d") -> dict:
    df = test_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Date","Stock"], inplace=True)

    df["pred_str"]   = df[pred_col].map(LABEL_MAP_INV)   if df[pred_col].dtype in [int,float] else df[pred_col]
    df["actual_str"] = df[label_col].map(LABEL_MAP_INV)  if df[label_col].dtype in [int,float] else df[label_col]

    df["strat_ret"] = np.where(df["pred_str"]=="UP", df[ret_col].fillna(0), 0.0)
    df["bh_ret"]    = df[ret_col].fillna(0)

    daily = df.groupby("Date")[["strat_ret","bh_ret"]].mean()
    daily["strat_cum"] = (1+daily["strat_ret"]).cumprod()
    daily["bh_cum"]    = (1+daily["bh_ret"]).cumprod()

    strat_ret  = float(daily["strat_cum"].iloc[-1]-1)
    bh_ret     = float(daily["bh_cum"].iloc[-1]-1)
    std        = daily["strat_ret"].std()
    sharpe     = (daily["strat_ret"].mean()/std*np.sqrt(252)) if std>0 else 0.0
    max_dd     = float(((daily["strat_cum"]-daily["strat_cum"].cummax())/daily["strat_cum"].cummax()).min())
    active     = df[df["pred_str"]=="UP"]
    win_rate   = float((active["strat_ret"]>0).mean()) if len(active)>0 else 0.0

    print(f"\n{'─'*45}")
    print("  BACKTEST RESULTS")
    print(f"{'─'*45}")
    print(f"  Strategy Return : {strat_ret*100:.2f}%")
    print(f"  Buy & Hold      : {bh_ret*100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd*100:.2f}%")
    print(f"  Win Rate        : {win_rate*100:.1f}%")
    print(f"  UP Trades       : {len(active)}")
    print(f"{'─'*45}\n")

    return {"strategy_return":strat_ret,"bh_return":bh_ret,"sharpe":sharpe,
            "max_drawdown":max_dd,"win_rate":win_rate,"daily":daily}