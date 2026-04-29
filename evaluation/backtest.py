"""
evaluation/backtest.py
======================
BUGS FIXED:
  BUG-6  pred_str mapped using LABEL_MAP_INV which maps internal {0,1,2}.
         If Predicted column holds external {-1,0,1}, that mapping is wrong.
         Fixed: detect whether values are internal or external and map correctly.
         Also fixed: strat_ret now captures both UP and short-sells DOWN signals.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config.settings import LABEL_MAP_INV

# String direction → position multiplier (1=long, -1=short, 0=flat)
_DIR_TO_POS = {"UP": 1.0, "DOWN": -1.0, "FLAT": 0.0}


def _to_direction_str(series: pd.Series) -> pd.Series:
    """
    Convert a Predicted column (internal int 0/1/2 OR external int -1/0/1
    OR string 'UP'/'FLAT'/'DOWN') → string direction.
    """
    if pd.api.types.is_object_dtype(series):
        # Already strings
        return series.astype(str)

    vals = series.dropna().unique().tolist()
    if any(v in (-1,) for v in vals):
        # External space {-1, 0, 1}
        ext_to_str = {-1: "DOWN", 0: "FLAT", 1: "UP"}
        return series.astype(int).map(ext_to_str)
    else:
        # Internal space {0, 1, 2}
        return series.astype(int).map(LABEL_MAP_INV)


def run_backtest(test_df, pred_col="Predicted",
                 label_col="label", ret_col="Return_1d") -> dict:
    df = test_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Stock"])

    df["pred_str"] = _to_direction_str(df[pred_col])

    if ret_col not in df.columns:
        raise KeyError(f"Return column '{ret_col}' not in dataframe")

    # FIX: long on UP, short on DOWN, flat on FLAT
    df["position"]  = df["pred_str"].map(_DIR_TO_POS).fillna(0.0)
    df["strat_ret"] = df["position"] * df[ret_col].fillna(0.0)
    df["bh_ret"]    = df[ret_col].fillna(0.0)

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

    active   = df[df["pred_str"] == "UP"]
    win_rate = float((active["strat_ret"] > 0).mean()) if len(active) > 0 else 0.0

    print(f"\n{'-' * 45}")
    print("  BACKTEST RESULTS")
    print(f"{'-' * 45}")
    print(f"  Strategy Return : {strat_ret * 100:.2f}%")
    print(f"  Buy & Hold      : {bh_ret * 100:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd * 100:.2f}%")
    print(f"  Win Rate (UP)   : {win_rate * 100:.1f}%")
    print(f"  UP Trades       : {len(active)}")
    print(f"  DOWN Trades     : {int((df['pred_str'] == 'DOWN').sum())}")
    print(f"{'-' * 45}\n")

    return {"strategy_return": strat_ret, "bh_return": bh_ret, "sharpe": sharpe,
            "max_drawdown": max_dd, "win_rate": win_rate, "daily": daily}
