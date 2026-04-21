import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from models.xgboost.predict import load_xgb, get_feature_importance


def get_top_features(top_n=15):
    return get_feature_importance(load_xgb(), top_n=top_n)


def importance_for_row(row: pd.Series, payload=None) -> list:
    if payload is None: payload = load_xgb()
    fi = get_feature_importance(payload, top_n=10)
    return [(f, float(row.get(f, np.nan)), float(s))
            for f,s in fi.items() if not pd.isna(row.get(f, np.nan))]


if __name__ == "__main__":
    fi = get_top_features(15)
    print("\nTop 15 XGBoost Features:")
    for f,s in fi.items(): print(f"  {f:<35} {s:.1f}")