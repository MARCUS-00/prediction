import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from prediction.single_stock    import predict_single
from app.components.output_card import render_card
from config.nifty50_tickers     import get_stocks
from config.settings            import MERGED_CSV


def render_single_stock_page():
    st.header("🔍 Single Stock Prediction")
    symbol = st.selectbox("Select stock:", sorted(get_stocks()))
    if st.button("🔮 Predict"):
        with st.spinner(f"Predicting {symbol} ..."):
            try:
                df     = pd.read_csv(MERGED_CSV)
                result = predict_single(symbol, df)
                if "error" in result: st.error(result["error"]); return
                render_card(result)
            except FileNotFoundError:
                st.error("merged_final.csv not found. Run python features/merge_features.py")
            except Exception as e:
                st.error(f"Error: {e}")