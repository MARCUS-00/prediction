import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="Stock Prediction", page_icon="📈", layout="wide")

from app.pages.watchlist_page    import render_watchlist_page
from app.pages.single_stock_page import render_single_stock_page

st.title("📈 Stock Prediction System")
st.caption("Nifty 50 · XGBoost + LSTM + FinBERT · Explainable AI")

page = st.sidebar.radio("", ["🏆 Watchlist", "🔍 Single Stock"])
if page == "🏆 Watchlist": render_watchlist_page()
else:                       render_single_stock_page()