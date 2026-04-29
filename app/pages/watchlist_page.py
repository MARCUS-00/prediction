import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from prediction.watchlist        import generate_watchlist
from app.components.output_card  import render_card
from config.settings             import MERGED_CSV


@st.cache_data(ttl=3600)
def _load_merged():
    if not os.path.exists(MERGED_CSV):
        return None
    return pd.read_csv(MERGED_CSV, parse_dates=["Date"])


def render_watchlist_page():
    st.header("Daily Watchlist")
    if st.button("Generate"):
        with st.spinner("Running predictions ..."):
            df = _load_merged()
            if df is None:
                st.error("merged_final.csv not found. "
                         "Run: python features/merge_features.py")
                return
            try:
                wl = generate_watchlist(df)
            except Exception as e:
                st.error(f"Error: {e}")
                return
            if wl.empty:
                st.error("No predictions. Train models first.")
                return

            st.success(f"Predictions for {len(wl)} stocks")
            cols = ["Stock", "Prediction", "Expected_Movement", "Confidence",
                    "Confidence_Level", "Recommendation", "Last_Close"]
            visible = [c for c in cols if c in wl.columns]
            st.dataframe(wl[visible], use_container_width=True)

            st.subheader("Cards")
            for _, row in wl.iterrows():
                xai_raw = str(row.get("XAI_Factors", ""))
                xai = [b.strip() for b in xai_raw.split("|") if b.strip()]
                render_card({**row.to_dict(), "XAI_Factors": xai})
    else:
        st.info("Click Generate to run predictions.")
