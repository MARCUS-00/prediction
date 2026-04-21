import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st
from prediction.watchlist       import generate_watchlist
from app.components.output_card import render_card
from config.settings            import MERGED_CSV


def render_watchlist_page():
    st.header("🏆 Daily Watchlist")
    if st.button("🔄 Generate"):
        with st.spinner("Running predictions ..."):
            try:
                df = pd.read_csv(MERGED_CSV)
                wl = generate_watchlist(df)
                if wl.empty: st.error("No predictions. Train models first."); return
                st.success(f"Predictions for {len(wl)} stocks")
                cols = ["Stock","Prediction","Expected_Movement","Confidence",
                        "Confidence_Level","Recommendation","Last_Close"]
                st.dataframe(wl[[c for c in cols if c in wl.columns]], use_container_width=True)
                st.subheader("Cards")
                for _,row in wl.iterrows():
                    xai = [b for b in str(row.get("XAI_Factors","")).split(" | ") if b]
                    render_card({**row.to_dict(),"XAI_Factors":xai})
            except FileNotFoundError:
                st.error(f"merged_final.csv not found. Run python features/merge_features.py")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Click Generate to run predictions.")