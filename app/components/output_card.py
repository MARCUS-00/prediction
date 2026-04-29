import html
import streamlit as st


def render_card(result: dict):
    colour = {
        "BUY":     "#d4edda",
        "SELL":    "#f8d7da",
        "HOLD":    "#fff3cd",
        "OBSERVE": "#e2e3e5",
    }
    rec = str(result.get("Recommendation", ""))
    bg  = colour.get(rec, "#ffffff")

    stock        = html.escape(str(result.get("Stock", "")))
    prediction   = html.escape(str(result.get("Prediction", "")))
    expected     = html.escape(str(result.get("Expected_Movement", "")))
    confidence   = html.escape(str(result.get("Confidence", "")))
    conf_level   = html.escape(str(result.get("Confidence_Level", "")))
    recommend    = html.escape(rec)
    last_close   = html.escape(str(result.get("Last_Close", "")))
    last_date    = html.escape(str(result.get("Last_Date", "")))

    st.markdown(
        f"""
        <div style="background:{bg};padding:14px;border-radius:10px;
                    border:1px solid #ccc;margin-bottom:10px;">
            <h3 style="margin:0">{stock} - {prediction}
                <span style="font-size:13px;color:#555;"> {expected}</span>
            </h3>
            <p style="margin:4px 0">
                <b>Confidence:</b> {confidence} &nbsp;|&nbsp;
                <b>Signal:</b> {conf_level} &nbsp;|&nbsp;
                <b>Recommendation:</b> {recommend}
            </p>
            <p style="margin:4px 0;font-size:13px;color:#555;">
                Close: {last_close} &nbsp;|&nbsp; Date: {last_date}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    factors = result.get("XAI_Factors", []) or []
    if factors:
        with st.expander("Key Factors (XAI)"):
            for b in factors:
                st.markdown(f"- {b}")
