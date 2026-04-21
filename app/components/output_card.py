import streamlit as st

def render_card(result: dict):
    colour = {"BUY":"#d4edda","SELL":"#f8d7da","HOLD":"#fff3cd","OBSERVE":"#e2e3e5"}
    bg = colour.get(result.get("Recommendation",""), "#ffffff")
    st.markdown(f"""
        <div style="background:{bg};padding:14px;border-radius:10px;
                    border:1px solid #ccc;margin-bottom:10px;">
            <h3 style="margin:0">{result.get('Stock','')} — {result.get('Prediction','')}
                <span style="font-size:13px;color:#555;"> {result.get('Expected_Movement','')}</span>
            </h3>
            <p style="margin:4px 0">
                <b>Confidence:</b> {result.get('Confidence','')} &nbsp;|&nbsp;
                <b>Signal:</b> {result.get('Confidence_Level','')} &nbsp;|&nbsp;
                <b>Recommendation:</b> {result.get('Recommendation','')}
            </p>
            <p style="margin:4px 0;font-size:13px;color:#555;">
                Close: {result.get('Last_Close','')} &nbsp;|&nbsp; Date: {result.get('Last_Date','')}
            </p>
        </div>""", unsafe_allow_html=True)
    with st.expander("🧠 XAI — Key Factors"):
        for b in result.get("XAI_Factors",[]):
            st.markdown(f"- {b}")