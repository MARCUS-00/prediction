def confidence_label(score: float) -> str:
    if score >= 0.85: return "Strong Signal ✅"
    if score >= 0.70: return "Moderate Signal ⚠️"
    return "Weak Signal ❌"

def recommendation(direction: str, confidence: float) -> str:
    if confidence < 0.70: return "OBSERVE"
    return {"UP":"BUY","DOWN":"SELL","FLAT":"HOLD"}.get(direction,"OBSERVE")

def expected_movement(row, direction: str) -> str:
    atr   = float(row.get("ATR",  0.0) or 0.0)
    close = float(row.get("Close",1.0) or 1.0)
    pct   = round((atr / max(close,1)) * 100, 2)
    if direction == "UP":   return f"+{pct}%"
    if direction == "DOWN": return f"-{pct}%"
    return "~0%"