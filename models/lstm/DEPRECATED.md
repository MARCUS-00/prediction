# LSTM Model — DEPRECATED (v10)

The LSTM model has been removed from the active pipeline in v10.

**Reason:** The LSTM provided no measurable benefit over XGBoost:
- Val AUC 0.523, test AUC 0.505 — indistinguishable from random
- Early stopping triggered at epoch 16/60 (val_loss diverged from epoch 1)
- 26K parameters trained on noisy labels with 15-step sequences → guaranteed overfit
- Sequence modelling requires much larger per-stock history; 50 stocks × 2700 days 
  is insufficient for a temporal model to learn stable patterns

**What replaced it:**
- Alpha-based target (stock outperformance vs NIFTY) provides a cleaner signal
- Calibrated XGBoost with strong regularisation captures the same temporal 
  momentum patterns without sequence modelling overhead

**If you want to re-enable LSTM in future:**
- Use minimum 10+ years of tick data per stock
- Use transformer architecture (not vanilla LSTM)
- Target alpha directly (not raw direction)
- Train separate model per sector, not one model for all stocks
