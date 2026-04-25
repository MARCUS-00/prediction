# Ensemble Meta-Learner — DEPRECATED (v10)

The stacking ensemble has been removed from the active pipeline in v10.

**Reason:** The meta-learner collapsed to predicting UP on 99% of test rows:
- Confusion matrix: DOWN recall = 0.01, UP recall = 0.99
- Root cause: both XGBoost and LSTM produced flat probabilities (0.48–0.52)
  so the logistic regression had no useful signal to stack on
- Stacking only helps when base models have complementary, non-degenerate errors

**What replaced it:**
- Single XGBoost with CalibratedClassifierCV (isotonic method)
- Calibration fixes the flat probability issue directly at the source
- One well-tuned model with calibrated outputs outperforms stacking 
  two poorly-calibrated models

**To re-enable ensemble in future:**
- First ensure each base model has AUC > 0.57 independently
- Use proper probability calibration on each base model before stacking
- Consider diversity: e.g., XGBoost on technical features + LightGBM on 
  fundamental features (different feature spaces, not different architectures
  on the same features)
