import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import (classification_report, f1_score,
                              accuracy_score, confusion_matrix, roc_auc_score)
from config.settings import DIRECTION_LABELS


def evaluate(y_true, y_pred, y_proba=None, model_name="Model", verbose=True) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    rep  = classification_report(y_true, y_pred,
                                  target_names=DIRECTION_LABELS, zero_division=0)
    auc = np.nan
    if y_proba is not None:
        try: 
            if y_proba.shape[1] == 2: # Binary
                auc = roc_auc_score(y_true, y_proba[:,1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception: pass

    if verbose:
        print(f"\n{'-'*55}")
        print(f"  {model_name} Results")
        print(f"{'-'*55}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Weighted F1 : {f1_w:.4f}")
        print(f"  Macro F1    : {f1_m:.4f}")
        if not np.isnan(auc): print(f"  AUC-ROC     : {auc:.4f}")
        print(f"\n{rep}")
        print(f"  Confusion Matrix (DOWN/UP):\n{cm}")
        print(f"{'-'*55}\n")

    return {"model":model_name,"accuracy":acc,"weighted_f1":f1_w,
            "macro_f1":f1_m,"auc_roc":auc,"report":rep, "cm": cm}

def evaluate_all(y_tr, p_tr, pr_tr, 
                 y_v, p_v, pr_v, 
                 y_te, p_te, pr_te, model_name="Model") -> dict:
    
    res_tr = evaluate(y_tr, p_tr, pr_tr, verbose=False)
    res_v  = evaluate(y_v, p_v, pr_v, verbose=False)
    res_te = evaluate(y_te, p_te, pr_te, verbose=False)
    
    print(f"\n{'='*65}")
    print(f"  {model_name.upper()} - EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Metric       | Train Set  | Val Set    | Test Set")
    print(f"  -------------+------------+------------+------------")
    print(f"  Accuracy     |   {res_tr['accuracy']:.4f}   |   {res_v['accuracy']:.4f}   |   {res_te['accuracy']:.4f}")
    print(f"  Weighted F1  |   {res_tr['weighted_f1']:.4f}   |   {res_v['weighted_f1']:.4f}   |   {res_te['weighted_f1']:.4f}")
    print(f"  Macro F1     |   {res_tr['macro_f1']:.4f}   |   {res_v['macro_f1']:.4f}   |   {res_te['macro_f1']:.4f}")
    print(f"  AUC-ROC      |   {res_tr['auc_roc']:.4f}   |   {res_v['auc_roc']:.4f}   |   {res_te['auc_roc']:.4f}")
    print(f"{'='*65}\n")
    
    print(f"  [ Test Set Detailed Report ]")
    print(f"{res_te['report']}")
    print(f"  Confusion Matrix (DOWN/UP):")
    print(f"{res_te['cm']}\n")
    
    return res_te
