# save as eval_on_test.py (or run in a notebook cell)
import json, numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

TEST_CSV = "outputs/modernbert_multitask/splits/test.csv"
PRED_CSV = "./test_split_predictions_multitask.csv"

df_true = pd.read_csv(TEST_CSV)
df_pred = pd.read_csv(PRED_CSV)

# ---- Category metrics
y_true_cat = df_true["category"].astype(str).values
y_pred_cat = df_pred["pred_category"].astype(str).values

cat_acc = accuracy_score(y_true_cat, y_pred_cat)
cat_macro_f1 = f1_score(y_true_cat, y_pred_cat, average="macro")
cat_weighted_f1 = f1_score(y_true_cat, y_pred_cat, average="weighted")
cat_report = classification_report(
    y_true_cat, y_pred_cat, zero_division=0, output_dict=True
)

# ---- Importance metrics (threshold = 0.5 by default)
y_true_imp = (
    df_true["important"]
    .astype(str)
    .str.lower()
    .isin({"1", "true", "t", "yes", "y"})
    .astype(int)
    .values
)
p1 = df_pred["pred_important_p"].astype(float).values
y_pred_imp = (p1 >= 0.5).astype(int)

imp_acc = accuracy_score(y_true_imp, y_pred_imp)
imp_prec, imp_rec, imp_f1, _ = precision_recall_fscore_support(
    y_true_imp, y_pred_imp, average="binary", zero_division=0
)
imp_roc = roc_auc_score(y_true_imp, p1)
imp_pr_auc = average_precision_score(y_true_imp, p1)

# ---- Confusion matrices
cm_cat = pd.DataFrame(confusion_matrix(y_true_cat, y_pred_cat))
cm_imp = pd.DataFrame(
    confusion_matrix(y_true_imp, y_pred_imp),
    index=["not_important", "important"],
    columns=["pred_not_important", "pred_important"],
)

out = {
    "category": {
        "accuracy": cat_acc,
        "macro_f1": cat_macro_f1,
        "weighted_f1": cat_weighted_f1,
        "per_class": cat_report,
    },
    "important": {
        "accuracy": imp_acc,
        "precision": float(imp_prec),
        "recall": float(imp_rec),
        "f1": float(imp_f1),
        "roc_auc": float(imp_roc),
        "pr_auc": float(imp_pr_auc),
    },
}

print(json.dumps(out, indent=2))
cm_cat.to_csv("confusion_category.csv", index=False)
cm_imp.to_csv("confusion_important.csv")
print("Wrote confusion_category.csv and confusion_important.csv")
