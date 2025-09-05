# eval_with_charts.py
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def main(true_csv, pred_csv, outdir):
    os.makedirs(outdir, exist_ok=True)

    # ---- Load
    df_true = pd.read_csv(true_csv)
    df_pred = pd.read_csv(pred_csv)

    # ---- Category metrics
    y_true_cat = df_true["category"].astype(str).values
    y_pred_cat = df_pred["pred_category"].astype(str).values

    cat_acc = accuracy_score(y_true_cat, y_pred_cat)
    cat_macro_f1 = f1_score(y_true_cat, y_pred_cat, average="macro")
    cat_weighted_f1 = f1_score(y_true_cat, y_pred_cat, average="weighted")
    cat_report = classification_report(
        y_true_cat, y_pred_cat, zero_division=0, output_dict=True
    )

    # ---- Importance metrics (threshold = 0.5)
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
    # Guard against single-class edge cases
    try:
        imp_roc = float(roc_auc_score(y_true_imp, p1))
    except ValueError:
        imp_roc = float("nan")
    try:
        imp_pr_auc = float(average_precision_score(y_true_imp, p1))
    except ValueError:
        imp_pr_auc = float("nan")

    # ---- Confusion matrices
    cm_cat = confusion_matrix(y_true_cat, y_pred_cat, labels=np.unique(y_true_cat))
    cm_imp = confusion_matrix(y_true_imp, y_pred_imp, labels=[0, 1])

    # Save CSVs (keeps parity with your current script)
    pd.DataFrame(
        cm_cat, index=np.unique(y_true_cat), columns=np.unique(y_true_cat)
    ).to_csv(os.path.join(outdir, "confusion_category.csv"))
    pd.DataFrame(
        cm_imp,
        index=["not_important", "important"],
        columns=["pred_not_important", "pred_important"],
    ).to_csv(os.path.join(outdir, "confusion_important.csv"))

    # ---- PLOTS

    # 1) Confusion Matrix Heatmap - Category
    plt.figure()
    im = plt.imshow(cm_cat, interpolation="nearest")
    plt.title("Confusion Matrix — Category")
    plt.colorbar(im)
    classes = list(np.unique(y_true_cat))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    # annotate cells
    thresh = cm_cat.max() / 2.0 if cm_cat.size else 0
    for i in range(cm_cat.shape[0]):
        for j in range(cm_cat.shape[1]):
            plt.text(
                j,
                i,
                format(cm_cat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_cat[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(outdir, "confusion_category.png"), bbox_inches="tight")
    plt.close()

    # 2) Confusion Matrix Heatmap — Importance
    plt.figure()
    im = plt.imshow(cm_imp, interpolation="nearest")
    plt.title("Confusion Matrix — Importance")
    plt.colorbar(im)
    plt.xticks([0, 1], ["pred_not_important", "pred_important"], rotation=20)
    plt.yticks([0, 1], ["not_important", "important"])
    thresh = cm_imp.max() / 2.0 if cm_imp.size else 0
    for i in range(cm_imp.shape[0]):
        for j in range(cm_imp.shape[1]):
            plt.text(
                j,
                i,
                format(cm_imp[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_imp[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(outdir, "confusion_important.png"), bbox_inches="tight")
    plt.close()

    # 3) ROC curve — Importance (if meaningful)
    if len(np.unique(y_true_imp)) == 2:
        plt.figure()
        RocCurveDisplay.from_predictions(y_true_imp, p1)
        plt.title("ROC Curve — Importance")
        plt.savefig(
            os.path.join(outdir, "roc_curve_importance.png"), bbox_inches="tight"
        )
        plt.close()

        # 4) Precision–Recall curve — Importance
        plt.figure()
        PrecisionRecallDisplay.from_predictions(y_true_imp, p1)
        plt.title("Precision–Recall Curve — Importance")
        plt.savefig(
            os.path.join(outdir, "pr_curve_importance.png"), bbox_inches="tight"
        )
        plt.close()

    # 5) Per-class F1 bar chart — Category
    per_class = []
    labels = []
    for k, v in cat_report.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        # k is the class label
        if isinstance(v, dict) and "f1-score" in v:
            labels.append(str(k))
            per_class.append(v["f1-score"])
    if labels:
        order = np.argsort(per_class)[::-1]
        labels_ordered = [labels[i] for i in order]
        scores_ordered = [per_class[i] for i in order]

        plt.figure()
        plt.bar(range(len(scores_ordered)), scores_ordered)
        plt.xticks(range(len(scores_ordered)), labels_ordered, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("F1")
        plt.title("Per-class F1 — Category")
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, "per_class_f1_category.png"), bbox_inches="tight"
        )
        plt.close()

    # ---- Output summary (JSON) to stdout and file
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
            "roc_auc": imp_roc,
            "pr_auc": imp_pr_auc,
        },
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--true",
        default="outputs/modernbert_multitask/splits/test.csv",
        help="Path to ground-truth CSV",
    )
    parser.add_argument(
        "--pred",
        default="./test_split_predictions_multitask.csv",
        help="Path to predictions CSV",
    )
    parser.add_argument(
        "--outdir", default=".", help="Where to write CSVs/PNGs/metrics.json"
    )
    args = parser.parse_args()
    main(args.true, args.pred, args.outdir)
