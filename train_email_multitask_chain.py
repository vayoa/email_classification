#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train ONE multitask model with a shared encoder and two heads:
- category (multiclass)
- important (binary)

Why: lets the 'important' head learn signals correlated with category via shared features.
Saves to: outputs/modernbert_multitask
"""

import argparse, json, os, random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -----------------------------
# Utils
# -----------------------------


def infer_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16  # fp16 works well on 30xx
    if torch.backends.mps.is_available():
        return "mps", None
    return "cpu", None


def prepare_text_column(df: pd.DataFrame, text_col: Optional[str]) -> pd.Series:
    if text_col and text_col in df.columns:
        return df[text_col].astype(str)
    raise ValueError(f"Text column '{text_col}' not found in CSV.")


def clean_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")
    s = df[label_col]
    if label_col == "important":
        s = s.map({False: 0, True: 1}).astype(int)
    else:
        s = s.astype(str)
    return s


def compute_class_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_labels)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights * (num_labels / weights.sum())
    return torch.tensor(weights, dtype=torch.float)


# -----------------------------
# Model
# -----------------------------


class MultiTaskEmailModel(nn.Module):
    """
    Shared encoder with two classification heads.
    Saves/loads via state_dict.
    """

    def __init__(self, model_name: str, num_cat: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier_category = nn.Linear(hidden, num_cat)
        self.classifier_important = nn.Linear(hidden + num_cat, 2)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling
        cls = out.last_hidden_state[:, 0]
        h = self.dropout(cls)
        logits_cat = self.classifier_category(h)
        pc = torch.softmax(logits_cat, dim=-1)  # [batch, num_cat]
        h_imp = torch.cat([h, pc], dim=-1)  # [batch, hidden + num_cat]
        logits_imp = self.classifier_important(h_imp)
        return {"logits": (logits_cat, logits_imp)}


# -----------------------------
# Dataset
# -----------------------------


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self, encodings: Dict[str, torch.Tensor], y_cat: List[int], y_imp: List[int]
    ):
        self.enc = encodings
        self.yc = y_cat
        self.yi = y_imp

    def __len__(self):
        return len(self.yc)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels_category"] = torch.tensor(self.yc[idx], dtype=torch.long)
        item["labels_important"] = torch.tensor(self.yi[idx], dtype=torch.long)
        return item


# -----------------------------
# Trainer
# -----------------------------


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        *args,
        cw_cat: Optional[torch.Tensor] = None,
        cw_imp: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cw_cat = cw_cat
        self.cw_imp = cw_imp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_cat = inputs.pop("labels_category")
        labels_imp = inputs.pop("labels_important")
        outputs = model(**inputs)
        logits_cat, logits_imp = outputs["logits"]
        if self.cw_cat is not None:
            self.cw_cat = self.cw_cat.to(logits_cat.device)
        if self.cw_imp is not None:
            self.cw_imp = self.cw_imp.to(logits_imp.device)
        loss_cat = nn.CrossEntropyLoss(weight=self.cw_cat)(logits_cat, labels_cat)
        loss_imp = nn.CrossEntropyLoss(weight=self.cw_imp)(logits_imp, labels_imp)
        loss = loss_cat + loss_imp  # equal weights; change if desired
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Metrics
# -----------------------------


def multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "weighted_f1": wf1,
        "macro_precision": p,
        "macro_recall": r,
    }


def binary_metrics(y_true, y_pred, p1=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = float("nan")
    if p1 is not None:
        try:
            auc = roc_auc_score(y_true, p1)
        except Exception:
            pass
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}


def make_splits_and_save(
    df,
    texts,  # pd.Series or list (the text column already prepared)
    y_cat,  # list/np.ndarray of category ids
    y_imp,  # list/np.ndarray of important (0/1)
    val_size: float,
    test_size: float,
    seed: int,
    output_dir: str,
):
    """
    3-way split (train/val/test) stratified by category only.
    Saves CSVs + index arrays under {output_dir}/splits and returns train/val arrays for training.
    Test split is not returned (left untouched for later evaluation).
    """
    import numpy as np, os
    from sklearn.model_selection import train_test_split

    assert (
        0 < val_size < 1 and 0 < test_size < 1 and (val_size + test_size) < 1
    ), "val_size + test_size must be < 1"

    n = len(df)
    idx_all = np.arange(n)
    strat = np.array(y_cat)

    # 1) train vs holdout (val+test)
    holdout_size = val_size + test_size
    train_idx, hold_idx = train_test_split(
        idx_all,
        test_size=holdout_size,
        random_state=seed,
        stratify=strat,
    )

    # 2) holdout -> val vs test
    val_prop_within_hold = val_size / holdout_size
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=(1.0 - val_prop_within_hold),
        random_state=seed,
        stratify=strat[hold_idx],
    )

    # Save splits
    split_dir = os.path.join(output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    df.iloc[train_idx].to_csv(os.path.join(split_dir, "train.csv"), index=False)
    df.iloc[val_idx].to_csv(os.path.join(split_dir, "val.csv"), index=False)
    df.iloc[test_idx].to_csv(os.path.join(split_dir, "test.csv"), index=False)

    np.save(os.path.join(split_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(split_dir, "val_indices.npy"), val_idx)
    np.save(os.path.join(split_dir, "test_indices.npy"), test_idx)

    # Materialize arrays for training (test is kept aside)
    def take(seq, idxs):
        if hasattr(seq, "iloc"):  # pandas Series
            return seq.iloc[idxs].tolist()
        seq_list = seq.tolist() if hasattr(seq, "tolist") else list(seq)
        return [seq_list[i] for i in idxs]

    X_train = take(texts, train_idx)
    X_val = take(texts, val_idx)
    ycat_tr = take(y_cat, train_idx)
    ycat_va = take(y_cat, val_idx)
    yimp_tr = take(y_imp, train_idx)
    yimp_va = take(y_imp, val_idx)

    return X_train, X_val, ycat_tr, ycat_va, yimp_tr, yimp_va


# -----------------------------
# Train
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="outputs/modernbert_multitask")
    ap.add_argument("--no-weighted-loss", action="store_true")
    ap.add_argument("--val-size", type=float, default=0.15, help="validation fraction")
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="test fraction (held out, never used in training)",
    )
    args = ap.parse_args()
    assert (
        0 < args.val_size < 1
        and 0 < args.test_size < 1
        and args.val_size + args.test_size < 1
    ), "val_size + test_size must be < 1"

    MODEL_NAME = "answerdotai/ModernBERT-base"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    df = pd.read_csv(args.csv)
    texts = prepare_text_column(df, args.text_col)
    df["category"] = clean_labels(df, "category")
    df["important"] = clean_labels(df, "important")

    # Labels
    cat_labels_sorted = sorted(df["category"].astype(str).unique())
    label2id_cat = {lab: i for i, lab in enumerate(cat_labels_sorted)}
    id2label_cat = {i: lab for lab, i in label2id_cat.items()}
    y_cat = df["category"].map(label2id_cat).tolist()
    y_imp = df["important"].astype(int).tolist()

    # Split our dataset to train/validation and test (saved to disk) for hyperparameter tuning.
    # Notice we stratify on category. Our dataset is big enough to support important without stratifying on it + each category will have different importance freq (e.g. spam has 0 important emails).
    X_train, X_val, ycat_tr, ycat_va, yimp_tr, yimp_va = make_splits_and_save(
        df=df,
        texts=texts,
        y_cat=y_cat,
        y_imp=y_imp,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # Tokenize
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    enc_tr = tok(
        X_train,
        truncation=True,
        max_length=args.max_length,
        padding="longest",
        return_tensors="pt",
    )
    enc_va = tok(
        X_val,
        truncation=True,
        max_length=args.max_length,
        padding="longest",
        return_tensors="pt",
    )

    train_ds = MultiTaskDataset(enc_tr, ycat_tr, yimp_tr)
    val_ds = MultiTaskDataset(enc_va, ycat_va, yimp_va)

    # Model
    num_cat = len(label2id_cat)
    model = MultiTaskEmailModel(MODEL_NAME, num_cat=num_cat)

    # Loss weights for imbalance
    cw_cat = (
        compute_class_weights(ycat_tr, num_cat) if not args.no_weighted_loss else None
    )
    cw_imp = compute_class_weights(yimp_tr, 2) if not args.no_weighted_loss else None

    device, dtype = infer_device_dtype()

    # Training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_score",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,
        bf16=False,
        fp16=(device == "cuda"),
        warmup_ratio=0.05,
        weight_decay=0.01,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
        label_names=["labels_category", "labels_important"],
    )

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    # Metrics wrapper
    def compute_metrics_fn(p):
        # p.predictions is a tuple: (logits_cat, logits_imp)
        logits_cat, logits_imp = p.predictions

        # Robustly extract y_true for both heads
        ycat_true = yimp_true = None
        lid = p.label_ids
        if isinstance(lid, dict):
            # when Trainer returns a dict keyed by label_names
            ycat_true = lid["labels_category"]
            yimp_true = lid["labels_important"]
        elif isinstance(lid, (list, tuple)) and len(lid) == 2:
            # common in recent HF: a tuple/list per label_name
            ycat_true, yimp_true = lid[0], lid[1]
        else:
            # fallback: single array with columns
            arr = np.asarray(lid)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                ycat_true = arr[:, 0]
                yimp_true = arr[:, 1]
            else:
                raise TypeError(
                    f"Unrecognized label_ids format: type={type(lid)}, shape={getattr(arr,'shape',None)}"
                )

        # Predictions
        ycat_pred = np.argmax(logits_cat, axis=-1)
        yimp_prob = torch.softmax(torch.tensor(logits_imp), dim=-1).numpy()[:, 1]
        yimp_pred = (yimp_prob >= 0.5).astype(int)

        # Metrics
        m_cat = multiclass_metrics(ycat_true, ycat_pred)
        m_imp = binary_metrics(yimp_true, yimp_pred, p1=yimp_prob)

        # Combined score (equal weight). Adjust weights if you care more about one head.
        overall_score = 0.5 * m_cat["macro_f1"] + 0.5 * m_imp["f1"]

        return {
            "overall_score": overall_score,  # <-- key used for best-model tracking
            **{f"cat_{k}": v for k, v in m_cat.items()},
            **{f"imp_{k}": v for k, v in m_imp.items()},
        }

    trainer = MultiTaskTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        cw_cat=cw_cat,
        cw_imp=cw_imp,
        processing_class=tok,
    )

    trainer.train()
    eval_out = trainer.evaluate()

    os.makedirs(args.output_dir, exist_ok=True)
    # Save model weights + assets for reuse
    torch.save(
        model.state_dict(), os.path.join(args.output_dir, "pytorch_model_multitask.bin")
    )
    model_config = {
        "base_model": MODEL_NAME,
        "num_category": num_cat,
        "label2id_category": label2id_cat,
        "id2label_category": id2label_cat,
        "max_length": args.max_length,
    }
    with open(
        os.path.join(args.output_dir, "multitask_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(model_config, f, indent=2)
    with open(
        os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(eval_out, f, indent=2)

    print(json.dumps({"output_dir": args.output_dir, **eval_out}, indent=2))


if __name__ == "__main__":
    main()
