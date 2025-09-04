# save as infer_email_multitask.py
import argparse, json, torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch.nn as nn


class MultiTaskEmailModel(nn.Module):
    def __init__(self, model_name: str, num_cat: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier_category = nn.Linear(hidden, num_cat)
        # importance head now takes [CLS] + category softmax (num_cat features)
        self.classifier_important = nn.Linear(hidden + num_cat, 2)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        h = self.dropout(cls)
        logits_cat = self.classifier_category(h)
        # classifier-chain: feed soft category evidence into importance
        pc = torch.softmax(logits_cat, dim=-1)  # [batch, num_cat]
        h_imp = torch.cat([h, pc], dim=-1)  # [batch, hidden + num_cat]
        logits_imp = self.classifier_important(h_imp)
        return logits_cat, logits_imp


def load_texts(df, text_col):
    if text_col and text_col in df.columns:
        return df[text_col].astype(str)
    subj = df["subject"].fillna("").astype(str) if "subject" in df.columns else ""
    body = df["body"].fillna("").astype(str) if "body" in df.columns else ""
    return (subj + "\n\n" + body).str.strip()


@torch.no_grad()
def predict(model, tok, texts, device, max_length=512, batch_size=64):
    ycat_ids, ycat_probs, yimp_ids, yimp_p1 = [], [], [], []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(
            list(chunk),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits_cat, logits_imp = model(**enc)
            pc = torch.softmax(logits_cat, dim=-1)
            pi = torch.softmax(logits_imp, dim=-1)
        ycat_ids.extend(pc.argmax(dim=-1).tolist())
        ycat_probs.extend(pc.tolist())
        yimp_ids.extend(pi.argmax(dim=-1).tolist())
        yimp_p1.extend(pi[:, 1].tolist())
    return ycat_ids, ycat_probs, yimp_ids, yimp_p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/modernbert_multitask")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out-csv", default="email_predictions_multitask.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = json.load(open(f"{args.model_dir}/multitask_config.json"))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])

    num_cat = cfg["num_category"]
    model = MultiTaskEmailModel(cfg["base_model"], num_cat=num_cat)
    state = torch.load(
        f"{args.model_dir}/pytorch_model_multitask.bin", map_location="cpu"
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    df = pd.read_csv(args.csv)
    texts = load_texts(df, args.text_col).tolist()

    ycat_ids, ycat_probs, yimp_ids, yimp_p1 = predict(
        model,
        tok,
        texts,
        device,
        max_length=cfg["max_length"],
        batch_size=args.batch_size,
    )
    id2label_cat = {int(k): v for k, v in cfg["id2label_category"].items()}
    cat_labels = [id2label_cat[int(i)] for i in ycat_ids]
    imp_labels = ["important" if p >= 0.5 else "not_important" for p in yimp_p1]

    out = df.copy()
    out["pred_category"] = cat_labels
    out["pred_category_probs"] = ycat_probs
    out["pred_important"] = imp_labels
    out["pred_important_p"] = yimp_p1
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
