import os, torch, numpy as np
import warnings

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging as hf_logging
from datasets import load_dataset, Dataset

hf_logging.set_verbosity_error()
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class TextModel:
    def __init__(self, model_dir="./roberta_phish", model_name="roberta-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def predict(self, text):
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0,1].item()
        return {"score": probs}

    # --------------------------
    # quick HuggingFace Trainer fine-tune wrapper
    # --------------------------
    def finetune(self, train_csv, val_csv, output_dir="./roberta_phish", batch_size=8, epochs=3, lr=2e-5):
        # load CSVs with columns 'text' and 'label'
        train_ds = load_dataset("csv", data_files=train_csv)["train"]
        val_ds = load_dataset("csv", data_files=val_csv)["train"]
        def preprocess(ex):
            return self.tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
        train_ds = train_ds.map(preprocess, batched=True)
        val_ds = val_ds.map(preprocess, batched=True)
        train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
        val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size*2,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_roc_auc"
        )
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "roc_auc": roc_auc_score(labels, probs)
            }
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics
        )
        trainer.train()
        trainer.save_model(output_dir)

    # --------------------------
    # Export quick dynamic quantized CPU model (fast)
    # --------------------------
    def export_dynamic_quant(self, out_dir="./roberta_quant"):
        self.model.cpu()
        qmodel = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        qmodel.save_pretrained(out_dir)
        print("Saved dynamic quantized model at:", out_dir)


# -----------------------------------------------------------------------------
# ✅ Add this wrapper for compatibility with app.py
# -----------------------------------------------------------------------------
class TextScorer(TextModel):
    """Simple wrapper to keep app.py compatibility with .score() calls"""
    def score(self, text: str) -> dict:
        out = self.predict(text)
        return {"score": float(out.get("score", 0.0)), "reasons": [], "components": out}

__all__ = ["TextModel", "TextScorer"]
