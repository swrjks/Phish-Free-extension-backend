"""
backend/cnn_model.py
Fully integrated CLIP-based visual phishing detector (PhishFree-compatible).
"""

from __future__ import annotations
import os, io, math, json, numpy as np
import warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from typing import Dict, Any, Optional, List
from PIL import Image
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor, logging as hf_logging

# Suppress HuggingFace "Loading weights" and "UNEXPECTED keys" logs
hf_logging.set_verbosity_error()

# -------------------------
# Device + Global CLIP model
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

print(f"[CNNModel] Loading HuggingFace CLIP model '{CLIP_MODEL_ID}' on {DEVICE} ...")
_clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
_clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
_clip_model.eval()

try:
    CLIP_EMBED_DIM = _clip_model.config.projection_dim
except Exception:
    with torch.no_grad():
        dummy = _clip_model.get_image_features(
            **_clip_processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").to(DEVICE)
        )
    CLIP_EMBED_DIM = dummy.shape[-1]

print(f"[CNNModel] CLIP embed dimension: {CLIP_EMBED_DIM}")

# -------------------------
# MLP Head for phishing vs benign classification
# -------------------------
class HeadMLP(nn.Module):
    def __init__(self, embed_dim: int = CLIP_EMBED_DIM, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# -------------------------
# CNNModel main class
# -------------------------
class CNNModel:
    def __init__(self,
                 device: Optional[str] = None,
                 clip_model_name: str = CLIP_MODEL_ID,
                 head_ckpt: Optional[str] = None,
                 brand_templates: Optional[Dict[str, np.ndarray]] = None):
        self.device = device or DEVICE
        self.clip_model = _clip_model
        self.clip_processor = _clip_processor
        self.clip_model.eval()

        # head network
        self.head = HeadMLP(CLIP_EMBED_DIM).to(self.device)
        if head_ckpt and os.path.exists(head_ckpt):
            try:
                self.head.load_state_dict(torch.load(head_ckpt, map_location=self.device))
                print(f"[CNNModel] Loaded head checkpoint from {head_ckpt}")
            except Exception as e:
                print("[CNNModel] ⚠️ Failed to load head checkpoint:", e)

        self.criterion_bce = nn.BCEWithLogitsLoss()
        self.brand_templates = brand_templates or {}
        self.temp = 0.07  # for contrastive similarity (if training)

    # -------------------------
    # Embedding functions
    # -------------------------
    def embed_pil(self, pil: Image.Image) -> np.ndarray:
        """Compute normalized CLIP embedding for PIL image."""
        # Resize image to standard size for faster processing
        if pil.size != (224, 224):
            pil = pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        inputs = self.clip_processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
            if not isinstance(emb, torch.Tensor):
                if hasattr(emb, "image_embeds"):
                    emb = emb.image_embeds
                elif hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                else:
                    emb = emb[0]
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy()

    def embed_from_bytes(self, img_bytes: bytes) -> np.ndarray:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.embed_pil(pil)

    def embed_from_path(self, path: str) -> np.ndarray:
        pil = Image.open(path).convert("RGB")
        return self.embed_pil(pil)

    # -------------------------
    # Prediction / Scoring
    # -------------------------
    def predict_from_pil(self, pil: Image.Image) -> Dict[str, Any]:
        """Return phishing probability + brand similarity map."""
        emb_np = self.embed_pil(pil)
        emb_t = torch.tensor(emb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.head(emb_t).squeeze(-1)
            prob = float(torch.sigmoid(logits).cpu().numpy())

        # Compare with brand templates if available
        sims = {}
        if self.brand_templates:
            for brand, ve in self.brand_templates.items():
                sims[brand] = float(np.dot(emb_np, ve))
            best_brand, best_sim = max(sims.items(), key=lambda kv: kv[1])
        else:
            best_brand, best_sim = "", 0.0

        return {
            "score": prob,
            "best_brand": best_brand,
            "best_sim": best_sim,
            "sims": sims,
            "reasons": [f"Similarity to {best_brand}: {best_sim:.3f}"] if best_brand else []
        }

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Convenience wrapper for image bytes (used by Flask backend)."""
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict_from_pil(pil)

    # -------------------------
    # Compute brand embeddings
    # -------------------------
    def compute_brand_embeddings(self, templates_dir: str) -> Dict[str, np.ndarray]:
        """Compute and store normalized brand embeddings from template images."""
        templates = {}
        if not os.path.isdir(templates_dir):
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
        for fn in os.listdir(templates_dir):
            path = os.path.join(templates_dir, fn)
            if not os.path.isfile(path):
                continue
            try:
                emb = self.embed_from_path(path)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                name = os.path.splitext(fn)[0]
                templates[name] = emb
            except Exception as e:
                print(f"[CNNModel] ⚠️ Skipped {fn}: {e}")
        self.brand_templates = templates
        print(f"[CNNModel] Loaded {len(templates)} brand templates.")
        return templates

    # -------------------------
    # Optional TorchScript Export
    # -------------------------
    def export_torchscript(self, out_path="./cnn_ts.pt"):
        """Best-effort TorchScript export (for deployment)."""
        class Wrapper(nn.Module):
            def __init__(self, clip_model, head):
                super().__init__()
                if hasattr(clip_model, "vision_model"):
                    self.visual = clip_model.vision_model
                elif hasattr(clip_model, "visual"):
                    self.visual = clip_model.visual
                else:
                    raise RuntimeError("CLIP model missing visual submodule")
                self.projection = getattr(clip_model, "visual_projection", None)
                self.head = head

            def forward(self, x):
                v = self.visual(x)
                if self.projection is not None:
                    emb = self.projection(v)
                else:
                    emb = v
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
                return torch.sigmoid(self.head(emb)).squeeze(-1)

        print("[CNNModel] Exporting TorchScript (best-effort)...")
        wrapper = Wrapper(self.clip_model, self.head).to(self.device).eval()
        ex = torch.randn(1, 3, 224, 224).to(self.device)
        try:
            traced = torch.jit.trace(wrapper, ex, check_trace=False)
            traced.save(out_path)
            print(f"[CNNModel] ✅ TorchScript saved at {out_path}")
        except Exception as e:
            print(f"[CNNModel] ❌ TorchScript export failed: {e}")
   
        # -------------------------
    # Compatibility wrapper for app.py
    # -------------------------
    def score_image_bytes(self, image_bytes: bytes) -> dict:
        """
        Wrapper for app.py compatibility. Converts image bytes to CLIP embedding,
        passes through the MLP head, and returns a standard dict.
        """
        try:
            res = self.predict_from_bytes(image_bytes)
            score = float(res.get("score", 0.0))
            reasons = []
            if "best_brand" in res and res["best_brand"]:
                reasons.append(f"Closest brand: {res['best_brand']} ({res.get('best_sim', 0):.2f})")
            return {
                "ok": True,
                "score": score,
                "reasons": reasons,
                "best_brand": res.get("best_brand", ""),
                "best_sim": res.get("best_sim", 0.0),
                "sims": res.get("sims", {})
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(e), "score": 0.0}

# -------------------------
# CLI test mode
# -------------------------
if __name__ == "__main__":
    import sys, base64
    cnn = CNNModel()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            print("Testing:", path)
            res = cnn.predict_from_bytes(open(path, "rb").read())
            print(json.dumps(res, indent=2))
        else:
            print("File not found:", path)
    else:
        print("Usage: python cnn_model.py <image_path>")
