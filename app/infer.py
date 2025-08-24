# app/infer.py
from __future__ import annotations
import os, sys, glob
from pathlib import Path
import typing as t

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# add repo root so "src" can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models import DistilBertForClassification

# -------- Config --------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_LOC = (Path(__file__).resolve().parents[1] / "saved_models").resolve()
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_LOC))


# -------- Utilities --------
def list_checkpoints(directory: str | Path) -> list[Path]:
    """Return .pth checkpoints in a directory, newest first."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve_checkpoint(path: str | Path) -> Path:
    """Return a valid .pth path (if dir, pick newest)."""
    p = Path(path)
    if p.is_file() and p.suffix == ".pth":
        return p
    if p.is_dir():
        files = list_checkpoints(p)
        if files:
            print(f"[INFO] Using checkpoint: {files[0]}")
            return files[0]
    raise FileNotFoundError(f"No .pth checkpoint found at: {p}")


# -------- Model wrapper --------
class SentimentModel:
    def __init__(self, model_path: str | Path | None = None, num_classes: int = 2):
        self.model_path = _resolve_checkpoint(model_path or MODEL_PATH)
        self.num_classes = num_classes
        self.id2label = {0: "Negative comment", 1: "Positive comment"}
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = None

    def load(self) -> "SentimentModel":
        self.model = DistilBertForClassification(n_classes=self.num_classes)
        state = torch.load(str(self.model_path), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(DEVICE).eval()
        return self

    @torch.inference_mode()
    def predict(self, texts: t.List[str], max_len: int = 256):
        if not texts:
            return [], []
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        logits = self.model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs = F.softmax(logits, dim=-1).cpu().numpy().tolist()
        preds = [int(max(range(len(row)), key=row.__getitem__)) for row in probs]
        return preds, probs


# -------- Cache --------
_CACHE: dict[str, SentimentModel] = {}

def get_model(model_path: str | None = None) -> SentimentModel:
    resolved = str(_resolve_checkpoint(model_path or MODEL_PATH))
    if resolved not in _CACHE:
        _CACHE[resolved] = SentimentModel(resolved).load()
    return _CACHE[resolved]