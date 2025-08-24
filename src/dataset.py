import os
import tarfile
import urllib.request
import tarfile, urllib.request
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def download_and_extract_dataset(cfg):
    os.makedirs(cfg.data_dir, exist_ok=True)
    tar_path = Path(cfg.data_dir) / "domain_sentiment_data.tar.gz"

    if not tar_path.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(cfg.data_url, tar_path)
    else:
        print("Dataset archive already exists, skipping download.")

    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=cfg.data_dir)
    print(f"Dataset extracted to: {cfg.data_dir}")

    # Return actual extracted folder path
    candidates = [
        Path(cfg.data_dir) / "domain_sentiment_data",
        Path(cfg.data_dir) / "sentiment" / "domain_sentiment_data",
        Path(cfg.data_dir) / "sorted_data_acl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(cfg.data_dir)


def _read_reviews(file_path, label):
    """
    Each review file has one review per blank-line-separated block.
    Some files are encoded latin-1; fall back if utf-8 fails.
    """
    encodings = ("utf-8", "latin-1")
    text = None
    for enc in encodings:
        try:
            text = file_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("Could not decode file", str(file_path), 0, 0, "encoding failed")

    blocks, buf = [], []
    for line in text.splitlines():
        if line.strip() == "":
            if buf:
                blocks.append(" ".join(buf).strip())
                buf = []
        else:
            buf.append(line.strip())
    if buf:
        blocks.append(" ".join(buf).strip())

    if label is None:
        label_value = -1  # unlabeled
    else:
        label_value = int(label)

    return [{"text": r, "label": label_value} for r in blocks if r]

def load_multi_domain_dataframe(root, include_unlabeled = True):
    """
    Expected structure (one of the common layouts):
      root/
        books/positive.review
        books/negative.review
        books/unlabeled.review   (optional)
        dvd/...
        electronics/...
        kitchen/...
    """
    rows = []
    domains = []
    for domain_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        domain = domain_dir.name
        pos = domain_dir / "positive.review"
        neg = domain_dir / "negative.review"
        unl = domain_dir / "unlabeled.review"

        if pos.exists():
            rows.extend(_read_reviews(pos, label=1))
            domains.extend([domain] * len(rows[-len(_read_reviews(pos, 0)):]))
        if neg.exists():
            rows_neg = _read_reviews(neg, label=0)
            rows.extend(rows_neg)
            domains.extend([domain] * len(rows_neg))
        if include_unlabeled and unl.exists():
            rows_unl = _read_reviews(unl, label=None)
            rows.extend(rows_unl)
            domains.extend([domain] * len(rows_unl))

    df = pd.DataFrame(rows)
    df["domain"] = domains[:len(df)]  # align lengths
    return df


class TextDataset(Dataset):
    def __init__(self, df, sentence_len = 128,
        label_col= "label",
        text_col = None,
        model_name = 'distilbert-base-uncased'
    ):
        """
        Custom Dataset for binary sentiment classification (0=negative, 1=positive).
        Pre-tokenizes the text data using a HuggingFace tokenizer.

        args
        df : DataFrame containing labels and text data
        sentence_len : maximum sequence length for tokenization (default=128)
        label_col : name of the label column (default="label")
        text_col : name of the text column (if None, the longest column is selected)
        model_name : HuggingFace model name for tokenizer (default=distilbert-base-uncased)
        """

        # If no text column is specified, pick the one with the longest average length
        if text_col is None:
            column_length = {
                col: df[col].astype(str).str.len().mean()
                for col in df.columns if col != label_col
            }
            if not column_length:
                raise ValueError("No candidate text column found; specify text_col.")
            text_col = max(column_length, key=column_length.get)

        # Extract labels (assumed to be 0=negative, 1=positive)
        self.labels = torch.tensor(df[label_col].astype(int).tolist(), dtype=torch.long)

        # Extract text values
        texts = df[text_col].astype(str).tolist()

        # Load HuggingFace tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Tokenize all texts at once
        enc = tokenizer(
            texts,
            max_length=sentence_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",  # directly as PyTorch tensors
        )

        # Store tokenized inputs
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]


    def __len__(self):
        # Return dataset size
        return self.labels.size(0)

    def __getitem__(self, index):
        # Return model inputs, masks and label for a given index
        return self.input_ids[index], self.attention_mask[index], self.labels[index]
        
