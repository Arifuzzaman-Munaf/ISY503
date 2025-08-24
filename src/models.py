import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertForClassification(nn.Module):
    """
    DistilBERT-based text classifier with masked mean-pooling and a 2-layer MLP head.
    Designed for binary or multi-class sentiment classification.
    """
    def __init__(self,
                 n_classes: int = 2,
                 dropout: float = 0.4,        # dropout rate for stronger regularization
                 use_mean_pool: bool = True,  # choose between mean pooling or [CLS] token
                 freeze_encoder: bool = False,
                 freeze_n_layers: int = 0):   # number of bottom layers to freeze
        super().__init__()

        # load pretrained DistilBERT encoder
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # option to freeze the entire encoder
        if freeze_encoder:
            for p in self.bert.parameters():
                p.requires_grad = False

        # option to partially freeze bottom N transformer layers (0–6 in DistilBERT)
        if not freeze_encoder and freeze_n_layers > 0:
            n = min(max(freeze_n_layers, 0), len(self.bert.transformer.layer))
            for i in range(n):
                for p in self.bert.transformer.layer[i].parameters():
                    p.requires_grad = False

        h = self.bert.config.hidden_size  # hidden dimension size (768)

        # classification head: normalization + 2-layer MLP with dropout and GELU activation
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Dropout(dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, n_classes),
        )

        self.use_mean_pool = use_mean_pool

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None):
        """
        Compute masked mean-pooling across tokens.
        If attention_mask is provided, padding tokens are excluded from the average.
        """
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)  # fallback: unmasked mean

        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)   # expand mask to [B, L, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                   # sum over sequence length
        denom = mask.sum(dim=1).clamp(min=1e-6)                          # avoid division by zero
        return summed / denom

    def forward(self, input_ids, attention_mask=None):
        # run inputs through DistilBERT encoder
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state  # token-level embeddings [B, L, H]

        # pool token embeddings to a single vector per sequence
        if self.use_mean_pool:
            pooled = self._mean_pool(token_emb, attention_mask)          # masked mean pooling
        else:
            pooled = token_emb[:, 0, :]                                  # use first token ([CLS])

        # pass pooled representation through classification head
        logits = self.head(pooled)                                       # class logits [B, C]
        return logits