"""Frozen ViT (BeiT) feature extractor.

Hard requirement from the project spec: adaptation never backpropagates through
the backbone. We enforce that at the source — every parameter is frozen and every
forward pass runs under `torch.no_grad()` — rather than merely relying on callers
to not call `.backward()`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from transformers import BeitImageProcessor, BeitModel


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenBeit:
    """Wraps a pretrained BeiT model as a frozen feature extractor.

    Pooling strategy:
      - "cls" (default): the CLS token embedding — the standard ViT-family
        image representation, simple and well-suited to a nearest-prototype
        classifier.
      - "mean_patch": mean over patch-token embeddings, offered as a
        configurable alternative; not the default because CLS pooling is what
        BeiT-family checkpoints are typically evaluated with.
    """

    def __init__(
        self,
        model_name: str = "microsoft/beit-base-patch16-224",
        pooling: str = "cls",
        device: str = "auto",
    ):
        if pooling not in {"cls", "mean_patch"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.device = resolve_device(device)
        self.processor = BeitImageProcessor.from_pretrained(model_name)
        self.model = BeitModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def embed_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        hidden = out.last_hidden_state  # (B, 1 + num_patches, D)
        if self.pooling == "cls":
            embeddings = hidden[:, 0, :]
        else:
            embeddings = hidden[:, 1:, :].mean(dim=1)
        return embeddings.detach().cpu()
