"""
Foundation model encoder wrappers for SHL 2026.

Two encoders:
  moment   -- Frozen MOMENT-1-large (341M params, 1024-dim embeddings).
               Requires: pip install momentfm
  fallback -- Frozen random projection (NOT a real foundation model).
               Reproducible placeholder when MOMENT is unavailable.

Both accept (B, T, C) = (B, 500, 9) and return (B, embed_dim) float32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MOMENT_EMBED_DIM   = 1024
FALLBACK_EMBED_DIM = 1024


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


class MomentEncoder(nn.Module):
    """Frozen MOMENT-1-large backbone.  Input (B, T, C) -> output (B, 1024)."""

    embed_dim: int = MOMENT_EMBED_DIM
    is_placeholder: bool = False

    def __init__(self) -> None:
        super().__init__()
        from momentfm import MOMENTPipeline
        self._pipeline = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )
        # Convert to FP16 to halve model weight memory (~9.6 GiB -> ~4.8 GiB),
        # freeing ~5 GiB for larger batches during extraction.
        self._pipeline.half()
        _freeze(self._pipeline)
        self._pipeline.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) with T=500, C=9  ->  (B, 1024) float32"""
        # MOMENT expects (B, C, T); run in FP16 to match model weights
        x_enc = x.permute(0, 2, 1).half()
        mask  = torch.ones(x.shape[0], x.shape[1],
                           dtype=torch.float16, device=x.device)
        out = self._pipeline.embed(x_enc=x_enc, input_mask=mask, reduction="mean")
        return out.embeddings.float()   # cast back to FP32 for downstream use


class FallbackEncoder(nn.Module):
    """
    NOT a real foundation model.

    Frozen orthogonal random projection:
    (B, 500, 9) -> flatten (B, 4500) -> project (B, 1024).
    Fixed seed=0 ensures reproducibility.  Use only as a placeholder.
    """

    embed_dim: int = FALLBACK_EMBED_DIM
    is_placeholder: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(500 * 9, FALLBACK_EMBED_DIM, bias=False)
        # Deterministic init: save/restore RNG state so we don't pollute training
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        nn.init.orthogonal_(self.proj.weight)
        torch.set_rng_state(rng_state)
        _freeze(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 1024)"""
        return self.proj(x.float().reshape(x.shape[0], -1))


def get_encoder(name: str) -> nn.Module:
    """
    Factory returning the requested frozen encoder.

    Parameters
    ----------
    name : "moment" | "fallback"
    """
    if name == "moment":
        return MomentEncoder()
    if name == "fallback":
        return FallbackEncoder()
    raise ValueError(f"Unknown encoder {name!r} -- choose 'moment' or 'fallback'")
