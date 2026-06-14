"""
Foundation model encoder wrappers and lightweight head architectures for SHL 2026.

Encoders (all frozen):
  moment   -- MOMENT-1-large (341M params). Requires: pip install momentfm
  fallback -- Frozen random projection (NOT a real foundation model; placeholder only)

Alternative encoders (chronos, uni2ts) are NOT installed in this environment.
Attempted import is documented in get_encoder() with a clear error message.

Input convention for ALL encoders: (B, C, T) float32  (batch, channels, time)
Output:                             (B, embed_dim) float32

Embedding strategies for MomentEncoder
---------------------------------------
mean_pool      : mean over all patch tokens → (B, 1024)
last_patch     : try momentfm reduction=None for last token; fall back to mean of
                 last-quarter time segment if API doesn't support patch-level output
sensorwise     : run MOMENT on each channel c∈[0,C) as (B, 1, T) independently,
                 then concatenate → (B, C×1024)
flatten_patches: alias of sensorwise (per-channel means concatenated)

Bug fix vs. stage-4 code
-------------------------
The previous MomentEncoder incorrectly did x.permute(0,2,1) on (B,C,T) input,
giving MOMENT a (B,T,C)=(B,500,9) tensor it interpreted as 500 "channels" with
9 time steps each.  The input mask was also (B,9) instead of (B,500).
Both are corrected here: encoder receives (B,C,T) directly; mask is (B,T).

Head architectures (PyTorch, nn.Module)
----------------------------------------
build_pytorch_head(head_type, embed_dim, n_classes, hidden, dropout) → nn.Module

Sklearn-compatible heads (xgb, logistic) are instantiated in the training script.
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


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class MomentEncoder(nn.Module):
    """Frozen MOMENT-1-large backbone.

    Parameters
    ----------
    embed_strategy : "mean_pool" | "last_patch" | "sensorwise" | "flatten_patches"
    n_channels     : number of input channels (9 base, +3 if magnitude, +C if delta)
    """

    is_placeholder: bool = False

    def __init__(
        self,
        embed_strategy: str = "mean_pool",
        n_channels: int = 9,
    ) -> None:
        super().__init__()
        from momentfm import MOMENTPipeline
        self._pipeline = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )
        self._pipeline.half()
        _freeze(self._pipeline)
        self._pipeline.eval()
        self._strategy = embed_strategy
        self._n_channels = n_channels
        self.embed_dim = self._analytical_embed_dim()

    def _analytical_embed_dim(self) -> int:
        base = MOMENT_EMBED_DIM
        if self._strategy in ("mean_pool", "last_patch"):
            return base
        if self._strategy in ("sensorwise", "flatten_patches"):
            return self._n_channels * base
        raise ValueError(f"Unknown embed_strategy {self._strategy!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) float32 → (B, embed_dim) float32"""
        B, C, T = x.shape

        if self._strategy == "mean_pool":
            x_enc = x.half()
            mask  = torch.ones(B, T, dtype=torch.float16, device=x.device)
            out   = self._pipeline.embed(x_enc=x_enc, input_mask=mask,
                                         reduction="mean")
            return out.embeddings.float()

        elif self._strategy == "last_patch":
            x_enc = x.half()
            mask  = torch.ones(B, T, dtype=torch.float16, device=x.device)
            try:
                out = self._pipeline.embed(x_enc=x_enc, input_mask=mask,
                                           reduction=None)
                emb = out.embeddings
                if emb.dim() == 3:        # (B, n_patches, D) — got patch tokens
                    return emb[:, -1, :].float()
                return emb.float()        # API returned mean anyway
            except Exception:
                # Fallback: mean-pool the last quarter of the time series
                T4 = max(T // 4, 1)
                x_last = x[:, :, -T4:].half()
                mask_last = torch.ones(B, T4, dtype=torch.float16, device=x.device)
                out = self._pipeline.embed(x_enc=x_last, input_mask=mask_last,
                                           reduction="mean")
                return out.embeddings.float()

        elif self._strategy in ("sensorwise", "flatten_patches"):
            return self._sensorwise_forward(x)

        else:
            raise ValueError(f"Unknown embed_strategy {self._strategy!r}")

    def _sensorwise_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process each channel independently; concat → (B, C*1024)."""
        B, C, T = x.shape
        embs = []
        for c in range(C):
            x_c    = x[:, c : c + 1, :].half()          # (B, 1, T)
            mask_c = torch.ones(B, T, dtype=torch.float16, device=x.device)
            out    = self._pipeline.embed(x_enc=x_c, input_mask=mask_c,
                                          reduction="mean")
            embs.append(out.embeddings.float())           # (B, 1024)
        return torch.cat(embs, dim=1)                     # (B, C*1024)


class FallbackEncoder(nn.Module):
    """Frozen orthogonal random projection — NOT a real foundation model.

    Accepts variable-channel inputs by flattening (B, C, T) → (B, C*T).
    Fixed seed ensures reproducibility across runs.
    """

    is_placeholder: bool = True

    def __init__(self, n_channels: int = 9, T: int = 500) -> None:
        super().__init__()
        in_dim = n_channels * T
        self.proj = nn.Linear(in_dim, FALLBACK_EMBED_DIM, bias=False)
        rng_state = torch.get_rng_state()
        torch.manual_seed(0)
        nn.init.orthogonal_(self.proj.weight)
        torch.set_rng_state(rng_state)
        _freeze(self.proj)
        self.embed_dim = FALLBACK_EMBED_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, embed_dim)"""
        return self.proj(x.float().reshape(x.shape[0], -1))


# ---------------------------------------------------------------------------
# Head architectures (PyTorch)
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLPHead(nn.Module):
    """Two-block residual MLP with LayerNorm.

    embed_dim → hidden → [Residual×2] → n_classes
    """

    def __init__(self, embed_dim: int, hidden: int, n_classes: int,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            _ResidualBlock(hidden, dropout),
            _ResidualBlock(hidden, dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_pytorch_head(
    head_type: str,
    embed_dim: int,
    n_classes: int,
    hidden: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    """Return a lightweight PyTorch head.

    head_type : "linear" | "mlp" | "residual_mlp"
    """
    if head_type == "linear":
        return nn.Linear(embed_dim, n_classes)

    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    if head_type == "residual_mlp":
        return ResidualMLPHead(embed_dim, hidden, n_classes, dropout)

    raise ValueError(f"Unknown pytorch head {head_type!r} — choose linear/mlp/residual_mlp")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_encoder(
    name: str,
    embed_strategy: str = "mean_pool",
    n_channels: int = 9,
    T: int = 500,
) -> nn.Module:
    """Return the requested frozen encoder.

    Parameters
    ----------
    name           : "moment" | "fallback" | "chronos" | "uni2ts"
    embed_strategy : embedding strategy (moment only)
    n_channels     : input channel count after preprocessing
    T              : time-series length (samples per window)
    """
    if name == "moment":
        return MomentEncoder(embed_strategy=embed_strategy, n_channels=n_channels)

    if name == "fallback":
        return FallbackEncoder(n_channels=n_channels, T=T)

    if name == "chronos":
        raise ImportError(
            "chronos encoder requested but 'chronos-forecasting' is not installed.\n"
            "  pip install chronos-forecasting\n"
            "Chronos is a forecasting model; embedding extraction requires accessing\n"
            "internal T5 encoder hidden states — not supported in this version."
        )

    if name == "uni2ts":
        raise ImportError(
            "uni2ts encoder requested but 'uni2ts' (Moirai) is not installed.\n"
            "  pip install uni2ts\n"
            "Moirai is a forecasting model; embedding extraction is not directly\n"
            "supported without custom forward-pass modifications."
        )

    raise ValueError(f"Unknown encoder {name!r} — choose moment / fallback")
