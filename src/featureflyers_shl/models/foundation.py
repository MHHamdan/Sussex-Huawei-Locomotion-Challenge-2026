"""
Foundation model encoder wrappers and lightweight head architectures for SHL 2026.

Encoders (all frozen):
  moment   -- MOMENT-1-large (341M params). Requires: pip install momentfm
  chronos2 -- Chronos-2 patch-based multivariate FM. Requires: pip install chronos-forecasting
  fallback -- Frozen random projection (NOT a real foundation model; placeholder only)

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

Embedding strategies for Chronos2Encoder
-----------------------------------------
mean_pool  : mean over all variates and patch tokens → (B, d_model)
last_token : global [REG] summary token (index 0), mean over variates → (B, d_model)

Chronos-2 natively processes multivariate input (B, C, T), sharing information
across all 9 IMU channels jointly within each window.

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

MOMENT_EMBED_DIM      = 1024
FALLBACK_EMBED_DIM    = 1024
# Default Chronos-2 model. If this HuggingFace ID is not found (models still
# gated/unreleased), Chronos2Encoder falls back to amazon/chronos-t5-small (v1)
# and processes each IMU channel independently (per-channel fallback mode).
CHRONOS2_DEFAULT_MODEL    = "amazon/chronos-t5-small"
CHRONOS2_NATIVE_MODEL     = "amazon/chronos-t5-small-r2"  # reserved for future use


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


class Chronos2Encoder(nn.Module):
    """Frozen Chronos encoder for window-level embedding extraction.

    Two operating modes — selected automatically at init time:

    1. NATIVE MULTIVARIATE (if a Chronos-2 patch model is found)
       Uses Chronos2Pipeline. All C channels processed jointly per window.
       Input shape to pipeline: (B, C, T).
       Embed output per sample: (C, num_patches+2, d_model) → pooled to (B, d_model).

    2. PER-CHANNEL FALLBACK (default; uses publicly-available Chronos v1 model)
       Uses ChronosPipeline with amazon/chronos-t5-small (d_model=512).
       Each channel processed as an independent univariate series (9 forward passes
       per outer batch). Per-channel embeds are mean-pooled → (B, d_model).
       This is the active mode unless a Chronos-2 model ID is explicitly provided
       and successfully loaded.

    Embedding strategies (both modes)
    ----------------------------------
    mean_pool  : mean over token/patch dimension (and variates in native mode)
    last_token : last token (v1) or [REG] global-summary token at index 0 (v2)

    Device handling
    ---------------
    self.inner_model is the underlying nn.Module registered as a sub-module,
    so .to(device) propagates to the Chronos backbone automatically.

    Install
    -------
    pip install chronos-forecasting
    """

    is_placeholder: bool = False

    def __init__(
        self,
        model_name: str = CHRONOS2_DEFAULT_MODEL,
        embed_strategy: str = "mean_pool",
        n_channels: int = 9,
    ) -> None:
        super().__init__()
        try:
            from chronos import Chronos2Pipeline, ChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "Chronos encoder requires 'chronos-forecasting'.\n"
                "  pip install chronos-forecasting\n"
                f"Original error: {exc}"
            ) from exc

        if embed_strategy not in ("mean_pool", "last_token"):
            raise ValueError(
                f"Chronos2Encoder only supports embed_strategy 'mean_pool' or 'last_token'; "
                f"got {embed_strategy!r}.  Strategies 'sensorwise'/'flatten_patches'/"
                f"'last_patch' are MOMENT-specific and do not apply to Chronos."
            )

        self._strategy   = embed_strategy
        self._n_channels = n_channels
        self._native_mv  = False   # True = native multivariate Chronos-2 mode

        # --- attempt native Chronos-2 multivariate pipeline ---
        if model_name != CHRONOS2_DEFAULT_MODEL:
            try:
                print(f"  Loading Chronos-2 (native multivariate) '{model_name}' ...",
                      flush=True)
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )
                self._native_mv = True
                print(f"  [Chronos2] native multivariate mode active", flush=True)
            except (OSError, ValueError, AssertionError) as exc:
                print(f"  [Chronos2] model '{model_name}' unavailable: {exc}", flush=True)
                print(f"  [Chronos2] falling back to per-channel mode", flush=True)

        # --- per-channel fallback (or primary path when model_name == default) ---
        if not self._native_mv:
            v1_model = CHRONOS2_DEFAULT_MODEL   # amazon/chronos-t5-small
            print(f"  Loading Chronos v1 per-channel '{v1_model}' ...", flush=True)
            self._pipeline = ChronosPipeline.from_pretrained(
                v1_model,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            print(f"  [Chronos2] per-channel fallback mode: 9 channels × 1D embed",
                  flush=True)

        # Register backbone as nn.Module sub-module for .to(device) propagation
        self.inner_model = self._pipeline.model
        _freeze(self.inner_model)
        self.inner_model.eval()

        self._d_model  = self._probe_d_model()
        self.embed_dim = self._d_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_d_model(self) -> int:
        """Resolve d_model from model config (tries common attribute paths)."""
        for attr_path in (
            "inner_model.model.config.d_model",
            "inner_model.config.d_model",
        ):
            try:
                obj = self
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return int(obj)
            except AttributeError:
                pass
        # Empirical probe: tiny dummy batch
        print("  [Chronos2] probing d_model empirically ...", flush=True)
        dummy = torch.zeros(1, 64)
        with torch.no_grad():
            emb, _ = self._pipeline.embed(dummy)
        d = int(emb.shape[-1])
        print(f"  [Chronos2] d_model = {d}", flush=True)
        return d

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) float32 → (B, embed_dim) float32"""
        if self._native_mv:
            return self._forward_native(x)
        return self._forward_per_channel(x)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Native Chronos-2 multivariate path. (B, C, T) → (B, D)"""
        x_np = x.detach().cpu().float().numpy()   # (B, C, T)
        with torch.no_grad():
            embs_list, _ = self._pipeline.embed(x_np)
        # embs_list: list of B tensors, each (C, num_patches+2, D) on CPU
        embs = torch.stack(embs_list, dim=0).float()   # (B, C, P+2, D)
        if self._strategy == "mean_pool":
            result = embs.mean(dim=(1, 2))             # (B, D)
        else:   # last_token: [REG] global-summary token at index 0
            result = embs[:, :, 0, :].mean(dim=1)      # (B, D)
        return result.to(x.device)

    def _forward_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel Chronos v1 fallback. Processes each of C channels as a
        separate univariate series; pools embeddings across channels. (B, C, T) → (B, D)
        """
        B, C, T = x.shape
        x_cpu = x.detach().cpu().float()
        channel_embs: list[torch.Tensor] = []
        with torch.no_grad():
            for c in range(C):
                x_c = x_cpu[:, c, :]                  # (B, T) — one channel, all samples
                # ChronosPipeline.embed returns (B, n_tokens+1, D) on CPU
                emb, _ = self._pipeline.embed(x_c)    # (B, T+1, D)
                emb = emb.float()
                if self._strategy == "mean_pool":
                    pooled = emb.mean(dim=1)           # (B, D)
                else:  # last_token: last context token (EOS at index -1)
                    pooled = emb[:, -1, :]             # (B, D)
                channel_embs.append(pooled)
        # Mean over all C channels → (B, D)
        result = torch.stack(channel_embs, dim=0).mean(dim=0)
        return result.to(x.device)


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
    chronos2_model: str = CHRONOS2_DEFAULT_MODEL,
) -> nn.Module:
    """Return the requested frozen encoder.

    Parameters
    ----------
    name            : "moment" | "chronos2" | "fallback" | "uni2ts"
    embed_strategy  : embedding strategy (see per-encoder docstrings)
    n_channels      : input channel count after preprocessing
    T               : time-series length (samples per window)
    chronos2_model  : HuggingFace model ID for Chronos-2 (chronos2 only)
    """
    if name == "moment":
        return MomentEncoder(embed_strategy=embed_strategy, n_channels=n_channels)

    if name == "chronos2":
        # Uses per-channel Chronos v1 fallback when chronos2_model == CHRONOS2_DEFAULT_MODEL.
        # For native multivariate mode, pass a Chronos-2 patch model ID explicitly.
        return Chronos2Encoder(
            model_name=chronos2_model,
            embed_strategy=embed_strategy,
            n_channels=n_channels,
        )

    if name == "fallback":
        return FallbackEncoder(n_channels=n_channels, T=T)

    if name == "chronos":
        raise ImportError(
            "Use --encoder chronos2 for the Chronos-2 patch-based encoder.\n"
            "The old 'chronos' key referenced Chronos v1 (tokenizer-based, univariate only)\n"
            "and is not supported. Install with:\n"
            "  pip install chronos-forecasting"
        )

    if name == "uni2ts":
        raise ImportError(
            "uni2ts encoder requested but 'uni2ts' (Moirai) is not installed.\n"
            "  pip install uni2ts\n"
            "Moirai is a forecasting model; embedding extraction is not directly\n"
            "supported without custom forward-pass modifications."
        )

    raise ValueError(f"Unknown encoder {name!r} — choose moment / chronos2 / fallback")
