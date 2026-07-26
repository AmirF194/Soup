"""soup train --stream-layers — layer streaming planner (v0.72.0 BETA).

The pure half: tier choice, pinned-vs-pageable decision, the architecture
allowlist, and the VRAM / throughput arithmetic. **No top-level torch** — this
module sits on the light CLI's import path so `soup profile` and friends can
forecast a streaming run without pulling in the training stack.

The runtime half (buffer pool, weight source, prefetch scheduler, layer
wrapper) lives in ``layer_stream_runtime.py``; the checkpoint sharder lives in
``layer_shard.py``.

Model of the mechanism: the frozen base lives in CPU RAM and is streamed into a
small pool of pre-allocated VRAM buffers one decoder layer at a time, so peak
VRAM is bounded by ONE layer rather than the whole model. Only the LoRA
adapters, their gradients and optimizer state stay resident.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

from rich.panel import Panel

# --- tiers (plan 5.1) -----------------------------------------------------
TIER_RAM = "ram"
TIER_DISK = "disk"
STREAM_SOURCES = ("auto", "ram", "disk")

#: model_bytes must be under this fraction of free RAM to claim the RAM tier.
RAM_TIER_HEADROOM = 0.7

# --- buffers --------------------------------------------------------------
MIN_STREAM_BUFFERS = 2
MAX_STREAM_BUFFERS = 8
DEFAULT_STREAM_BUFFERS = 2

#: FLOPs per parameter per token. 6 == WITH gradient checkpointing
#: (2 forward + 2 recompute + 2 dL/dx; base weight-grads are skipped because
#: the base is frozen). Streaming always checkpoints, so this is never 4.
FLOPS_PER_PARAM_PER_TOKEN = 6

#: v0.72.0 scope. Breadth (Mistral / Gemma / Phi) is v0.72.2.
SUPPORTED_STREAM_ARCHS = ("llama", "qwen2", "qwen3")

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
SUPPORTED_STREAM_DTYPES = tuple(sorted(_DTYPE_BYTES))

#: CUDA context + allocator fragmentation headroom (plan 4.1).
DEFAULT_WORKSPACE_BYTES = 1_000_000_000

_NVME = "nvme"


def dtype_bytes(name: str) -> int:
    """Bytes per element. THE single dtype-size table for layer streaming —
    the sharder, the planner and the trainer must not each keep their own."""
    try:
        return _DTYPE_BYTES[name]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r} for layer streaming; "
            f"supported: {', '.join(SUPPORTED_STREAM_DTYPES)}"
        ) from None


# ==========================================================================
# architecture gate
# ==========================================================================
def stream_arch_of(config: Any) -> str:
    """Return the streaming architecture family, or raise naming the allowlist.

    Mirrors ``utils/shrink.arch_family_of_config`` — an allowlist, not a
    heuristic, because a half-supported architecture streams weights into the
    wrong module and produces silently wrong numbers rather than a crash.
    """
    model_type = getattr(config, "model_type", None)
    if not model_type or not isinstance(model_type, str):
        raise ValueError(
            "layer streaming needs config.model_type to pick an architecture; "
            "none was found on the model config"
        )
    family = model_type.strip().lower()
    if family not in SUPPORTED_STREAM_ARCHS:
        raise ValueError(
            f"layer streaming does not support model_type={family!r} in "
            f"v0.72.0. Supported: {', '.join(SUPPORTED_STREAM_ARCHS)}. "
            f"More architectures land in v0.72.2."
        )
    return family


# ==========================================================================
# storage tier (plan 5.1)
# ==========================================================================
def choose_tier(
    model_bytes: int,
    free_ram_bytes: int,
    disk_kind: str,
    *,
    headroom: float = RAM_TIER_HEADROOM,
) -> str:
    """RAM is Tier 1; disk is overflow only; spinning rust is refused.

    plan 2.3: streaming from NVMe measured 2-11 TFLOPS against multiples of
    that from CPU RAM, so RAM is the primary source and disk is a fallback.
    plan P11: on an HDD, 80 shards x 2 reads = 160 seeks per step.
    """
    if model_bytes < free_ram_bytes * headroom:
        return TIER_RAM
    if disk_kind == _NVME:
        return TIER_DISK
    raise ValueError(
        f"layer streaming needs NVMe or more RAM: the base needs "
        f"{model_bytes / 1e9:.1f} GB, only {free_ram_bytes / 1e9:.1f} GB of RAM "
        f"is free, and the detected disk is {disk_kind!r} (not NVMe). "
        f"Free RAM, pick a smaller base, or move the model to an NVMe drive."
    )


@dataclass(frozen=True)
class PinDecision:
    """Whether the RAM store can be page-locked, and why not when it cannot."""

    pinned: bool
    reason: str


def decide_pinning(store_bytes: int, pinned_limit_bytes: Optional[int]) -> PinDecision:
    """Pin the RAM store when the box can actually page-lock it.

    Measured on the dev box (RTX 3050 4 GB / 16.9 GB RAM): the maximum
    page-locked host allocation was 7.12 GB even with 9.1 GB "available", so a
    5.55 GB base plus a CUDA context and a model skeleton did not fit. Falling
    back to a pageable store is correct, but it makes
    ``copy_(non_blocking=True)`` synchronous and therefore costs overlap:
    measured GPU utilisation dropped from 96.8% (pinned) to 79.3% (pageable).
    That cost is stated out loud rather than absorbed silently.
    """
    if pinned_limit_bytes is None:
        return PinDecision(
            pinned=True,
            reason="pinned-host ceiling unknown; attempting a page-locked store",
        )
    if store_bytes <= pinned_limit_bytes:
        return PinDecision(pinned=True, reason="store fits under the page-locked ceiling")
    return PinDecision(
        pinned=False,
        reason=(
            f"base store is {store_bytes / 1e9:.2f} GB but this box can only "
            f"page-lock {pinned_limit_bytes / 1e9:.2f} GB — falling back to a "
            f"pageable store. Host-to-device copies become synchronous, which "
            f"costs overlap: measured GPU utilisation drops from ~97% to ~79%. "
            f"Free RAM or use a smaller base to keep the pinned store."
        ),
    )


# ==========================================================================
# buffers
# ==========================================================================
def validate_stream_buffers(value: Any) -> int:
    """plan 8: 1 buffer is a scheduler bug, not a config."""
    if isinstance(value, bool):
        raise ValueError("training.stream_buffers must be an int, not bool")
    if not isinstance(value, int):
        raise ValueError(f"training.stream_buffers must be an int; got {type(value).__name__}")
    if value < MIN_STREAM_BUFFERS or value > MAX_STREAM_BUFFERS:
        raise ValueError(
            f"training.stream_buffers must be between {MIN_STREAM_BUFFERS} and "
            f"{MAX_STREAM_BUFFERS}; got {value}. A single buffer cannot overlap "
            f"load with compute — that is a scheduler bug, not a configuration."
        )
    return value


# ==========================================================================
# arithmetic (plan 4.1 / 4.3)
# ==========================================================================
def should_enable_hf_gradient_checkpointing(
    gradient_checkpointing: Any, *, stream_layers: bool
) -> bool:
    """Whether HF's own gradient checkpointing should be switched on.

    ``StreamedDecoderLayer`` already wraps every layer in
    ``checkpoint(use_reentrant=False)`` — that is what bounds activation memory
    AND what triggers the second weight read per step. Letting the HF Trainer
    also enable checkpointing double-recomputes every layer: the run still
    converges, it is just silently ~1.5x slower. So streaming always wins.
    """
    return bool(gradient_checkpointing) and not stream_layers


def free_ram_bytes() -> Optional[int]:
    """Available host RAM, or None when psutil is not installed."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except (AttributeError, OSError, ValueError):
        return None


def estimate_stream_tokens_per_sec(params: int, effective_tflops: float) -> float:
    """tok/s ceiling when compute-bound: TFLOPS_eff / (C * P), C = 6."""
    if params <= 0:
        raise ValueError(f"params must be positive; got {params}")
    if effective_tflops <= 0 or not math.isfinite(effective_tflops):
        raise ValueError(f"effective_tflops must be positive and finite; got {effective_tflops}")
    return (effective_tflops * 1e12) / (FLOPS_PER_PARAM_PER_TOKEN * params)


def estimate_epoch_seconds(tokens: int, tokens_per_sec: float) -> float:
    """Wall-clock for a token budget — the pre-flight forecast (plan 10)."""
    if tokens_per_sec <= 0 or not math.isfinite(tokens_per_sec):
        raise ValueError(f"tokens_per_sec must be positive and finite; got {tokens_per_sec}")
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative; got {tokens}")
    return tokens / tokens_per_sec


def estimate_logits_bytes(
    *, vocab_size: int, seq_len: int, batch_size: int = 1, upcast_fp32: bool = True
) -> int:
    """plan P5: logits, not weights, OOM you first on a small card.

    Llama-3.2's 128256-entry vocab at S=2048 is ~1.6 GB transient once the
    fp32 cross-entropy upcast is counted — more than both layer buffers.
    """
    if vocab_size <= 0 or seq_len <= 0 or batch_size <= 0:
        raise ValueError("vocab_size, seq_len and batch_size must all be positive")
    elements = vocab_size * seq_len * batch_size
    total = elements * 2
    if upcast_fp32:
        total += elements * 4
    return total


def estimate_stream_vram(
    *,
    layer_bytes: int,
    buffers: int,
    embed_bytes: int,
    adapter_bytes: int = 0,
    activation_bytes: int = 0,
    logits_bytes: int = 0,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> int:
    """Peak VRAM for a streaming step (plan 4.1)."""
    return (
        layer_bytes * buffers
        + embed_bytes
        + adapter_bytes
        + activation_bytes
        + logits_bytes
        + workspace_bytes
    )


# ==========================================================================
# specs and plans
# ==========================================================================
@dataclass(frozen=True)
class LayerSpec:
    """Shape + dtype of ONE tensor inside a decoder layer."""

    name: str
    shape: Tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        dtype_bytes(self.dtype)  # fail fast on an unsupported dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class StreamPlan:
    """What a streaming run will do, before it does it."""

    arch: str
    tier: str
    n_layers: int
    layer_bytes: int
    store_bytes: int
    embed_bytes: int
    buffers: int
    buffer_bytes: int
    pinned: bool
    notes: Tuple[str, ...]


def build_stream_plan(
    *,
    arch: str,
    n_layers: int,
    layer_bytes: int,
    embed_bytes: int,
    available_ram_bytes: int,
    pinned_limit_bytes: Optional[int],
    buffers: int = DEFAULT_STREAM_BUFFERS,
    disk_kind: str = _NVME,
) -> StreamPlan:
    """Decide tier + pinning and record every caveat as a visible note."""
    buffers = validate_stream_buffers(buffers)
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive; got {n_layers}")
    store_bytes = n_layers * layer_bytes
    tier = choose_tier(store_bytes + embed_bytes, available_ram_bytes, disk_kind)
    notes = []
    if tier == TIER_DISK:
        notes.append(
            "base does not fit in RAM — the disk tier lands in v0.72.2; "
            "v0.72.0 supports stream_source='ram' only"
        )
    decision = decide_pinning(store_bytes, pinned_limit_bytes)
    if not decision.pinned:
        notes.append(decision.reason)
    return StreamPlan(
        arch=arch,
        tier=tier,
        n_layers=n_layers,
        layer_bytes=layer_bytes,
        store_bytes=store_bytes,
        embed_bytes=embed_bytes,
        buffers=buffers,
        buffer_bytes=layer_bytes * buffers,
        pinned=decision.pinned,
        notes=tuple(notes),
    )


def render_stream_panel(plan: StreamPlan, extra_lines: Sequence[str] = ()) -> Panel:
    """Pre-flight summary. plan 10: tell the user the cost BEFORE the run."""
    lines = [
        f"[bold]Layer streaming[/] [yellow]BETA[/] — arch [cyan]{plan.arch}[/], "
        f"tier [cyan]{plan.tier}[/]",
        f"  base store   {plan.store_bytes / 1e9:.2f} GB across {plan.n_layers} layers "
        f"({'pinned' if plan.pinned else 'pageable'})",
        f"  VRAM buffers {plan.buffers} x {plan.layer_bytes / 1e6:.0f} MB "
        f"= {plan.buffer_bytes / 1e6:.0f} MB",
        f"  resident     {plan.embed_bytes / 1e6:.0f} MB embeddings + adapters",
    ]
    lines.extend(extra_lines)
    for note in plan.notes:
        lines.append(f"  [yellow]![/] {note}")
    return Panel("\n".join(lines), title="soup train --stream-layers", border_style="cyan")
