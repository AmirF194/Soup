"""DeepSpeed configuration templates for multi-GPU training."""

import copy
import json
import tempfile

# ZeRO Stage 2: splits optimizer states + gradients across GPUs
ZERO_STAGE_2 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# ZeRO Stage 3: splits model params + optimizer + gradients across GPUs
ZERO_STAGE_3 = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

# ZeRO Stage 2 with CPU offload (for memory-constrained setups)
ZERO_STAGE_2_OFFLOAD = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


# ZeRO Stage 3 with CPU parameter offload — the configuration a user whose
# weights do not fit in VRAM actually wants. Until this preset existed the only
# offload config here was stage 2, optimizer-only, so the DeepSpeed comparison in
# benchmarks/gate-h100-validation.md (STEP 3) had to be run from hand-written
# JSON. Measured there on one H100 (Llama-3.1-8B, bf16, LoRA r=8, 256 steps):
# 21.65 tok/s at a 38,135 MiB peak.
#
# `offload_optimizer` stays "none" deliberately. Turning it on makes DeepSpeed
# JIT-build its `cpu_adam` op, which needs a matching CUDA toolkit; on a box with
# no `nvcc` that fails with `CUDAMismatchException` and then
# `'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`. Users who do have a
# toolkit can flip it in a copy of this JSON.
#
# The stage3_* knobs are left at the stage-3 defaults. Tightening all four
# (max_live_parameters/max_reuse_distance 1e9 -> 1e7, param_persistence_threshold
# auto -> 0, prefetch_bucket_size auto -> 5e6) was measured on the same box at
# 5.6% less peak VRAM for 12% less throughput — a trade worth making by hand, not
# by default.
ZERO_STAGE_3_OFFLOAD = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": int(1e9),
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": int(1e9),
        "stage3_max_reuse_distance": int(1e9),
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


# ZeRO++ (v0.27.0): stage-3 base + hierarchical partitioning + quantized
# weights/gradients. Reduces inter-node communication 4-8x on 8+ GPUs.
ZERO_PLUS_PLUS = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": int(1e9),
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": int(1e9),
        "stage3_max_reuse_distance": int(1e9),
        "stage3_gather_16bit_weights_on_model_save": True,
        # ZeRO++ specifics
        "zero_hpz_partition_size": 8,
        "zero_quantized_weights": True,
        "zero_quantized_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


CONFIGS = {
    "zero2": ZERO_STAGE_2,
    "zero3": ZERO_STAGE_3,
    "zero2_offload": ZERO_STAGE_2_OFFLOAD,
    "zero3_offload": ZERO_STAGE_3_OFFLOAD,
    "zero++": ZERO_PLUS_PLUS,
    "zero_pp": ZERO_PLUS_PLUS,
}


def get_deepspeed_config(stage: str = "zero2") -> dict:
    """Get a DeepSpeed config dict by name."""
    if stage not in CONFIGS:
        raise ValueError(f"Unknown DeepSpeed config: {stage}. Options: {', '.join(CONFIGS.keys())}")
    return copy.deepcopy(CONFIGS[stage])


def write_deepspeed_config(stage: str = "zero2") -> str:
    """Write a DeepSpeed config to a temp file and return the path."""
    config = get_deepspeed_config(stage)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="ds_config_", delete=False
    )
    json.dump(config, tmp, indent=2)
    tmp.close()
    return tmp.name


def detect_multi_gpu() -> dict:
    """Detect multiple GPUs and return info."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_count": 0, "gpus": []}

        gpu_count = torch.cuda.device_count()
        gpus = []
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            gpus.append({
                "index": idx,
                "name": props.name,
                "memory_gb": props.total_memory / (1024 ** 3),
            })

        return {"gpu_count": gpu_count, "gpus": gpus}
    except ImportError:
        return {"gpu_count": 0, "gpus": []}
