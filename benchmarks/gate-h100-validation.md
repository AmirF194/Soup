<!--
Measurement record for Soup layer streaming on 8x H100, published verbatim.

Written during the work, in the order things happened, including the failures,
the corrected assumptions and the discarded numbers. Not a report assembled
afterwards.

Every previous record in this folder was measured on ONE machine: an RTX 3050
Laptop (4 GB) under Windows. This file is the first measurement on different
hardware, a different OS, and at model sizes the 4 GB box could not hold a
resident reference for.
-->

# H100 validation — external-hardware gate

Written as the work happened and committed after each measurement. Sections
appear in the order they were run, not in the order they would read best.

> **STATUS, read this first.** The defect in item 4 is **REPAIRED** (STEP 14) and
> the repair is gated on real 32B at 256/256 gradients, +2.9% VRAM, −4.8%
> throughput. Items 1 and 4 below describe the state *as measured*, which is what a
> working record is for; they are no longer the state of the shipped code. A second
> defect, #328, was diagnosed to a cause in Soup and also repaired (STEP 15). The
> steps are in the order they happened, not in the order they read best.

## What this session established

1. **Layer streaming's FORWARD is bit-exact at real model sizes, not just on
   toys.** Logits `torch.equal` to a resident reference of matching numerics at
   0.5B, 8B, 14B, 32B and 72B. Every previously published bit-exactness result
   was on 3-layer from-config checkpoints, because a 4 GB card cannot hold a
   resident 8B to compare against.
   **The BACKWARD is a separate claim and, AS MEASURED, it did not hold above
   ~165 MiB per NF4 layer** — gradients were exact at 0.5B / 8B / 14B and wrong at
   32B / 72B (item 4). Nothing in this record says "bit-exact" about a model without
   saying which half; see the table in STEP 6, which exists because two readers
   took the unqualified phrase to cover both. **STEP 14 repairs the backward and
   re-gates it at 256/256 on real 32B.**
2. **The published laptop throughput reproduces on completely different
   hardware.** Llama-3.1-8B NF4: 119.6 tok/s in 3.32 GB peak on an RTX 3050
   Laptop (v0.72.2 record) against a median 113.00 tok/s in 3.32 GB here, on an
   H100. The method is bound by host-to-device transfer, not by the GPU, and this
   is the first evidence of that from outside the original box.
3. **Against DeepSpeed ZeRO-3 CPU offload, same box, same data, same model:
   2.93x the throughput in 9.7x less peak VRAM** at matched numerics.
4. **A silent correctness defect, found and its cause named in the library that
   creates it.** `bitsandbytes.MatMul4Bit` stashes the packed weight and
   `quant_state` on `ctx` as plain attributes rather than through
   `save_for_backward`, so gradient checkpointing — which discards and recomputes
   *saved tensors* — cannot see them. The reference is captured in the forward,
   aliases the streaming buffer, and is read in the backward after that slot has
   been refilled. Confirmed by de-aliasing on the forward alone: exact in 5/5
   runs, while de-aliasing on the backward alone is exactly as broken as the
   control. That is also why bf16 was never affected at four times the bytes.
   With an NF4 base and
   layers above a threshold bracketed at 163.8–171.5 MiB, the streamed
   *backward* produces gradients
   that disagree with a resident reference on every layer but the last
   `stream_buffers` — while the forward stays bit-exact and the loss curve looks
   healthy. It affects 32B and 72B, not 8B or 14B; it is reproducible in about a
   minute on a synthetic 7.7 GB model. The mechanism is **aliasing, not a race**:
   a full `cuda.synchronize()` does not fix it and de-aliasing the pooled buffers
   does. Six competing hypotheses were tested and rejected along the way.
5. **A model trained by streaming is as good as one trained resident.** Paired
   over five disjoint training subsets and judged by Soup's own `soup ship`:
   mean difference +0.006 against an identical 0.013 within-arm spread on both
   sides. This had never been measured anywhere in the project.
6. **Four Linux/multi-GPU findings** the single-GPU Windows dev box could not
   have surfaced: `nn.DataParallel` versus `meta` parameters; **`device_map="auto"`
   making the multi-GPU path the tool itself advertises fail on every rank**;
   the `#328` meta leak reaching CUDA and all four preference losses; and
   `detect_disk_kind` classifying a 1.5 GB/s virtio disk as an HDD. The first two
   are fixed here, with tests that run on CPU-only CI as well as on eight cards.
7. **Eight cards of ZeRO-3 are slower than one card training resident** for a
   model that fits — which makes the honest positioning narrow: layer streaming
   is not "faster than DeepSpeed", it is for the case where the one card you have
   is too small.
8. **The defect is not issue #328**, and the two share no trigger axis —
   quantisation, model size, task and manifestation all differ, and each is the
   other's control. Two bugs, not one.
9. **Turning pinning off costs 6.56x throughput** at 32B NF4, measured with
   correctness asserted in the same process. That is too expensive to be a
   silent automatic fallback.
10. **The defect is repaired, and the repair is cheap** (STEP 13–14). Dequantising
    inside the checkpointed region keeps the weight out of `MatMul4Bit` entirely.
    Gated on real 32B against a resident NF4 reference with the repair-disabled arm
    as control: **256/256 gradients exact against the control's 8–12/256**, at
    **+2.9% peak VRAM and −4.8% throughput** — where the de-aliasing repair rejected
    in STEP 9 cost O(model). It is not even a numerics change at training shapes:
    `bitsandbytes::gemm_4bit` dispatches on M and already takes `_dequant_linear_fallback`
    at every M measured from 8 to 2048 on real projection shapes.
11. **#328 has a cause in Soup, not in torch** (STEP 15). `should_enable_hf_gradient_checkpointing`
    was called only by `sft.py`; the four preference wrappers left the value to TRL's
    default, which is `False` on trl 0.19.1 and `True` on 0.26.2 — one omission with
    two opposite symptoms. Repaired; `test_v07204.py` goes 5 failed → 67 passed here.
12. **Quality holds in NF4, not just bf16** (STEP 19). Mean paired difference
    **+0.0053** against within-arm spreads of 0.0333 and 0.0200 — 1.6 items in 300.
    Leg 2 points against the convenient direction: 3 of 5 **resident** runs came back
    DON'T SHIP against 1 of 5 streamed.
13. **Three shipped features had never been run, and two were broken** (STEP 17–18).
    `training.use_liger: true` crashed at step 0 across the whole supported trl pin;
    `--backend sglang` returned 500 on 100% of requests; `data.max_length` above 1024
    was silently ignored on **every** SFT run. All three repaired. FlashAttention and
    Liger measured for the first time at **1.015x** and **1.051x / −12.9% VRAM**
    against documented claims of "2-4x" and "20-60% / 20-40%", now corrected.
14. **The VRAM pre-flight's over-prediction is not preference-specific** (STEP 16) —
    the SFT control misses identically at the same effective row count, so the row
    multiplier is right and one shared coefficient is wrong. Deliberately **not**
    changed: 14 was measured on another stack, and swapping it would trade a
    documented over-prediction for an undocumented under-prediction.

Also here, because the record is the working one: a false alarm about the HF
cache that a control disproved, two inconclusive user-level attempts at the
defect, an invalid pair of VRAM numbers, an experiment whose test arm crashed
and could not answer the question it was built for, a wrong SM-clock column, an
8-GPU benchmark so short it reported CPU offload as *faster* than no offload, a
reading of the quality result that n=3 supported and n=5 destroyed, and nine
hypotheses about the defect's trigger that measurement rejected.

## Why this run exists

Layer streaming shipped in v0.72.0–v0.72.4 with every number measured on a
single RTX 3050 Laptop under Windows. Three questions are unanswerable on that
box *in principle*, not merely inconveniently:

1. **Bit-exactness at real model sizes.** The shipped bit-exactness gates use
   tiny from-config checkpoints (3 layers, hidden 32, vocab 64) because the
   standard demands a *resident* reference and a resident 8B does not fit in
   4 GB. So the strongest existing claim is "exact on 135M". On an 80 GB card a
   resident NF4 8B / 14B / 32B all fit, which turns that into "exact on real
   models". This is the most valuable of the three.
2. **A comparison against DeepSpeed ZeRO-3 CPU offload** on the same hardware,
   same data, same model. There is currently no such comparison anywhere in the
   record, and it is the first thing a reviewer asks.
3. **RAM vs disk tier.** v0.72.3 shipped the disk tier bit-exact but with its
   *speed* relative to RAM explicitly unmeasured. See "Constraint 2" below —
   this box probably cannot answer it either, for a different reason.

Plus two things that only 8 cards make practical: **variance** (every published
number so far is n=1) and a **size sweep**.

## The box

```
$ hostname
h100
$ nvidia-smi --query-gpu=index,name,memory.total,clocks.sm,driver_version --format=csv
index, name, memory.total [MiB], clocks.current.sm [MHz], driver_version
0, NVIDIA H100 80GB HBM3, 81559 MiB, 345 MHz, 590.48.01
   ... 8 identical rows (0-7) ...
$ python3 --version
Python 3.12.3
$ free -g | head -2
               total        used        free      shared  buff/cache   available
Mem:             503           4         501           0           0         499
$ df -h /
/dev/vda1       991G  106G  886G  11% /
$ nproc
32
```

Ubuntu 24.04.3 LTS, kernel 6.8.0-100. 8 x H100 80 GB HBM3, PIX between all
pairs (PCIe switch, **not** NVLink). Idle SM clock 345 MHz — every throughput
number below is quoted with the clock it was actually taken at.

**The box was not clean.** `/root/.cache/huggingface` already held 74 GB from a
previous tenant (`google/gemma-3-27b-it` 52 GB, `deepseek-ai/DeepSeek-OCR`
6.3 GB, two embedding models). Left in place, noted here because it is charged
against the same 886 GB disk budget everything else has to fit in.

### Constraint 1 — pinned memory ceiling is 63 GB, not 503 GB

```
$ ulimit -l
66020128
```

66,020,128 KB = 62.96 GB. Layer streaming puts the frozen base in **page-locked**
host RAM, so this — not the 503 GB of installed RAM — is the real ceiling on
store size. NF4 rates: 8B ≈ 5.7 GB, 14B ≈ 10 GB, 32B ≈ 20 GB, 70B ≈ 40 GB all
fit; any bf16 base above ~30B does not.

Not raised. Deliberately: the shipped code's RAM-tier decision is written against
whatever `ulimit -l` reports, and raising it would test a configuration no
ordinary user has.

### Constraint 2 — there is no NVMe on this machine

```
$ cat /sys/block/vda/queue/rotational
1
```

The only block device is a virtual `vda` reporting `rotational=1`.
`utils/layer_stream.detect_disk_kind` refuses the disk overflow tier on anything
that is not NVMe, so the "RAM vs disk" question is **not answerable on this box
in the shipped configuration**. Per the brief this gate is not to be bypassed
silently: the disk's real throughput is measured first, then the decision is
recorded. See the DISK section below.

---

## Timeline

### Setup — network is not the long pole (2026-08-06, ~11:20–11:30 +03:00)

First measurement of the session, and it changes the plan. Downloads were
started before anything else on the assumption that network would be the
constraint over a 72-hour calendar window.

```
=== START NousResearch/Meta-Llama-3.1-8B-Instruct 2026-08-06T11:24:57+03:00
DONE  ... in 68.4s
=== END   NousResearch/Meta-Llama-3.1-8B-Instruct 2026-08-06T11:26:06+03:00 df=870G
```

16 GB in 68.4 s. That is 234 MB/s (division, not a measurement of steady-state
bandwidth). At that rate the whole planned ~250 GB of weights is ~18 minutes,
so the download schedule stops being a scheduling constraint at all and the
72-hour budget is bounded by GPU work and by debugging, not by transfer.

Models chosen non-gated, because no HF token is available on this box and
`meta-llama/*` is gated:

| role | repo | arch | fp16 size |
|---|---|---|---|
| smoke | `Qwen/Qwen2.5-0.5B-Instruct` | qwen2 | ~1 GB |
| 8B (flagship reproduction) | `NousResearch/Meta-Llama-3.1-8B-Instruct` | llama | ~16 GB |
| 14B | `Qwen/Qwen2.5-14B-Instruct` | qwen2 | ~28 GB |
| 32B | `Qwen/Qwen2.5-32B-Instruct` | qwen2 | ~64 GB |

`NousResearch/Meta-Llama-3.1-8B-Instruct` is an ungated mirror of the exact
checkpoint the v0.72.2 record used, so the 8B row is a genuine cross-hardware
reproduction rather than a different model of the same size.

Whole download chain, for the record:

| repo | fp16 | wall time | implied rate |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.95 GB | 6.1 s | — |
| `NousResearch/Meta-Llama-3.1-8B-Instruct` | ~16 GB | 68.4 s | 234 MB/s |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | 138.2 s | 203 MB/s |
| `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | 303.0 s | 211 MB/s |

Rates are division, not sustained-bandwidth measurements.

### The installed stack is much newer than every published number

```
torch 2.13.0+cu130   cuda 13.0   ngpu 8
bitsandbytes 0.50.0
transformers 4.57.6  trl 0.26.2  peft 0.20.0  accelerate 1.14.0
```

Against the dev box that produced every prior record (torch 2.5.1+cu121,
bnb 0.49.2, trl 0.19.1, peft 0.18.1, transformers 4.57.6). Only `transformers`
matches. `pip install -e ".[train]"` resolved this on its own; it was not pinned.
That is worth stating up front because the first two findings below are both
version-sensitive, and neither would have appeared on the dev stack.

Note `trl 0.26.2` sits directly under the `<0.27` cap the v0.72.4 record derived
by construction. The cap holds: all six preference configs build.

---

## STEP 1 — the shipped streaming test suites on CUDA

The brief: run the existing suites, because a chunk of them are CUDA-gated and
have only ever run on a 4 GB card or been skipped. Any failure is a finding.

```
$ /root/venv/bin/python -m pytest tests/test_v07200.py tests/test_v07202.py \
      tests/test_v07203.py tests/test_v07204.py -v --no-cov
9 failed, 419 passed, 65 warnings in 36.12s
```

**9 failures.** They are three distinct defects, not one.

### FINDING 1 — layer streaming is broken on any multi-GPU box (`nn.DataParallel` vs `meta`)

Eight of the nine failed with the identical error:

```
RuntimeError: module must have its parameters and buffers on device cuda:0
(device_ids[0]) but found one of them on device: meta
  .../torch/nn/parallel/data_parallel.py:180
```

HF `Trainer` wraps the model in `nn.DataParallel` whenever
`torch.cuda.device_count() > 1` and the run was not launched distributed.
`DataParallel` validates that every parameter lives on `device_ids[0]` — and
layer streaming's entire design is that the decoder parameters stay on `meta`.
So the two are incompatible by construction.

Confirmed by re-running the same suites with a single card visible:

```
$ CUDA_VISIBLE_DEVICES=0 python -m pytest <same four files> --no-cov -q
6 failed, 422 passed, 72 warnings in 36.43s
```

The two end-to-end training-step tests
(`test_v07200::test_one_training_step_actually_runs`,
`test_v07202::test_one_nf4_training_step_actually_runs`) flip to pass, as does
`test_v07202::test_the_saved_adapter_is_canonical`.

**This is not a Linux finding, it is a multi-GPU finding.** The dev box had one
GPU, so `device_count() > 1` was never true and the branch was never taken. It
would fail identically on a two-GPU Windows machine. Since streaming exists to
fit a model on *one* small card, a user with several cards is not the target
case — but they get a raw torch error naming `meta`, with nothing pointing at
`stream_layers`, which is the actual defect. Every measurement below therefore
runs with `CUDA_VISIBLE_DEVICES` pinned, and that pinning is stated each time.

### FINDING 2 — the meta leak (#328) is far wider than the issue records

The remaining five preference-loss failures share one signature:

```
RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
  .../torch/_prims_common/__init__.py:931 in check_same_device
```

`tests/test_v07204.py::TestKtoNeedsMoreThanOneRow::test_kto_streams_at_batch_two`
carries a long docstring describing exactly this as known issue **#328**,
tolerated *on CPU only*, with "the same signature on CUDA is a hard failure" and
"the variable is the torch version, not the device". On this box it fires **on
CUDA**, so that tolerance is doing its job: the test failed rather than hiding it.

**A hypothesis I formed and then had to discard.** The four failing parametrized
cases were `[dpo]` and `[kto]` — precisely the two preference losses that take a
reference model — so I wrote down that the leak was reachable from the reference
forward, which would have been a sharp localization. It is wrong. That test class
is only parametrized over dpo/kto in the first place, so it could not have
reported anything else. Testing the claim directly:

```
$ CUDA_VISIBLE_DEVICES=0 python /root/repro_328.py
torch 2.13.0+cu130 cuda_devices 1
sft    OK
dpo    FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
orpo   FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
simpo  FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
kto    FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
```

**All four fail. SFT passes.** ORPO and SimPO are genuinely reference-free
(v0.72.4 verified they have no `ref_model` attribute at all), so the reference
forward is not the mechanism. The reason the suite reported only dpo/kto is a
coverage gap: there is no CUDA `train()` test for orpo or simpo.

Real localization, from the full traceback:

```
transformers/trainer.py:4071  training_step
accelerate/accelerator.py:2850  backward
torch/autograd/graph.py:979   _engine_run_backward
torch/utils/checkpoint.py:314  backward          <-- recompute
transformers/models/llama/modeling_llama.py:292  LlamaDecoderLayer.forward
transformers/models/llama/modeling_llama.py:67   LlamaRMSNorm.forward
torch/_refs/__init__.py:1801  mul
torch/_prims_common/__init__.py:931  check_same_device
RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
```

Line 67 is `return self.weight * hidden_states.to(input_dtype)`. So the failure
is in the **backward recompute of gradient checkpointing**, where the RMSNorm
weight is still the `meta` placeholder — i.e. that recompute did not get the
streamed substitution the original forward got. It is a *backward-pass* defect,
not a loss-formulation one, which is consistent with SFT passing only on the dev
torch and with "newer torch decomposes more ops" in the issue text.

Not investigated further and **no Soup code changed** — the brief forbids editing
the code to make a measurement pass, and SFT is what the size sweep and the
DeepSpeed comparison need. Recorded so #328 can be re-scoped: it is not
CPU-only and not KTO-only.

### FINDING 3 — the GEMM-ceiling plausibility bound is calibrated to a laptop

```
E  assert 786.4800164584345 < 200.0
E   +where 786.4800164584345 = GemmCeiling(tflops=786.48, sm_clock_mhz=1980, size=4096).tflops
```

`utils/layer_stream_runtime.measure_gemm_tflops` works correctly — 786.5 TFLOPS
at 1980 MHz is a sane bf16 number for an H100. The *test* asserts the result is
below 200 TFLOPS as a sanity bound, a bound written when the only hardware in
existence for this project was an RTX 3050. The probe is fine; the assertion does
not generalize to datacenter GPUs. Cosmetic, but it means the shipped suite
cannot go green on this class of machine.

> **Fixed while this session was still running**, in `f715218` — the bound is now
> a named `_MAX_PLAUSIBLE_GEMM_TFLOPS` chosen to catch a wrong *order of
> magnitude* rather than to encode one generation of hardware.

---

## STEP 2 — bit-exactness at real model sizes

The point of the whole trip. The shipped gates compare a streamed model against
a **resident** model of the same numerics, and on a 4 GB card the largest thing
with a resident reference is a 3-layer toy. Here the reference fits.

**Protocol** — copied from `gate-v0.72.3-breadth.md` GATE 1 and
`tests/test_v07202.py::TestNF4BitExactVsResident`, changed in exactly two ways:
the checkpoint is a real downloaded model rather than a from-config toy, and the
device is CUDA/bf16 rather than CPU/float32. Everything else is verbatim,
including the **vacuity defence**: PEFT initialises `lora_B = 0`, so a completely
detached adapter is byte-identical to a fresh one and every parity assertion
passes for the wrong reason. Each run randomises `lora_B` on the reference,
copies the adapters across the `.inner.` wrapper difference, and asserts a
non-zero number of tensors were copied.

Reference numerics always match: **streamed NF4 is compared against resident
NF4**, never against resident bf16, whose quantisation error is wider than a real
bug and would hide one inside it.

Checks per model: (1) `torch.equal` on logits; (2) layer-0 LoRA gradient
non-zero and every layer non-zero (plan P2 — a severed graph still lowers loss);
(3) decoder parameters still on `meta`; (4) 5-step loss curves identical.

### A false alarm I raised and had to withdraw

The first attempt pointed the script at the raw HF snapshot directory and died:

```
layer-stream sharder: skipping symlinked shard model.safetensors
FileNotFoundError: no .safetensors weight files found in
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae5576...
 — layer streaming needs a safetensors checkpoint
```

`snapshot_download` on Linux populates `snapshots/<rev>/` with **symlinks** into
`blobs/`, and `layer_shard._discover_safetensors` deliberately skips symlinked
shards. On Windows the HF cache copies rather than symlinks without developer
mode, so this could not have shown up on the dev box. I wrote it down as a major
Linux finding: *layer streaming cannot shard anything in the standard HF cache*.

**That is wrong, and the check that settled it was running the real CLI.** A
genuine `soup train` with `stream_layers: true` on the same model works:

```
Layer streaming ready: 24 layers, 0.18 GB pinned RAM store, 2 x 8 MB VRAM buffers
LoRA applied: 540,672 trainable / 494,032,768 total (0.11%)
{'train_runtime': 15.3764, 'train_samples_per_second': 4.162, ...}
```

because `stream_setup` does not pass a snapshot path at all — it calls
`spectrum_scan.resolve_model_weights`, which materialises real files under
`~/.soup/spectrum/weights/<slug>/`:

```
resolved: /root/.soup/spectrum/weights/Qwen__Qwen2.5-0.5B-Instruct
   config.json        symlink= False
   model.safetensors  symlink= False
```

So the defect was in my harness, not in Soup, and the symlink guard is doing its
job. Kept here because I had already written the finding down, and because it has
one real consequence: **weights exist twice on disk**, once in the HF cache and
once under `~/.soup`. For this session's four models that is ~250 GB duplicated,
which is a live constraint against 886 GB.

That run is also the milestone the brief asked for on its own terms — the first
real `soup train` layer-streaming run on Linux, and it worked unmodified.

### Smoke — Qwen2.5-0.5B-Instruct, NF4, CUDA bf16

```
CUDA_VISIBLE_DEVICES=0 python /root/gate/bitexact.py \
  --weights Qwen/Qwen2.5-0.5B-Instruct --shards /root/shards/qwen05b_nf4 \
  --quant nf4 --seq 64
```

```
max_abs_logit_diff  0.0        bit_exact  true
adapter_tensors_copied  96     (non-vacuous)
layer0_lora_grad  9.093866e-01  24/24 layers non-zero
curves_equal  true             curve_max_rel  0.0
meta_params  288               store 0.18 GB pinned, tier ram
```

### **Llama-3.1-8B-Instruct, NF4, CUDA bf16 — bit-exact**

The result this trip existed for.

```
CUDA_VISIBLE_DEVICES=0 python /root/gate/bitexact.py \
  --weights NousResearch/Meta-Llama-3.1-8B-Instruct \
  --shards /root/shards/llama8b_nf4 --quant nf4 --seq 128
```

```json
{
  "weights": "/root/.soup/spectrum/weights/NousResearch__Meta-Llama-3.1-8B-Instruct",
  "quant": "nf4", "dtype": "bfloat16", "seq": 128,
  "torch": "2.13.0+cu130", "gpu": "NVIDIA H100 80GB HBM3",
  "shard_seconds": 23.3,
  "stream_stats": {"n_layers": 32, "buffers": 2, "buffer_bytes": 225058872,
                   "store_bytes": 3600941952, "pinned": true, "tier": "ram",
                   "device": "cuda:0", "total_params": 8030261248},
  "meta_params": 288,
  "adapter_tensors_copied": 128,
  "max_abs_logit_diff": 0.0,
  "bit_exact": true,
  "logit_abs_max": 26.875,
  "layer0_lora_grad": 0.173665851354599,
  "layers_with_grad": 32, "n_layers_seen": 32,
  "curve_streamed":  [11.953645706176758, 11.442726135253906, 10.979948043823242,
                      10.62893295288086, 9.73975944519043],
  "curve_resident":  [11.953645706176758, 11.442726135253906, 10.979948043823242,
                      10.62893295288086, 9.73975944519043],
  "curves_equal": true, "curve_max_rel": 0.0,
  "sm_clock_mhz_at_start": 345, "sm_clock_mhz_at_end": 1980
}
```

`max_abs_logit_diff` is exactly `0.0` and `torch.equal` is true over a
`[1, 128, 128256]` logits tensor whose largest element is 26.875 — so this is
equality on real values, not equality of two zeros. 128 adapter tensors copied,
so it is not the vacuous comparison. All 32 layers receive gradient.

`total_params` reports 8,030,261,248, i.e. the honest count from the sharder
rather than PEFT's inflated NF4 figure — the v0.72.2 display defect is fixed and
stays fixed at 8B.

### An invalid measurement in that same JSON — the VRAM peaks

The script also records `peak_vram_streamed_bytes: 8386682880` and
`peak_vram_resident_bytes: 8004792832`. **Neither is a peak-VRAM measurement of
anything, and they must not be quoted as one.** The script loads the resident
reference and never frees it before timing the streamed loss curve, so the
"streamed" peak contains a whole resident NF4 8B sitting alongside. That is why
the streamed number is *larger* than the resident one, which would otherwise
contradict the entire feature.

Left in the record rather than deleted. It does not touch the bit-exactness
claim — that comparison requires both models in memory *by construction* — but a
real streamed-peak number has to come from a separate single-model run.

### 14B and 32B — logits bit-exact, and then 32B's loss curve did not match

Run in parallel on separate cards (GPU 1 and GPU 2). Parallelism cannot affect
an equality claim, only a timing one, and no timing is claimed here.

| model | params | layers | store (pinned) | shard | copied | max abs logit diff | bit-exact | layer-0 grad | layers w/ grad | curves equal |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 494,032,768 | 24 | 0.18 GB | 1.0 s | 96 | 0.0 | yes | 9.093866e-01 | 24/24 | yes |
| Llama-3.1-8B | 8,030,261,248 | 32 | 3.35 GB | 23.3 s | 128 | 0.0 | yes | 1.736659e-01 | 32/32 | yes |
| Qwen2.5-14B | 14,770,033,664 | 48 | 6.35 GB | 35.9 s | 192 | 0.0 | yes | 1.185666e-02 | 48/48 | yes |
| Qwen2.5-32B | 32,763,876,352 | 64 | 14.99 GB | 79.2 s | 256 | 0.0 | yes | 6.008708e-03 | 64/64 | **no** |

All four are bit-exact in the **forward**. 32B is the odd one: `curves_equal:
false`, `curve_max_rel: 0.0586`.

```
streamed               resident                diff
13.05783462524414      13.05783462524414       +0.000000e+00
12.815812110900879     12.571563720703125      +2.442484e-01
12.531695365905762     12.131075859069824      +4.006195e-01
12.24169921875         11.740943908691406      +5.007553e-01
11.970612525939941     11.308257102966309      +6.623554e-01
```

Step 0 is identical, as it must be given bit-exact logits. Divergence begins
after the first optimizer step, so the gradients are what differ.

## STEP 2b — chasing the 32B divergence

This turned into the most important thread of the session, and it took four
measurements and one self-contradiction to land.

### Measurement 1 — gradients matched, and both models failed to reproduce themselves

Direct comparison of all LoRA gradients after **one** backward, plus each model's
own 5-step curve run twice:

```
A grads: 256/256 bit-exact, worst abs 0.000000e+00 rel 0.000000e+00
B streamed self-identical: False
C resident self-identical: False
```

Read naively this says "gradients are fine, both models are just
non-deterministic, nothing to see". That reading is wrong, and it is wrong in a
way worth keeping: **it is internally inconsistent**. If the backward were
non-deterministic, two independently computed backwards would not agree
bit-exactly across 256 tensors. Both statements cannot hold.

### Measurement 2 — which stage actually varies

Repeating forward and backward on the same model with the same input:

```
                       forward reproducible   grads rep1 vs rep3   curve reproducible
Qwen2.5-32B streamed   yes                    8/256 bit-exact      no
Qwen2.5-32B resident   yes                    256/256 bit-exact    yes
```

So the **forward is deterministic in both**, the **resident backward is
deterministic**, and the **streamed backward is not**. That also explains
measurement 1's contradiction: `graddiff` compared each model's *first* backward,
and the first one happened to be right.

Not the GPU and not the activation size — it reproduces on a different card and
at `seq 32`:

```
32B on GPU 6:   grad repeat worst abs 7.917519e-01  bit-exact 8/256
32B at seq 32:  grad repeat worst abs 5.827999e-01  bit-exact 8/256
```

And it is size-dependent. 0.5B, 8B and 14B are perfectly reproducible:

```
0.5B  grad repeat worst abs 0.000000e+00  bit-exact 96/96   curves identical=True
8B    grad repeat worst abs 0.000000e+00  bit-exact 128/128 curves identical=True
14B   grad repeat worst abs 0.000000e+00  bit-exact 192/192 curves identical=True
```

### Measurement 3 — not "different", **wrong**

Non-reproducibility on its own is survivable; bf16 atomics do it. Being wrong is
not. The resident model reproduces itself 256/256, so it is a valid fixed
reference. Taking resident's gradients once and diffing every streamed
repetition against that same reference (streamed backwards run first, so nothing
about the resident run can be blamed):

```
resident reference loss 13.048010826  (256 grad tensors)
rep 1: loss 13.048010826 (==resident True)  grads exact 256/256  64 layers  worst_rel  0.0000
rep 2: loss 13.048010826 (==resident True)  grads exact   8/256  [62, 63]   worst_rel 55.9389
rep 3: loss 13.048010826 (==resident True)  grads exact   8/256  [62, 63]   worst_rel 55.9389
rep 4: loss 13.048010826 (==resident True)  grads exact   8/256  [62, 63]   worst_rel 55.9389
rep 5: loss 13.048010826 (==resident True)  grads exact   8/256  [62, 63]   worst_rel 55.9389
```

The loss is bit-identical to resident on **every** repetition — the forward is
never wrong — while the gradients for 62 of 64 layers are off by up to 56x
relative. This is the silent-failure class the codebase's own notes name: *"a
recycled buffer yields silently WRONG gradients not a crash."* Nothing in a
training log would show it. The loss still falls.

The surviving layers are 62 and 63: the **last two**, i.e. the first two the
backward touches.

### Measurement 4 — the survivor count is exactly the buffer count

```
buffers=2  rep2+  exact  8/256   layers [62, 63]
buffers=3  rep2+  exact 12/256   layers [61, 62, 63]
buffers=4  rep2+  exact 16/256   layers [60, 61, 62, 63]
buffers=8  rep2+  exact 32/256   8 layers
```

One layer of survivors per buffer, always the last ones. Those are precisely the
layers still sitting in the pool when the backward starts — the ones that need
**no transfer**. Every layer that has to be fetched is wrong.

The layers *are* being fetched — the counter rules out "the backward simply
doesn't reload":

```
8B    layer_loads  0 -> 62  (delta 62)   then 61, 61      # 32 fwd + 32 bwd - 2 pooled
32B   layer_loads  0 -> 126 (delta 126)  then 125, 125    # 64 fwd + 64 bwd - 2 pooled
```

So the transfers are issued and counted; the compute consumes the buffer before
the copy is complete. That is a **race**, not a missing load, and the
`LayerBufferPool.wait()` ownership check that exists specifically to prevent this
is not catching it at this transfer size (32B ≈ 234 MB/layer, against 14B ≈
132 MB and 8B ≈ 105 MB).

> **Written at this point in the session and left as written.** The parenthetical
> above attributes the trigger to transfer size in general. STEP 6 tests that and
> refutes it: bf16 is exact at 935 MiB per layer, nearly four times the NF4 size
> that fails. The trigger is the NF4 path specifically. The "race" half of the
> sentence survives; the "at this transfer size" half does not.

Severity varies run to run, which is the signature of a race rather than a logic
error: in one run rep 1 was 256/256 exact, in another 20/256 (`worst_rel 0.8169`),
and rep 3's `worst_rel` came out 8.5330 in one run and 55.9389 in another.

### What this means, stated carefully

- **Confirmed:** at Qwen2.5-32B, NF4, bf16, on an H100, the streamed backward
  produces gradients that disagree with a deterministic resident reference on
  every layer that requires a transfer, from the second backward pass onward.
  The forward stays bit-exact throughout.
- **Confirmed:** 0.5B, 8B and 14B do not show it — 96/96, 128/128 and 192/192
  gradients bit-exact across repeated passes, and 5-step curves identical to
  resident.
- **Not established:** the exact threshold, whether it is per-layer bytes,
  layer count, or transfer-vs-compute ratio; whether it appears at 8B under a
  longer sequence or larger batch (which lengthen compute, and would *narrow* the
  window, so probably not) or under a slower host-to-device path; whether the
  torch 2.13 stack matters. The dev box could not have seen this: it never ran a
  model this large, because it could not hold the resident reference to compare
  against.
- **Not done:** no Soup code was changed. The brief forbids editing the code to
  make a measurement pass, and this is a finding to hand over, not a patch to
  smuggle in.

Practical reading for now: **layer streaming is verified exact through 14B and
is not trustworthy at 32B** until the pool synchronisation is fixed. The
published claims — all at 8B and below — are unaffected, and the 8B row of this
session independently reproduces them on different hardware, a different OS and a
much newer torch.

### Two attempts to confirm it at the user level, both inconclusive — kept

The controlled harness is one thing; a reviewer will ask whether a real
`soup train` is affected. Two attempts, neither of which supports a claim.

**Attempt 1 — streamed vs resident through `soup train`.** Same data, same
hyper-parameters, 32B, one epoch of 64 rows:

```
32B stream    train_loss 1.2372903674840927   grad_norm 2.96875
32B resident  train_loss 1.1320892516523600   grad_norm 1.89843750
```

A 9.3% gap and a grad_norm nearly 60% apart, both runs finishing clean with no
warning — which looks like exactly the predicted silent failure. **It is not
valid evidence.** The 8B control is what exposed the flaw:

```
8B stream     train_loss 0.032066941927041626  grad_norm 0.031494140625
8B resident   train_loss 0.030412952972255880  grad_norm 0.019042968750
```

8B is bit-exact under the controlled harness (128/128 gradients), yet differs
here by 5.4%. So the gap cannot be the gradient defect. The cause is that the
two paths **initialise LoRA independently** — `build_streamed_model` seeds its
adapter init itself, the resident path takes the global seed — so the runs start
from different `lora_A` matrices and must diverge whatever the gradients do. The
bit-exactness harness controls for this by copying adapters across; `soup train`
has no reason to.

**Attempt 2 — run-to-run reproducibility of the same config.** If the streamed
backward races, two identical streamed runs should scatter more than two
identical resident runs:

```
32B stream   run1 1.1927178781479597   run2 1.2143159396946430   (1.8% apart)
32B resident run1 1.1417463254183530   run2 1.1260867975652218   (1.4% apart)
```

Also inconclusive: the resident path is not reproducible across *processes*
either — 1.4% — even though it reproduced itself 256/256 *within* a process.
`soup train` does not pin the adapter-init seed, so process-to-process scatter
swamps the effect being looked for. 1.8% vs 1.4% with n=2 separates nothing.

Both left in. The honest position is that the defect is established by the
controlled harness (a deterministic reference, adapters synced, the survivor
count tracking the buffer count) and **is not yet demonstrated end-to-end
through the CLI**, because the CLI has no seed control that would make such a
comparison mean anything. That gap is itself worth reporting: two `soup train`
runs of one unchanged config do not reproduce each other.

---

## STEP 3 — against DeepSpeed ZeRO-3 with CPU offload

The comparison the record has never had. Both techniques answer the same
question — *the weights do not fit in VRAM, now what* — by keeping parameters in
host RAM and bringing them to the GPU as needed. ZeRO-3 gathers a shard per
module; layer streaming copies a decoder layer into a pre-allocated buffer.

Everything held equal: one H100, `Llama-3.1-8B-Instruct`, the same 64-row
dataset, `max_length 256`, batch 1, 4 epochs (256 optimizer steps), LoRA r=8
alpha=16, and the same `soup train` entry point — DeepSpeed is reached through
Soup's own `--deepspeed` flag, so this is one tool against itself.

### Getting DeepSpeed to run at all took three fixes, and they are findings

1. **Soup ships no ZeRO-3 CPU-offload preset.** `utils/deepspeed.CONFIGS` has
   `zero2`, `zero3`, `zero2_offload`, `zero++` — `zero3` sets
   `offload_param: none`, and the only offload preset is stage 2 optimizer-only.
   The configuration a memory-constrained user actually wants is not among them.
   Supplied here as a hand-written JSON, which `--deepspeed <path>` accepts.
   **Fixed while this session was still running**, in `9ae7a9b`, which adds a
   `zero3_offload` preset holding exactly the config measured below.
2. **`offload_optimizer: cpu` cannot work on this box.** DeepSpeed JIT-builds
   its `cpu_adam` op and needs a matching CUDA toolkit; the machine has no
   `nvcc`. Installing `ninja` moved the error along to
   `CUDAMismatchException: Installed CUDA version ...` and then
   `AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`.
   Dropped to `offload_optimizer: none`, which is the **fairer** comparison
   anyway: layer streaming also keeps its optimizer on the GPU, because with a
   frozen base the optimizer only covers LoRA.
3. **torch 2.13 + DeepSpeed 0.19.4 + transformers 4.57.6 crash on the LR
   scheduler.**
   ```
   transformers/trainer.py:2750 in _inner_training_loop -> self.lr_scheduler.step()
   torch/optim/lr_scheduler.py:296 in _update_lr
     for param_group, lr in zip(self.optimizer.param_groups, values, strict=True)
   ValueError: zip() argument 2 is longer than argument 1
   ```
   torch 2.13 passes `strict=True` to that `zip`; the DeepSpeed-wrapped optimizer
   does not expose one param group per scheduler value. Nothing in Soup is in
   that call path. Worked around by letting DeepSpeed own both the optimizer and
   the scheduler (`"optimizer": {"type": "AdamW"}`, `"scheduler": {"type":
   "WarmupLR"}`), which is a supported DeepSpeed configuration.

Also required: `apt install libopenmpi-dev` + `pip install mpi4py`, absent from
the box.

### The numbers

VRAM sampled from `nvidia-smi` every 0.5 s for the whole run; tok/s is
`num_tokens / train_runtime`, i.e. division, from the values the trainer itself
reports. Same card class, SM clock 1980 MHz median-while-busy in every row.

| run | base dtype | tok/s | peak VRAM | train_runtime | mean GPU util | exit |
|---|---|---|---|---|---|---|
| layer streaming | NF4 | **121.46** | **3,399 MiB** | 67.44 s | 54.1% | 0 |
| layer streaming | bf16 | 63.52 | 3,935 MiB | 128.97 s | 72.4% | 0 |
| DeepSpeed ZeRO-3, param offload | bf16 | 21.65 | 38,135 MiB | 378.41 s | 45.5% | 0 |

At **matched numerics** (bf16 vs bf16), layer streaming is **2.93x** the
throughput (63.52 / 21.65) in **9.7x** less peak VRAM (38,135 / 3,935). Both
ratios are division of the measured values in the table.

Against the configuration Soup actually recommends (NF4), it is 5.61x the
throughput at 11.2x less VRAM — but that row changes two variables at once and
is quoted only as the practical end-to-end difference, not as a controlled
comparison.

Loss after 4 epochs lands in the same place for all three (0.0114 / 0.0105 /
0.0134), which is the sanity check that all three are training the same task and
not diverging.

### The competitor was given a second, memory-tuned chance

38 GB of peak VRAM is not ZeRO-3 doing its best — with 80 GB available and
`stage3_max_live_parameters: 1e9`, it has no reason to be frugal, and comparing
against an untuned competitor would be worthless. A second run tightened every
memory knob it has (`stage3_max_live_parameters` 1e9 -> 1e7,
`stage3_max_reuse_distance` 1e9 -> 1e7, `stage3_param_persistence_threshold`
auto -> 0, `stage3_prefetch_bucket_size` auto -> 5e6).

```
ds_z3tight  tok/s 19.09   peak 35,997 MiB   train_runtime 429.15 s   util 41.8%   exit 0
```

Tightening every knob bought **5.6% less VRAM for 12% less throughput**
(38,135 -> 35,997 MiB, 21.65 -> 19.09 tok/s). It does not change the comparison,
and the tuned row is the one to quote against streaming if only one is quoted.

**The honest caveat, stated rather than buried:** this is a *single-GPU*
comparison. ZeRO-3 is designed to shard across ranks, and with `world_size = 1`
there is no partner to shard to, so it is being used outside its intended
regime. That regime is precisely layer streaming's home ground — the question
both are answering here is "one GPU, weights that do not fit" — so the
comparison is the relevant one, but a multi-GPU ZeRO-3 run would answer a
different question and was not attempted. No multi-GPU claim is made either way.

---

## STEP 4 — variance, and the size sweep

Every previously published throughput number for this feature is n=1. Five
repeats per configuration, each a full `soup train` (4 epochs, 256 steps),
runs of different configurations on different cards so nothing queues behind
anything else. tok/s = `num_tokens / train_runtime`, both reported by the
trainer; peak VRAM from `nvidia-smi` sampled at 2 Hz.

| config | n | tok/s median | min | max | spread | peak VRAM median | store (pinned) |
|---|---|---|---|---|---|---|---|
| 8B NF4 streamed | 5 | **113.00** | 110.48 | 116.16 | 5.0% | 3,397 MiB | 3.35 GB |
| 8B bf16 streamed | 5 | 62.43 | 62.20 | 63.03 | 1.3% | 3,935 MiB | 15.0 GB |
| 14B NF4 streamed | 5 | 118.60 | 115.55 | 120.37 | 4.1% | 4,475 MiB | 6.35 GB |
| 32B NF4 streamed | 5 | 76.80 | 75.71 | 78.70 | 3.9% | 4,845 MiB | 14.99 GB |
| 8B bf16 ZeRO-3 offload | 2 | 19.79 | 19.74 | 19.84 | 0.5% | 38,111 MiB | — |

SM clock 1980 MHz median-while-busy in all 26 runs; no clock variation to
correct for, unlike the laptop record where it moved 13% between sessions.

Three things fall out of this table.

**Peak VRAM is flat in model size.** 8B -> 14B -> 32B moves peak VRAM
3,397 -> 4,475 -> 4,845 MiB while the model grows 4x. That is the whole claim of
the feature, measured across a 4x size range on one card, and it holds. The
growth that does occur is the logits tensor and the per-layer buffers, not the
weights.

**14B is faster than 8B (118.60 vs 113.00), which is the opposite of the naive
expectation.** Not investigated properly, so no mechanism is asserted; the
obvious candidate is that these are different architectures (Llama-3.1 vs
Qwen2.5) with different vocab sizes and layer shapes, so "8B vs 14B" is not a
clean size sweep. The clean sweep is the three Qwen2.5 rows, and there
14B -> 32B does drop throughput 118.60 -> 76.80 as the per-layer transfer grows.

**Streaming NF4 on an H100 runs at essentially the speed it ran on a laptop.**
The v0.72.2 record has Llama-3.1-8B NF4 at **119.6 tok/s in a 3.32 GB peak** on
an RTX 3050 Laptop. Here, on a card with roughly two orders of magnitude more
compute, the same model and configuration give a median **113.00 tok/s in a
3.32 GB peak** (3,397 MiB). Within a few percent of each other, and the H100 is
if anything slightly slower.

That is not a disappointing result, it is the mechanism being confirmed: the
method is bound by host-to-device transfer of the layer store, not by the GPU.
The 54.1% mean GPU utilisation on the NF4 rows says the same thing — the card is
idle half the time waiting for weights. It also explains why bf16 streaming is
*slower* than NF4 streaming (62.43 vs 113.00) despite being simpler arithmetic:
NF4 moves roughly a quarter of the bytes.

This is the strongest cross-hardware evidence in the record that the published
laptop numbers were not an artefact of that laptop.

---

## DISK — the RAM-vs-NVMe question, and why it is not answered here

The brief said not to bypass the gate silently: measure the disk first, then
decide, then write the decision down.

**The device.** One virtual `vda`, and the kernel reports it as rotational:

```
$ lsblk -d -o NAME,ROTA,MODEL,SIZE
NAME ROTA MODEL         SIZE
vda     1                 1T
$ cat /sys/block/vda/queue/rotational
1
```

**What it actually does**, with O_DIRECT so the page cache is out of the way:

```
$ dd if=/dev/zero of=/root/ddtest bs=1M count=2048 oflag=direct
2147483648 bytes (2.1 GB) copied, 3.92109 s, 548 MB/s
$ dd if=/root/ddtest of=/dev/null bs=1M iflag=direct
2147483648 bytes (2.1 GB) copied, 1.44622 s, 1.5 GB/s
```

1.5 GB/s sequential read is not a spinning disk. It is a virtio device backed by
something fast on the host, exposing no rotational hint, so the guest kernel
defaults `rotational` to 1.

**What Soup decides:**

```
>>> from soup_cli.utils.layer_stream import detect_disk_kind
>>> detect_disk_kind()
'hdd'
```

So `stream_source: disk` is refused, and this is a **third finding, milder than
the first two**: `detect_disk_kind` classifies from `BusType`/`MediaType`, which
a virtio device does not supply, and the fallback lands on `hdd`. On a cloud VM —
which is where most people who need this feature will run it — a genuinely
NVMe-backed disk can be refused the overflow tier. The guard is not wrong to be
conservative, but it currently cannot see through virtio.

**Decision: not bypassed.** Two reasons, and the second is the real one.
Bypassing needs the shipped classifier overridden, and a number measured on a
1.5 GB/s virtio device would not be an answer about NVMe anyway — it would be an
answer about this hypervisor's storage, published under a heading people would
read as "NVMe". The v0.72.3 record's statement that the RAM-vs-disk gap is
unmeasured stands, and this box does not change it.

What can be said, and is worth saying: the measured 1.5 GB/s read is in the same
range as the host-to-device transfer that the throughput numbers above show to be
the binding constraint, so a disk tier on storage like this would plausibly not
be catastrophic. That is an inference from two measurements, not a measurement,
and is flagged as such.

---

## STEP 5 — 72B, and then the root cause

72B was the explicit bonus, to be attempted only if everything before it went
cleanly. It did, and it turned out to be the run that cracked the 32B defect.

### Qwen2.5-72B-Instruct, NF4 — forward bit-exact, 80/80 layers trained

```
params 72,706,203,648   layers 80   store 33.74 GB pinned   shard 185.4 s
copied 320 adapter tensors
max_abs_logit_diff 0.0   bit_exact True   logit_abs_max 38.5
layer0_lora_grad 7.973011e-02   80/80 layers non-zero
curves_equal False   curve_max_rel 0.129
```

The store is 33.74 GB against the box's 62.96 GB page-locked ceiling, so it
pins. Sharding took 185.4 s. And the throughput:

```
v_72b r1  tok/s 37.94  peak 7,411 MiB  util 67.4%  clk 1980
v_72b r2  tok/s 37.70  peak 7,413 MiB  util 88.9%  clk 1980
```

**A 72B model fine-tuned in 7.2 GB of VRAM.** With the caveat that follows,
which is not a footnote: at 72B the gradients are wrong under the shipped
configuration, so that line describes a memory and throughput result, not a valid
training result.

The defect is worse at 72B than at 32B — even the *first* backward is corrupted:

```
resident reference loss 13.041974068  (320 grad tensors)
rep 1: loss 13.041974068 (==resident True)  grads exact 8/320  [78, 79]  worst_rel 8.8182
rep 2: loss 13.041974068 (==resident True)  grads exact 8/320  [78, 79]  worst_rel 8.8182
rep 3: loss 13.041974068 (==resident True)  grads exact 8/320  [78, 79]  worst_rel 8.8182
```

Same signature: forward loss bit-identical to resident every time, survivors
exactly the last `buffers` layers.

### The one experiment that named the cause

Everything so far said "race in the layer pool". The way to test that directly
is to remove the asynchrony: page-locked host memory is what makes a
host-to-device `copy_` genuinely asynchronous, so with **pinning off** the copies
become synchronous and any missing dependency stops mattering.

Qwen2.5-32B, identical in every other respect:

```
##### 32B pin=0
rep 1: grads exact 256/256  64 layers  worst_abs 0.000000e+00
rep 2: grads exact 256/256  64 layers  worst_abs 0.000000e+00
rep 3: grads exact 256/256  64 layers  worst_abs 0.000000e+00

##### 32B pin=1   (the shipped configuration)
rep 1: grads exact 256/256  64 layers  worst_abs 0.000000e+00
rep 2: grads exact   8/256  [62, 63]   worst_abs 1.940805e-01  worst_rel 8.7051
rep 3: grads exact   8/256  [62, 63]   worst_abs 9.720615e-02  worst_rel 8.4925
```

And at 72B, where pinning corrupts even the first pass:

```
##### 72B pin=0
rep 1: grads exact 320/320  80 layers  worst_abs 0.000000e+00
rep 2: grads exact 320/320  80 layers  worst_abs 0.000000e+00
rep 3: grads exact 320/320  80 layers  worst_abs 0.000000e+00
```

**With pinning disabled, layer streaming is bit-exact at 72B.** Every model
tested — 0.5B, 8B, 14B, 32B, 72B — produces gradients bit-identical to a
resident reference of the same numerics.

So the correct statement is not "layer streaming breaks above 14B" — the
algorithm is exact at every size measured, up to 72B, once pinning is off. The
next step was to find what actually triggers it.

---

## STEP 6 — isolating the trigger, including four hypotheses that were wrong

The natural first story — *a race whose window opens once the per-layer transfer
is large enough* — is wrong. It was tested four ways and survived none of them.
All of the following are on synthetic from-config Llamas unless a real model is
named, streamed vs a resident reference of matching numerics, adapters synced,
pinning on.

**Rejected 1 — compute-vs-transfer ratio.** If a longer forward gave the copy
time to land, sequence length would move the outcome. It does not, in either
direction:

```
14B NF4  seq 8    exact 192/192   (shorter compute -- should have broken it)
14B NF4  seq 16   exact 192/192
14B NF4  seq 128  exact 192/192
14B NF4  seq 512  exact 192/192
32B NF4  seq 128  WRONG  8/256
32B NF4  seq 512  WRONG  8/256    (4x the compute -- should have fixed it)
```

**Rejected 2 — per-layer bytes.** 8B in bf16 moves 480 MiB per layer, more than
twice what a broken 32B NF4 layer moves, and is exact:

```
8B  bf16  32 layers  ~480 MiB/layer  exact 128/128 x3
14B bf16  48 layers  ~570 MiB/layer  exact 192/192 x3
32B nf4   64 layers   234 MiB/layer  WRONG   8/256
```

**Rejected 3 — total store bytes.** 8B bf16's store is 15.0 GB and exact; 32B
NF4's is 14.99 GB and broken. Near-identical stores, opposite outcomes.

**Rejected 4 — layer count.** Synthetic Llamas swept across the 48 -> 64
boundary are exact at every depth, in both quantisations, as long as the layers
are small:

```
tiny   bf16  0.05 MiB/layer   32, 48, 56, 64, 80, 96 layers -> all EXACT
tiny   nf4   0.01 MiB/layer   32, 48, 56, 60, 64, 80 layers -> all EXACT
synth  bf16  84.5 MiB/layer   32, 48, 56, 64, 80 layers     -> all EXACT
synth  bf16   338 MiB/layer   32, 48, 64 layers             -> all EXACT
```

### What it actually is: the NF4 path, above a per-layer size threshold

The discriminating experiment holds the model shape **identical** and changes
only the quantisation. 32 layers, hidden 5120, intermediate 27648:

| quant | MiB/layer | store | result |
|---|---|---|---|
| bf16 | **935.0** | 29,921 MiB | **EXACT** `128/128 x3` |
| NF4 | 241.2 | 7,718 MiB | **WRONG** `24/128, 8/128, 8/128` |

bf16 is exact while moving **3.9x more bytes per layer** than the NF4 run that
fails. Whatever this is, it is not a transfer-size race in the general copy path.
It is in the NF4 path.

Bracketing the NF4 threshold, at a fixed 48 layers, hidden 5120, varying only
`intermediate_size`:

| intermediate | MiB/layer | store | result |
|---|---|---|---|
| 13,824 *(Qwen2.5-14B's own shape)* | 136.7 | 6,563 MiB | **EXACT** `192/192 x3` |
| 15,360 | 148.3 | — | **EXACT** `192/192 x3` |
| 17,408 | 163.8 | — | **EXACT** `192/192 x3` |
| 18,432 | **171.5** | — | **WRONG** `192/192, 8/192, 8/192` |
| 19,456 | 179.3 | — | **WRONG** `8/192 x3` |
| 20,480 | 187.0 | 8,977 MiB | **WRONG** `192/192, 8/192, 8/192` |
| 27,648 *(Qwen2.5-32B's own shape)* | 241.2 | 11,577 MiB | **WRONG** `8/192 x3` |

The 13,824 row is a synthetic reconstruction of the real 14B's layer shape and
it is exact, exactly as the real 14B is; the 27,648 row reconstructs the real
32B's and is broken, exactly as the real 32B is. The behaviour follows the
**shape**, not the checkpoint — so this is reproducible without downloading a
32B model.

**Threshold: between 163.8 and 171.5 MiB per NF4 layer** — a 4.7%-wide bracket,
monotone on both sides with no exceptions in seven points. The real models sit
either side of it exactly as predicted: 8B at 105 and 14B at 132 MiB/layer are
below and exact, 32B at 234 and 72B at 432 are above and broken.

That the boundary is this sharp, and this reproducible on synthetic weights,
argues against a pure timing race — a race would smear across a range and vary
between runs at the boundary. Something is switching behaviour at a size, and
what it is was not identified here.

Note the 48-layer/187 MiB row also kills the last version of the depth story: it
is broken at 48 layers, the same depth at which the real 14B is exact.

### The smallest reproducer found

```python
LlamaConfig(vocab_size=4096, hidden_size=5120, intermediate_size=27648,
            num_hidden_layers=32, num_attention_heads=40, num_key_value_heads=10,
            tie_word_embeddings=False, max_position_embeddings=512)
# shard NF4, stream with pin=True, buffers=2
# backward twice on the same input -> second one disagrees with a resident
#   NF4 reference on every layer except the last 2
```

7.7 GB store, runs in about a minute, no download.

### Stated as precisely as the evidence allows

- The **forward is bit-exact in every configuration tested**, at every size, in
  both quantisations, broken or not. Nothing in a loss curve can reveal this.
- The **backward** produces gradients that disagree with a deterministic resident
  reference on every layer except the last `stream_buffers`, when the base is
  **NF4** and the layer exceeds a threshold measured at **163.8–171.5 MiB**.
  Usually from the second backward onward; at 72B from the first.
- **Below that threshold it does not drift with use.** 50 consecutive backwards
  at 8B (105 MiB/layer) and at 14B (132 MiB/layer), each diffed against the same
  resident reference: `worst_abs = 0.0` on all 50 of each, all tensors. The
  published claims sit on the safe side of the boundary with margin, and stay
  there under repetition.
- **Turning pinning off makes it exact**, at 32B and at 72B. Pinned memory is
  what makes a host-to-device `copy_` genuinely asynchronous, so the missing
  dependency is between the NF4 layer's arrival and its use — most plausibly the
  `Params4bit` / `quant_state` views that `rebuild_quant_state` and
  `rebuild_params4bit` construct over the pooled buffer, which for an NF4 layer
  means several sidecar tensors (`::absmax`, `::nested_absmax`, `::nested_offset`)
  per weight rather than one. That last clause is a hypothesis pointing at the
  likely code, not a measurement.
- `LayerBufferPool.wait()` is documented as the ownership check that prevents
  exactly this, and `load_async` as issuing `stream.wait_stream(current)` before
  `copy_`. Both are evidently insufficient on this path.

**No fix attempted and no Soup code changed**, per the brief. There is also no
user-facing escape hatch: pinning is chosen automatically by `decide_pinning`,
with no config key to disable it, so a user hitting this cannot work around it
from `soup.yaml`.

### Everything measured, one table

**Read the two exactness columns separately.** "Bit-exact" in this record is
never a single verdict about a model: the **forward** (logits, `torch.equal`)
and the **backward** (every LoRA gradient tensor, against a resident reference of
matching numerics) are measured independently, and above the threshold they
disagree — the forward is exact while the backward is not. Two external readers
took an unqualified "bit-exact at 8B / 14B / 32B / 72B" to mean gradients too,
computed 72B's NF4 layer at ~449 MiB against the 165 MiB threshold, and reported
a contradiction. The record was literally correct and still misread; this table
is the fix.

| model | quant | MiB/layer | **forward** (logits) | **backward** (gradients), pinned | backward, `pin=False` | 5-step loss curve |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | NF4 | 7 | exact `0.0` | exact 96/96 | — | identical |
| Llama-3.1-8B | NF4 | 105 | exact `0.0` | exact 128/128 *(50 backwards)* | — | identical |
| Llama-3.1-8B | bf16 | 480 | exact `0.0` | exact 128/128 | — | — |
| Qwen2.5-14B | NF4 | 132 | exact `0.0` | exact 192/192 *(50 backwards)* | — | identical |
| Qwen2.5-14B | bf16 | 570 | exact `0.0` | exact 192/192 | — | — |
| Qwen2.5-32B | NF4 | 234 | exact `0.0` | **WRONG 8/256** | exact 256/256 | **diverged**, rel 0.0586 |
| Qwen2.5-72B | NF4 | 432 | exact `0.0` | **WRONG 8/320** | exact 320/320 | **diverged**, rel 0.1291 |

Throughput and peak VRAM, kept separate because they are a different claim:

| model | tok/s (median) | peak VRAM | store (pinned) |
|---|---|---|---|
| Llama-3.1-8B NF4 | 113.00 *(n=5)* | 3,397 MiB | 3.35 GB |
| Qwen2.5-14B NF4 | 118.60 *(n=5)* | 4,475 MiB | 6.35 GB |
| Qwen2.5-32B NF4 | 76.80 *(n=5)* | 4,845 MiB | 14.99 GB |
| Qwen2.5-72B NF4 | 37.94 *(n=2)* | 7,411 MiB | 33.74 GB |

The 72B row is n=2 and should be read as indicative.

### The loss curves independently confirm the split

Protocol check (4) of STEP 2 was "5-step loss curves identical", and it is worth
stating what it did rather than leaving it in a distant table. If the gradients
above the threshold are wrong, the curves **must** diverge; if they were exact,
the curves must match. They track perfectly:

```
model  forward diff   curves_equal   curve_max_rel
8b     0.0            True           0
14b    0.0            True           0
32b    0.0            False          0.05857
72b    0.0            False          0.12909
```

Step 0 of every curve is identical in every row — which it must be, given the
bit-exact forward — and the divergence starts only after the first optimizer
step, i.e. at the first point where a gradient can matter. That is an independent
confirmation of the diagnosis using a quantity measured before the defect was
even suspected.

---


## STEP 7 — is the NF4 defect the same bug as #328? **No.**

Proposed after the fact, and worth testing because the two look alike: #328's
traceback dies inside `torch.utils.checkpoint.backward` on a weight still on
`meta`, and the NF4 defect has the same shape of symptom — forward bit-exact,
backward wrong on every layer but the last `stream_buffers`. Both are "the second
pass sees something the first one did not". If they shared a root, one fix would
close both.

### Attempt 1 — turn gradient checkpointing off. Inconclusive, and the reason is interesting

`StreamedDecoderLayer` already carries a `use_checkpoint` attribute that its
`forward` reads, so it can be flipped on a built model without touching `src/`.
Single variable, both arms every time, synthetic reproducer (32 layers, hidden
5120, intermediate 27648, NF4, `pin=True`):

```
checkpointing ON  (control) -> WRONG  ['128/128', '8/128', '8/128']   layers flipped=32
checkpointing OFF (test)    -> ERROR  RuntimeError: one of the variables needed for
    gradient computation has been modified by an inplace operation:
    [CUDABFloat16Type [5120]] is at version 16; expected version 15
```

The control reproduces the defect, so the arm is live. The test arm did not turn
bit-exact and did not stay wrong — **it crashed**, which answers neither branch
of the hypothesis as posed.

But the crash says something precise. `[5120]` is a bf16 tensor of exactly
`hidden_size` — a layernorm weight, which an NF4 shard keeps in bf16 inside the
pooled buffer. Autograd's version counter caught that buffer being **overwritten
while the backward still needed it**. So the pool does recycle buffers that the
backward depends on, and gradient checkpointing is what converts that from a
detected error into silently wrong numbers.

That also makes the arm *invalid as a test of the hypothesis*: the pooled-buffer
design **requires** recompute, because without it autograd holds views into
buffers that are guaranteed to be reused. Turning checkpointing off is not a
configuration the design supports, so its failure is the documented consequence
of the design, not evidence about where the bug lives.

Kept as run. It is also the most useful by-product of the session for whoever
fixes this: **autograd will detect the reuse on its own when checkpointing is
not hiding it**, which is a ready-made assertion for a regression test.

### Attempt 2 — separate them by trigger conditions instead. Decisive

If the two bugs were one, they would fire under the same conditions. They do not
share a single one. The #328 reproducer runs on the *tiny* test checkpoint
(hidden 64, 2 layers, **0.05 MiB per layer**), which is four orders of magnitude
below the NF4 defect's threshold, and `_stream_cfg` sets `quantization: none`:

```
torch 2.13.0+cu130
quant=none  task=sft   OK
quant=none  task=dpo   FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
quant=none  task=kto   FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
quant=4bit  task=sft   OK
quant=4bit  task=dpo   FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
quant=4bit  task=kto   FAIL RuntimeError: Tensor on device cuda:0 is not on the expected device meta!
```

(The `orpo` rows are missing from this capture — a harness output problem, not a
result. The earlier full reproducer had orpo and simpo failing identically.)

| | #328 | the NF4 defect |
|---|---|---|
| quantisation | fires on **both** `none` and `4bit` | **NF4 only** — bf16 exact at 935 MiB/layer |
| model size | fires at **0.05 MiB/layer** | needs **> ~165 MiB/layer** |
| task | **preference losses only**; SFT passes | fires on **SFT** |
| manifestation | **crashes** with a `meta` tensor | **silent** wrong gradients |
| pinning off | not tested — it crashes before that matters | **fixes it** |

Not one axis in common, and each is the other's control: SFT is the task where
#328 does not fire and the NF4 defect does; bf16-tiny is the configuration where
#328 fires and the NF4 defect cannot.

**Hypothesis rejected. Two different bugs**, joined onto the list with the four
already rejected in STEP 6. They may still share an ancestor — both are the
streamed layer misbehaving on a second pass — but they are not one fix.

---

## STEP 8 — what does turning pinning off cost? **6.56x**

`pin=False` is the only thing found that makes the NF4 defect go away, so it is
the obvious candidate for an automatic mitigation: detect an NF4 layer above the
threshold, fall back to pageable host memory. Whether that is a reasonable
fallback or a configuration that should simply be refused is a question about a
number, so here is the number.

Both arms are the same harness on the same shards, alternating within one
process, `Qwen2.5-32B-Instruct` NF4, seq 256, batch 1, 20 timed steps after 5
warm-up, `torch.cuda.synchronize` around each step, n=3 per arm.
**Correctness is asserted inside each arm**, so a fast arm cannot quietly be the
wrong one — the gradient check runs in the same process as the timing.

| arm | rep | tok/s | median step | gradients | verdict |
|---|---|---|---|---|---|
| `pin=True` | 1 | 408.88 | 0.6261 s | 8/256 | **WRONG** |
| `pin=True` | 2 | 425.42 | 0.6018 s | 8/256 | **WRONG** |
| `pin=True` | 3 | 425.07 | 0.6022 s | 8/256 | **WRONG** |
| `pin=False` | 1 | 65.25 | 3.9232 s | 256/256 | correct |
| `pin=False` | 2 | 64.41 | 3.9745 s | 256/256 | correct |
| `pin=False` | 3 | 64.79 | 3.9513 s | 256/256 | correct |

```
median pin=True  425.07 tok/s
median pin=False  64.79 tok/s
cost of turning pinning off: 6.56x slower   (division of the two medians)
```

Spread within each arm is small (4.0% pinned, 1.3% pageable), so n=3 is enough to
carry the ratio.

**Reading it.** 6.56x is not a fallback, it is a different feature. The premise
of layer streaming is that a model you could not otherwise train becomes
trainable at a tolerable slowdown; multiplying that slowdown by another 6.6x
silently, to work around a bug, spends the entire margin the method exists to
provide. On the framing the question was posed with — cheap means fall back with
a loud notice, expensive means refuse — this lands clearly on **refuse**:
declining an NF4 configuration above the threshold, with the reason stated, is
more honest than shipping something correct and six times slower without telling
anyone. That said, the call belongs to whoever fixes this, and a *third* option
now exists that neither branch of the question anticipated: STEP 7 showed
autograd detects the underlying buffer reuse by itself once checkpointing is not
masking it, so a real fix may be cheaper than either workaround.

**Three caveats on these numbers, none of which touch the ratio.**

1. **The tok/s are not comparable to the `soup train` figures elsewhere in this
   record.** This harness replays one fixed padded batch with no data pipeline
   and counts all 256 positions; `soup train` counts real tokens on short rows.
   That is why 425 tok/s appears here against 76.80 tok/s for the same model in
   STEP 4. Only the within-experiment ratio is being claimed.
2. **The peak VRAM column was dropped from the table on purpose.** The harness
   holds the resident reference alongside the streamed model in order to check
   correctness, so its peak (26,713 MiB) measures both models, exactly the
   invalid-measurement trap already recorded at 8B. It is not a streaming peak.
3. **The SM clock this harness recorded is wrong** — it reads once immediately
   after a `synchronize`, before the timed loop, and got 345 MHz (the idle
   clock) on every row. The `nvidia-smi`-sampled runs elsewhere show 1980 MHz
   while busy. Both arms are affected identically, so the comparison stands, but
   no absolute throughput from this table should be quoted against a clock.

Not measured: whether 6.56x holds on slower host-to-device links. This box is
PCIe (PIX between all pairs, no NVLink); a machine with less host bandwidth
would likely show a *smaller* ratio, because the pinned arm has less advantage
to lose.

---

## STEP 9 — the mechanism: **aliasing, not a race**

STEP 6 rejected four hypotheses and STEP 7 rejected a fifth, but none of them
produced a mechanism. Two facts made the obvious "race" story suspect: reading
the shipped code shows `_body` calls `pool.wait(idx)` **and** builds the
substituted weights *inside* the checkpointed function, so the recompute does
re-wait and does rebuild — and the 163.8/171.5 MiB boundary is far too sharp for
a timing window, which would smear.

So the two candidate mechanisms were separated by two patches to `_body`,
applied to the generated class at runtime (nothing under `src/` touched), each
with the unpatched control in the same run:

| arm | what it changes | fixes a race? | fixes aliasing? |
|---|---|---|---|
| `control` | nothing | — | — |
| `sync` | `torch.cuda.synchronize()` immediately after `pool.wait` | yes | no |
| `clone` | hand `functional_call` a private copy of every pooled tensor | no | yes |

```
control  -> WRONG  ['48/128', '8/128', '8/128']
sync     -> WRONG  ['12/128', '12/128', '8/128']
clone    -> EXACT  ['128/128', '128/128', '128/128']
```

**A full device synchronise does not fix it. Cloning the buffers fixes it
completely.** This is not a timing race. It is **aliasing**: an autograd node
still holds a reference to a pooled buffer when a later prefetch overwrites it,
so the backward reads another layer's bytes. That is also exactly what STEP 7's
`use_checkpoint=False` crash was — autograd's version counter catching the same
overwrite once checkpointing stopped hiding it — and it retires the "race"
wording used earlier in this record.

It also explains fact (3) directly: the only layers that survive are the last
`stream_buffers`, which are the ones never evicted before the backward reaches
them.

### Which tensors are aliased: both kinds, and neither alone is enough

An NF4 layer arrives as packed `uint8` weights plus small sidecars
(`::absmax`, `::nested_absmax`, `::nested_offset`). Cloning one group and leaving
the other aliased:

```
control          -> WRONG  ['40/128', '8/128', '8/128']
clone_quantstate -> WRONG  ['8/128',  '8/128', '8/128']   (sidecars only)
clone_packed     -> WRONG  ['8/128',  '8/128', '8/128']   (packed weights only)
```

Neither half is sufficient; only cloning **everything** is. So the aliasing is a
property of the whole pooled NF4 tensor set, not of one forgotten sidecar — which
rules out the tidiest possible fix (copy the tiny quant-state, keep the big
weights zero-copy) and matters for anyone costing a real repair.

### What is still not explained

Facts (1) and (2) — NF4-only, and the sharp size threshold — remain open. The
bf16 path aliases the pool in exactly the same way (`_substituted_weights`
returns `buffers[ckpt]` itself) and is exact at 935 MiB/layer, 3.9x the bytes at
which NF4 fails. Something about how bitsandbytes' 4-bit matmul retains its
operands across the backward differs from a plain `F.linear`, and this session
did not establish what. **No mechanism is claimed for (1) and (2)**; what is
claimed is that the failure is aliasing and that de-aliasing removes it.

Practical consequence for a fix: the cheap-looking repair (clone the pooled
tensors) costs one extra copy of every streamed layer per forward, which is the
thing the pre-allocated pool exists to avoid. It was then measured — see below,
and the answer is not the one this paragraph expected.

### What the real fix would cost — and why the cheap one is not a fix

STEP 8 measured `pin=False` at 6.56x and concluded that a silent fallback was too
expensive. Cloning looked much more promising on first principles: a clone is a
device-to-device copy at HBM bandwidth, duplicating bytes that just arrived over
PCIe, so it should be nearly free. Measured, same process, correctness asserted
in each arm so a fast arm cannot quietly be the wrong one (synthetic 32-layer /
5120 / 27648 NF4, seq 256, 15 timed steps after 4 warm-up, n=2):

| arm | gradients | tok/s | vs control | **peak VRAM** |
|---|---|---|---|---|
| control *(shipped)* | **WRONG** 8/128 | 849.62 | 1.00x | **1,132 MiB** |
| clone *(de-alias)* | correct 128/128 | 801.22 | **1.06x** | **9,070 MiB** |
| `pin=False` | correct 128/128 | 114.70 | 7.41x | 1,129 MiB |

The time cost was the easy half and it is indeed almost nothing: **6%**, against
7.41x for turning pinning off. On wall-clock alone the clone wins overwhelmingly.

**And it is still not a fix, because of the last column.** Peak VRAM goes
1,132 -> 9,070 MiB, an **8x increase**, and 9,070 MiB is approximately the whole
model (32 layers x 241 MiB = 7.7 GB). That is the entire premise of layer
streaming deleted: peak VRAM is supposed to be bounded by one layer, and a
private copy per layer makes it bounded by the model instead. A "fix" that makes
the feature pointless is not a fix.

That number is also the sharpest statement of the defect available. If every
layer's weights must stay alive until the backward reaches that layer, then all
of them are alive at once — which is precisely what streaming exists to avoid.
The recompute is *supposed* to re-fetch instead, and the measurement says the
re-fetched buffer does not survive to its own backward.

So the fix has to keep a buffer alive across exactly its own
recompute-plus-backward window and no longer. Raising `stream_buffers` is not
that: at 8 buffers exactly 8 layers survive, so reaching correctness that way
means one buffer per layer, i.e. the whole model again.

**Handover, in one line:** the defect is aliasing; de-aliasing costs 6% time but
8x VRAM; `pin=False` costs 7.41x time at no VRAM cost; the correct repair is
neither, and was not found here.

### Hypothesis 8 — separate forward / backward pools. **Rejected, and it says where the fix is**

ZeRO-3 and FSDP2 have the same collect-use-release structure and do not solve it
by holding the buffer. DeepSpeed re-fetches in the backward
(`pre_sub_module_backward_function -> fetch_sub_module(sub_module, forward=False)`)
and — the load-bearing detail — gates `release_sub_module` on that sub-module's
**backward finishing**, counting inputs with `requires_grad` in
`ds_grads_remaining`, not on the next prefetch arriving. FSDP2 does the same with
`resize_(0)` after forward and a refill in the pre-backward hook; the RFC
(pytorch#114299) describes our exact situation — autograd packs a reference to
the unsharded parameter, FSDP frees the storage behind its back and restores it
before the backward. Peak memory is O(window), not O(model).

So the 8x of the clone route was the price of the *wrong* repair, not of repair.
The cheapest test of the cheap half: give the backward its own pool, so what the
graph looks at is not in the rotation the forward evicts from.

Prototyped by replacing `_body` at runtime (nothing under `src/`): when
`prefetcher.direction == -1`, read from a second `LayerBufferPool` filled
**synchronously** from the same source. Synchronous deliberately — a dedicated
CUDA stream would open a genuine cross-stream race, which is a *different* bug
class from the aliasing diagnosed here and must not be mixed into it
(cf. torchtune#1867: NaNs from a race deleting a tensor in a streamed offload).

```
control  bwd=0  -> WRONG  grads 8/128  866.86 tok/s  peak 1132 MiB
seppool  bwd=2  -> WRONG  grads 8/128  540.21 tok/s  peak 1612 MiB   branches={'fwd': 170, 'bwd': 150, 'bwd_fill': 150}
seppool  bwd=4  -> WRONG  grads 8/128  547.06 tok/s  peak 2094 MiB
```

**The patch fired** — 150 backward-branch calls, and `bwd_fill == bwd` means every
one of them had to fetch, i.e. the backward pool never already held the layer.
That instrumentation matters: without it an unfired patch is indistinguishable
from a failed hypothesis, and the first (uninstrumented) run of this experiment
returned exactly the control's `8/128` in all three arms, which is precisely what
an inert patch would also produce.

**Why it fails, and this is the useful part.** The second pool rotates on the
same `slot_for(idx) = idx % n`, so layer *i* is evicted from the *backward* pool
when layer *i-2* is filled. The eviction is not between the forward and the
backward — it is **inside the backward walk**. Separating the pools duplicated
the eviction rather than removing it.

Which isolates the load-bearing half of the ZeRO-3 design: **the re-fetch is not
the fix, the release-gating is.** A layer's weights are still required after its
own recompute, during the gradient computation that consumes them, and the very
next layer's fetch takes the slot. Gating release on "this sub-module's backward
has finished" is exactly the piece this prototype omitted, and exactly the piece
DeepSpeed spends a gradient counter on.

Two numbers worth keeping for whoever builds it: the extra host-to-device fetch
per layer per step cost **1.60x** here (866.86 -> 540.21 tok/s), in the same range
as the ~+50% I/O predicted from the method being bus-bound; and peak VRAM grew by
exactly one extra pool (1,132 -> 1,612 MiB), i.e. **O(window)** as intended — not
the 8x of the clone. So the memory shape of the ZeRO-3 approach is confirmed even
though this prototype's correctness is not.

Hypothesis 2 (explicit re-fetch with gated release) and the
`saved_tensors_hooks` alternative are **not attempted here** — both need changes
inside `src/`, and `src/` currently carries another agent's uncommitted work.

### Hypothesis 9 — re-fetch with release gated on the layer's own backward. **Rejected**

The half Hypothesis 8 omitted, and the one DeepSpeed spends a gradient counter
on. The signal is cheaper than a counter here: walking the backward L-1..0, the
gradient w.r.t. layer *i*'s **input** is the last thing its backward produces, so
`hidden_states.register_hook(...)` fires exactly at `PostBackwardFunction` time.

Prototype: forward keeps the shipped rotating pool; the backward takes a buffer
set from a free list of K, fills it synchronously from the same source, and
returns it from the input hook. Exhausting the free list raises rather than
silently reusing — silent reuse is the bug being fixed. Layer 0 gets no input
gradient (frozen embeddings) so it has no "done" signal; it is last in the walk,
so it simply holds its set to the end of the step.

```
control  k=0 -> WRONG 8/128  867.38 tok/s  peak 1132 MiB
gated    k=2 -> WRONG 8/128  546.64 tok/s  peak 1612 MiB  {'acquire':510,'release':493,'hooked':493,'unhooked':17,'max_held':1}
gated    k=4 -> WRONG 8/128  556.42 tok/s  peak 2094 MiB  {... 'max_held':1}
gated    k=8 -> WRONG 8/128  542.97 tok/s  peak 3059 MiB  {... 'max_held':1}
```

The gate works mechanically — 493 hooks fired, 17 unhooked (one per step, layer
0), and `k` is never the constraint. The telling number is **`max_held: 1`**:
never more than one buffer set is held at a time, i.e. the "this layer is
finished" signal always arrives before the next layer asks. So under the ZeRO-3
model there is no overlap at all, and the release is not premature by that
model's own definition — yet the gradients are unchanged.

**Holding longer does not help either.** Delaying the release by N further
layers (a ring on top of the free list) moves `max_held` 1 -> 2 -> 3 -> 4 exactly
as intended and moves nothing else:

```
gated delay=0 -> WRONG 8/128  max_held 1
gated delay=1 -> WRONG 8/128  max_held 2
gated delay=2 -> WRONG 8/128  max_held 3
gated delay=3 -> WRONG 8/128  max_held 4
```

### The result that reframes all of it: the count tracks the FORWARD pool

Sweeping `stream_buffers` (the **forward** pool) while the backward re-fetches
into its own gated buffers:

| `stream_buffers` | control | gated backward |
|---|---|---|
| 2 | WRONG 8/128 | WRONG 8/128 |
| 4 | WRONG 16/128 | *(see below)* |
| 8 | WRONG 32/128 | WRONG 32/128 |

The control column is the familiar `4 x buffers`. The gated column matches it at
2 and at 8 — **the backward-side intervention does not move the outcome**. What
governs how many layers come out right is the size of the pool the *forward* used,
not what the backward reads. Every backward-side fix tried so far (separate pool,
gated re-fetch, delayed release) is therefore aiming at the wrong half.

### And a near-miss that repetition destroyed

The `buffers=4` gated arm returned **`CORRECT 128/128`** on its first run — the
success criterion, met. Repeating the identical configuration:

```
run 1 (original)  128/128   CORRECT
run 2             100/128
run 3              40/128
run 4              64/128
run 5              16/128
run 6              40/128
```

Six observations of one configuration spanning 16 to 128. The `CORRECT` was
luck. Had it been published on n=1 it would have read as a working fix.

That scatter is itself a signal and it is worth separating from the aliasing
diagnosis: the shipped control is **deterministic** (STEP 2b: reps 2-5 byte-identical),
while this prototype is not. Nothing here added a CUDA stream — the fills are
synchronous on the compute stream on purpose — but the extra per-layer copy
changes the interleaving, and the outcome became run-dependent. This is the
second bug class flagged in advance (cf. torchtune#1867) showing up as soon as
the schedule is perturbed, and it is *not* the aliasing bug: it must not be
folded into the same diagnosis.

**Where this leaves the repair.** Both prototypes were built on the premise that
the backward reads stale weights. The forward-pool tracking says that premise is
at best incomplete. The remaining untested route is the one that does not depend
on it at all — `torch.autograd.graph.saved_tensors_hooks`, packing a layer index
and materialising in `unpack_hook`, which severs whatever reference the graph is
actually holding rather than guessing where it points. Not attempted here: it
needs changes inside `src/`, and `src/` is shared with another agent right now.

### THE MECHANISM, named in the library that causes it

Two fix attempts had been rejected and the forward-pool sweep had shown the
premise behind both ("the backward reads stale weights") was wrong. So instead of
a third guess: ask the graph what it holds.

**Step 1 — `saved_tensors_hooks`, purely to observe.** Installed around the
forward+backward, recording every packed tensor whose *storage pointer* falls in
a pool buffer (storage, not identity: a `Params4bit` view, a `.t()`, a slice all
share it).

```
quant=nf4  layers=32 buffers=2 -> packed 42, aliasing the pool 0
quant=none layers=32 buffers=2 -> packed 42, aliasing the pool 0
```

Zero, in both quantisations. So nothing *outside* the checkpointed regions holds
the pool. (Hooks were not installed inside those regions: non-reentrant
`checkpoint` is itself implemented with `saved_tensors_hooks`, and nesting inside
would replace the mechanism being measured.)

**Step 2 — walk the graph instead.** Read-only, after the forward, before the
backward. `fn.saved_tensors` is unreadable on most C++ nodes (2329
`AttributeError`s), so the census of node *types* is what carried the result:

```
quant=nf4  nodes=2550 saved=0 aliasing=0
   node types: {'MulBackward0': 350, 'ViewBackward0': 255, 'ToCopyBackward0': 255,
                'AddBackward0': 255, 'MatMul4BitBackward': 221, 'MmBackward0': 129, ...}
```

**221 `MatMul4BitBackward` nodes exist in the graph after the forward** — 32
layers x 7 quantised linears = 224. They are built during the forward and they
survive it.

**Step 3 — read what those nodes hold.** `bitsandbytes/autograd/_functions.py`,
`MatMul4Bit`:

```python
55:  ctx.state = quant_state          # a plain Python attribute
59:  ctx.tensors = (None, B)          # a plain Python attribute -- NOT save_for_backward
...
85:  grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(...))
```

bitsandbytes stashes the packed weight `B` and the `quant_state` (which carries
`absmax`, `offset`, `code`) on `ctx` as **ordinary attributes, bypassing
`save_for_backward` entirely**. Gradient checkpointing discards and recomputes
saved tensors *through the saved-tensor hooks*; a plain `ctx` attribute is
invisible to that mechanism. So the reference is captured in the **forward**,
aliases the pooled buffer, and survives untouched into the backward — by which
time the slot has been refilled with another layer.

### The experiment that confirms it: clone on the forward only

If the forward-time capture is the consumer, de-aliasing on the **forward** alone
must fix it, and de-aliasing on the **backward** alone must not. Five independent
runs, three backward repetitions each, control in every run:

```
run1 control  -> WRONG  ['60/128', '8/128', '8/128']
run1 clone_fwd -> EXACT ['128/128', '128/128', '128/128']
run1 clone_bwd -> WRONG ['8/128', '8/128', '8/128']
run2 control  -> WRONG  ['8/128', '8/128', '8/128']
run2 clone_fwd -> EXACT ['128/128', '128/128', '128/128']
run2 clone_bwd -> WRONG ['8/128', '8/128', '8/128']
run3 control  -> WRONG  ['8/128', ...]   clone_fwd -> EXACT   clone_bwd -> WRONG
run4 control  -> WRONG  ['8/128', ...]   clone_fwd -> EXACT   clone_bwd -> WRONG
run5 control  -> WRONG  ['8/128', ...]   clone_fwd -> EXACT   clone_bwd -> WRONG
```

**5/5 runs, 15/15 repetitions: forward-only de-aliasing is exact; backward-only
is exactly as broken as the control.** This clears the five-repeat bar that the
`buffers=4` near-miss earned, and the direction is not a coin flip — the two arms
separate completely and in the predicted direction.

### Narrowed to the exact tensors, with the control that makes it mean something

If bitsandbytes' `ctx` capture is the whole story, then de-aliasing **only the
tensors bnb captures** — the packed weight and its `::absmax` / `::nested_*`
sidecars — must fix it, while de-aliasing **only the rest** of the layer (the
layernorms and biases, which reach native ops that do use `save_for_backward`)
must not. Without the second arm the first is satisfied by "copying most of the
bytes helps somehow".

Five independent runs, three backward repetitions each, control in every run:

```
run1 control            -> WRONG  ['128/128', '8/128', '8/128']
run1 clone_fwd_quant    -> EXACT  ['128/128', '128/128', '128/128']
run1 clone_fwd_nonquant -> WRONG  ['8/128', '8/128', '8/128']
run2 .. run5            -> identical in all three arms
```

**5/5 runs, 15/15 repetitions, and the two arms separate completely.** The
diagnosis holds at tensor granularity: it is exactly the bitsandbytes-captured
tensors, and nothing else in the layer.

Note `run1 control` opening with `128/128` before collapsing to `8/128` — the
first backward after construction is sometimes right, which is the same
first-pass behaviour STEP 2b recorded. It is why every arm here is read across
three repetitions rather than one.

### Confirmed on the real Qwen2.5-32B, and then costed — the cost is the problem

Same two arms on the real 32B checkpoint (not the synthetic reconstruction),
five independent runs, three backward repetitions each:

```
run1 control         -> WRONG  ['256/256', '8/256', '8/256']
run1 clone_fwd_quant -> EXACT  ['256/256', '256/256', '256/256']
run2 .. run5         -> identical
```

**5/5 runs, 15/15 repetitions on a real model.** The diagnosis is not an artefact
of the synthetic reproducer.

Then the price, measured on the same model with correctness asserted in the same
process so a fast arm cannot quietly be the wrong one (5 repeats, seq 256, 15
timed steps after 4 warm-up):

| arm | gradients | tok/s (median) | vs control | **peak VRAM** |
|---|---|---|---|---|
| control *(shipped)* | **WRONG** 8/256 | 416.04 | 1.00x | **4,220 MiB** |
| `clone_fwd_quant` | correct 256/256 | 405.09 | **1.03x** | **19,720 MiB** |
| `clone` (everything, every call) | correct 256/256 | 389.41 | 1.07x | 19,720 MiB |

**Three percent of throughput, and 4.67x the peak VRAM.** 19,720 MiB is
approximately the entire NF4 store (14.99 GB) — which is not a surprise once the
mechanism is named, it is a *consequence* of it. bitsandbytes holds each layer's
forward-time reference until the backward reaches that layer, so a private copy
per layer must stay alive across the whole forward-to-backward span. Any
de-aliasing repair is therefore **O(model), not O(window)**, by construction.

That is the same wall the first `clone` measurement hit, reached again from the
opposite direction and now with a reason rather than a number. It closes the
whole clone family: correct, nearly free in time, and it deletes the premise of
layer streaming.

### What is left, and it is specific

The memory cost is forced by bnb holding the reference. So the repair has to stop
it holding one:

1. **Upstream**: `MatMul4Bit` uses `save_for_backward` for `B` and the
   `quant_state` tensors instead of plain `ctx` attributes. Gradient
   checkpointing would then discard and recompute them like every other op, and
   the cost returns to O(window). This is the clean fix and it is not in Soup.
2. **In Soup**: do not route streamed NF4 layers through `MatMul4Bit` at all —
   dequantise the layer's weights *inside* the checkpointed region and use a
   native matmul, which does use `save_for_backward`. Cost is a transient
   dequantised layer (O(window)) plus dequantisation compute. Untested.
3. **Refuse**, as STEP 8 concluded for `pin=False`, until 1 or 2 exists.

None of the three was implemented here. What is established is that the two
obvious repairs — hold the buffer, or copy it — are both O(model) and therefore
both self-defeating, and *why*.

### Every fact now accounted for

| fact | explanation |
|---|---|
| NF4 only; bf16 exact at 935 MiB/layer | bf16 goes through `MmBackward0`, a native op using `save_for_backward`, so checkpoint discards and recomputes it. bnb bypasses that mechanism. |
| forward bit-exact, backward wrong | the corruption is a stale *reference* taken in the forward; the forward itself computes on correct data |
| survivors = the last `stream_buffers` layers | exactly the layers whose slot was never refilled |
| correctness returns at `buffers == n_layers` | nothing is ever refilled |
| backward-side fixes do nothing | the recompute's fresh views are not what the node reads |
| `clone` fixes it; `clone_fwd` alone fixes it | the forward-time capture points at a private copy |
| `synchronize()` does not fix it | not a timing race |

Still not explained: **the sharpness of the 163.8 / 171.5 MiB boundary.** Nothing
in this mechanism is size-dependent, so the threshold is presumably about *when*
a slot gets refilled relative to the backward, not about whether the reference is
stale. Open.

### What this means for the repair

The fix is now specific rather than architectural: the pooled tensors handed to
**bitsandbytes** must not be recycled, because bnb holds them outside the
mechanism gradient checkpointing relies on. That is much narrower than "hold
every layer" — it is the NF4 path only, and it is why the bf16 path has been
correct all along at four times the bytes.

Whether the cheapest form is cloning only the quantised tensors on the forward,
teaching the pool not to recycle a slot bnb still references, or an upstream
change to bnb to use `save_for_backward`, is a design decision with costs this
record has not measured. What it has measured: **cloning everything on the
forward is 6% throughput and, unlike cloning on every call, does not have to be
paid on the recompute.**

### Two more attempts on the NF4-only asymmetry, both negative

**Rejected 7 — the number of tensors per layer.** An NF4 layer arrives as many
tensors (packed weights plus `::absmax`, and under double quant also
`::nested_absmax` and `::nested_offset`), where a bf16 layer is one tensor per
weight. Turning double quant off removes two sidecars per weight, so if tensor
count were the trigger it should move the outcome. It does not — it makes it
worse:

```
double_quant=1  241.2 MiB/layer  ->  WRONG  ['16/128', '8/128', '8/128']
double_quant=0  263.0 MiB/layer  ->  WRONG  [' 0/128', '0/128', '0/128']
```

With double quant off not even the last two layers survive. No explanation for
that offered; it is recorded because it is the opposite of what the hypothesis
predicted.

**Confirmed — correctness returns exactly when nothing is ever evicted.** STEP 6
observed that the survivor count tracks `stream_buffers`; taken to its limit on
the 32-layer synthetic model, the relationship is exactly linear and closes at
one buffer per layer:

| `stream_buffers` | exact gradient tensors | = 4 x buffers? |
|---|---|---|
| 2 | 8 / 128 | yes |
| 16 | 64 / 128 | yes |
| **32** *(= `n_layers`)* | **128 / 128 — EXACT** | pool holds everything |

Four tensors per layer are what this LoRA configuration trains (`q_proj` and
`v_proj`, A and B each), so "exact tensors = 4 x buffers" is "the last `buffers`
layers are correct" restated. At `buffers == n_layers` nothing is ever recycled
and the model is bit-exact.

That turns the earlier prediction into a measurement, and it is the mechanism's
tightest confirmation: **the defect is exactly the eviction**. It also closes off
the "just raise `stream_buffers`" route quantitatively — the pool would have to
hold all 32 layers, 7.7 GB, which is the whole model resident and therefore the
feature's negation. Same conclusion as the clone route, reached independently.

*(The 7.41x here and the 6.56x in STEP 8 are different models — this synthetic
32-layer one versus the real Qwen2.5-32B — so they are two measurements of the
same effect, not a discrepancy.)*

### Reproducing the mechanism arms

`mechanism.py`, arms `control,sync,clone` and `control,clone_quantstate,clone_packed`,
on the synthetic 32-layer / 5120 / 27648 NF4 model. Each run re-verifies its own
control.

---

# ── END OF THE MEASUREMENT PHASE ──

Everything above was measured against **unmodified Soup at `2c9a078`** (plus the
two fixes another agent landed on `main` mid-session, `f715218` and `9ae7a9b`,
neither of which touches the streaming path). No `src/` change of mine exists
above this line.

Below this line the brief changes: a `nn.DataParallel` guard cannot be written
without editing `src/`. **Every measurement from here on names the revision it
was taken at.**

---

## STEP 10 — the `nn.DataParallel` guard (FIX PHASE)

*Measured on the box at `9117da1` plus the patch below, which is the only `src/`
change in play. The box's clone predates `f715218` and `9ae7a9b`, so the
GEMM-ceiling row still fails there; that is the old clone, not a regression.*

FINDING 1 accounted for eight of STEP 1's nine failures and cannot be reproduced
on a one-GPU machine, which is why it survived the whole of v0.72.0–v0.72.4.

**Design choice: refuse, not silently drop to one card.** `Trainer._wrap_model`
applies `nn.DataParallel` whenever `args.n_gpu > 1`, and `TrainingArguments`
sets `_n_gpu = torch.cuda.device_count()` for a non-distributed run. Forcing
`n_gpu = 1` would make streaming "work" on an 8-GPU box while silently using one
eighth of it — the same class of silent degradation STEP 8 rejected for the
pinning fallback. The rest of this path refuses rather than warns (the VRAM fit
decision does), so the guard refuses, names `stream_layers`, states the
incompatibility, and gives the exact fix.

It fires **before** the tokenizer load, the weight resolve and the shard write:
finding out minutes into disk I/O is strictly worse.

Distributed launches are exempt: `torchrun` / `accelerate` give each process one
GPU and HF reports `n_gpu == 1`, so `DataParallel` is never applied and refusing
would ban a configuration that works. Detected via `WORLD_SIZE`; a malformed
value counts as non-distributed, because refusing with a clear message beats
proceeding into torch's `meta` error.

**Verified on the real eight cards**, which is the point of doing it here:

```
$ python -m soup_cli.cli train --config stream05b.yaml --yes      # 8 visible
Error: ValueError: training.stream_layers=true, but 8 CUDA devices are visible.
... with one card visible, e.g. CUDA_VISIBLE_DEVICES=0, or set stream_layers=false

$ CUDA_VISIBLE_DEVICES=6 python -m soup_cli.cli train --config stream05b.yaml --yes
Layer streaming ready: 24 layers, 0.18 GB pinned RAM store, 2 x 8 MB VRAM buffers
{'train_runtime': 14.6791, ..., 'train_loss': 0.47571421414613724, ...}
```

The single-card control still trains, and its `train_loss` is **byte-identical**
to the pre-guard run recorded in STEP 2 (`0.47571421414613724`) — the guard
changes nothing on the path that works.

**Tests run on both classes of hardware.** `tests/test_stream_multi_gpu_guard.py`
monkeypatches `torch.cuda.device_count`, so a CPU-only CI runner exercises the
same branch as a datacenter node; the one test that genuinely needs several
cards is skipped by hardware behind `SOUP_TEST_MULTI_GPU`, not left failing.

```
CUDA_VISIBLE_DEVICES=6 pytest tests/test_stream_multi_gpu_guard.py   13 passed, 1 skipped
SOUP_TEST_MULTI_GPU=1  pytest tests/test_stream_multi_gpu_guard.py   14 passed
```

Controls carry the weight here: single GPU, `--device cpu` with 8 visible, CUDA
unavailable, and a distributed launch must all be **allowed**, or the guard is
satisfied by one that always raises and every single-GPU run breaks.

Full streaming suite after the change, one card visible: **435 passed, 1 skipped,
6 failed** — the same six as STEP 1 (five #328 preference-loss failures plus the
GEMM-ceiling bound this clone predates the fix for), and +13 new passes. No
regression.

## STEP 11 — convergence and downstream quality

*Training and evaluation on the box at `9117da1`; the DataParallel guard of STEP
10 is inert here (one card visible per run), so these numbers are unaffected by
it.*

The largest gap in the whole project, and this record said so itself: *"No model
was trained to convergence and no downstream quality was evaluated anywhere in
this session."* That was true of the project, not just the session. Bit-exactness
proves the mechanism does not corrupt the arithmetic. It does not prove a model
trained by streaming is as **good** — and the entire project rests on the
principle that "the loss went down" is not enough.

**Setup.** `Llama-3.1-8B-Instruct`, **bf16 both arms** (the cleanest possible
claim — matched numerics, so any difference is the streaming path itself).
`dair-ai/emotion`, 6-way single-label classification: a closed label set that
`soup ship --task-mode metric` scores with plain exact match, no judge and no
network. 3000 training rows, 3 epochs, batch 8, `max_length` 128, LoRA r=8
alpha=16. Held-out is a fixed 300-row sample of the `test` split, never trained
on, identical for every run. Majority-class baseline 0.3333.

The judge is **Soup's own `soup ship`**, not a hand-rolled script — that was the
point of the exercise.

**The seed trap, and what was done about it.** This record already noted that the
CLI does not pin the adapter-init seed: `stream_setup` seeds its LoRA init at 0
(`tcfg.seed ... else 0`) while the resident path takes HF's global seed, and
`TrainingConfig` has **no `seed` field** to align them. Fixing that means editing
the schema, which was off-limits in the measurement phase. So the second option
was taken: **replicates paired by training subset**. Streamed run *i* and
resident run *i* see byte-identical data (verified: the subsets are disjoint and
their checksums stable across regeneration), and the within-arm spread across
subsets is what the between-arm difference has to beat. All ten adapters have
distinct checksums, so the pairing did what it was meant to.

### Leg 1 — task win

| run | tuned accuracy | base |
|---|---|---|
| resident s0 / s1 / s2 | 0.9033 / 0.9033 / 0.9033 | 0.4200 |
| streamed s0 / s1 / s2 | 0.8867 / 0.8933 / 0.8967 | 0.4200 |

Both arms win leg 1 decisively — 0.42 → ~0.89–0.90 against a 0.3333
majority-class floor, so both learned the task.

| | mean | min | max | spread |
|---|---|---|---|---|
| resident | 0.9033 | 0.9033 | 0.9033 | **0.0000** |
| streamed | 0.8922 | 0.8867 | 0.8967 | 0.0100 |

Paired differences (resident − streamed): **+0.0167, +0.0100, +0.0067**, mean
+0.0111. All three have the same sign, which is worth stating plainly rather than
burying — but 0.0111 is 3.3 items out of 300, and it is barely larger than the
streamed arm's own spread of 0.0100, which is exactly the comparison this design
exists to make. **On n=3 these arms are not distinguishable.** A sign test on
three same-sign pairs gives p = 0.125 one-sided; suggestive, not significant.
Extended to n=5 pairs (running).

Two honest caveats. The gap **cannot be attributed to streaming** while the two
paths still seed their adapters independently — that variable is uncontrolled by
construction, and it is the same one flagged earlier. And the resident arm's
spread of **exactly 0.0000** across three *different* 3000-row subsets with three
*different* adapters (271/300 each time) is anomalous; it is not explained here,
and it makes "the between-arm difference beats the within-arm spread" a weaker
test than it looks, because one arm's spread is degenerate.

### Leg 2 — general-capability regression, and it is not about streaming

Five of six runs came back **DON'T SHIP**, so the interesting question is which
arm. It is neither:

| run | decision | failed | task | arith | common_sense | format_json | instr | mmlu | safety | tool_call |
|---|---|---|---|---|---|---|---|---|---|---|
| resident s0 | **SHIP** | — | 0.9033 | −0.028 | +0.083 | +0.150 | 0.000 | +0.115 | −0.050 | 0.000 |
| resident s1 | DON'T SHIP | regression | 0.9033 | 0.000 | **−0.083** | +0.100 | 0.000 | **−0.077** | +0.050 | 0.000 |
| resident s2 | DON'T SHIP | regression | 0.9033 | 0.000 | **−0.417** | +0.100 | −0.042 | **−0.077** | −0.025 | 0.000 |
| streamed s0 | DON'T SHIP | regression | 0.8867 | −0.028 | **−0.167** | +0.175 | 0.000 | +0.038 | −0.050 | 0.000 |
| streamed s1 | DON'T SHIP | regression | 0.8933 | 0.000 | **−0.208** | +0.075 | 0.000 | 0.000 | +0.100 | 0.000 |
| streamed s2 | DON'T SHIP | regression | 0.8967 | 0.000 | **−0.333** | +0.125 | 0.000 | **−0.077** | +0.050 | 0.000 |

Bold = the suite `ship` flagged as regressed against its 0.05 threshold.

**The single worst regression in the matrix belongs to a resident run**
(`resident_s2`, −0.417 on common sense). The failure mode is symmetric across
arms and it is the expected one: turning a chat model into a one-word classifier
on 3000 narrow examples costs general ability. `soup ship` caught it, in both
arms, which is the tool working.

Read carefully, though: `mini_common_sense` has 24 items, so one item is 4.2% and
`−0.417` is ten items. These bundled suites are coarse by design (they are meant
to be runnable offline, not to be precise), and a 0.05 threshold against a
24-item suite trips on two items. The verdicts are directionally right and
numerically blunt.

One consistent side effect worth recording: `mini_format_json` **improved in all
six runs** (+0.075 to +0.175) from a base of 0.0 — plausibly because the task
teaches the model to emit a short bare answer instead of prose, which is what
that scorer rewards. Not investigated.

### Extended to n=5, and the n=3 reading did not survive

Two more paired subsets (the training split holds 16 000 rows, so five disjoint
3000-row subsets is the ceiling). The regenerated `train_s0.jsonl` and
`task_eval.jsonl` checksums are unchanged, so the first three pairs stay valid.

| pair | resident | streamed | difference |
|---|---|---|---|
| s0 | 0.9033 | 0.8867 | +0.0167 |
| s1 | 0.9033 | 0.8933 | +0.0100 |
| s2 | 0.9033 | 0.8967 | +0.0067 |
| s3 | 0.8900 | 0.8900 | **0.0000** |
| s4 | 0.8967 | 0.9000 | **−0.0033** |

| | mean | spread |
|---|---|---|
| resident | 0.8993 | 0.0133 |
| streamed | 0.8933 | 0.0133 |

**Both readings from n=3 dissolved.**

The "all three differences have the same sign" observation was an artifact of
which three subsets ran first: at n=5 it is three positive, one exact tie and one
negative. Mean paired difference **+0.0060** — 1.8 items out of 300 — against a
within-arm spread of **0.0133 in both arms**, so the between-arm difference is
**less than half** the noise the design measures it against.

The resident arm's suspicious 0.0000 spread also dissolved: subsets 3 and 4 give
0.8900 and 0.8967. Three identical values in a row was coincidence, and flagging
it as an anomaly was right but it needed more data, not an explanation.

That the two arms' spreads came out **identical** (0.0133 each) is the cleanest
form this result could take: the same variability, means 0.6pp apart.

### What this establishes

- A model trained through layer streaming **converges and reaches the same task
  quality as a resident run**. Paired over five subsets: +0.0060 mean difference
  against 0.0133 within-arm spread on both sides, 3 positive / 1 tie / 1
  negative. **No difference is detectable at this resolution.**
- It **does not** show streaming is quality-neutral to arbitrary precision — the
  experiment resolves differences of roughly 1pp on a 300-item held-out set, and
  the adapter-init seed remains uncontrolled because there is no config field
  for it.
- The general-capability regression both arms show is a property of the
  fine-tuning task, not of streaming, and the worst instance is resident.

---

## STEP 12 — ZeRO-3 on eight cards against streaming on one

*Box at `9117da1` + the DataParallel guard + the `device_map` fix below.*

The positioning question, in the form someone with a DGX asks it: *"I have eight
cards — why would I stream on one instead of sharding across eight?"*

### FINDING 4 — Soup's advertised multi-GPU path does not run at all

The first attempt died on every rank before training started:

```
$ torchrun --nproc_per_node 8 -m soup_cli.cli train --config b_ds.yaml --deepspeed ...
Error: ValueError: You can't train a model that has been loaded with
`device_map='auto'` in any distributed mode.
```

This is not about DeepSpeed. `soup train --gpus 8` prints

```
To train on 8 GPUs, re-run under accelerate:
    accelerate launch --num_processes 8 soup train -c /root/run/b_ds.yaml
```

and **that exact command fails the same way**, with and without `--deepspeed`,
under `torchrun` and under `accelerate launch` alike. The cause is one line
repeated across six trainers:

```python
dev_map = "cpu" if self.device == "cpu" else "auto"
```

`device_map="auto"` shards one model across every visible GPU; under a
distributed launch every rank would try to do that, and transformers refuses
outright. There was no distributed check anywhere in the trainer package.

So the multi-GPU story the tool advertises has never worked on the SFT path.
Like FINDING 1, it is invisible on a one-GPU dev box.

**Fixed** — `utils/gpu.resolve_device_map(device)`, one shared helper replacing
the line in `sft` (x3), `dpo`, `grpo`, `kto`, `pretrain` and `ppo` (x3): `"cpu"`
on CPU, `{"": LOCAL_RANK}` when `WORLD_SIZE > 1`, `"auto"` otherwise, with a
malformed environment falling back to `"auto"` because a process with no usable
distributed environment *is* single-process. Verified on the real eight cards:
the previously-failing command now trains to completion.

### The comparison

Everything identical: `Llama-3.1-8B-Instruct`, bf16, LoRA r=8, the STEP 11
dataset (3000 rows, 3 epochs, `max_length` 128, `batch_size` 8 per device),
**623,973 tokens** by the trainer's own count in every row. tok/s is
`num_tokens / train_runtime`, i.e. division.

| method | GPUs | train_runtime | tok/s | peak VRAM |
|---|---|---|---|---|
| **resident bf16** | **1** | **235.84 s** | **2645.4** | 30.0 GB |
| ZeRO-3, no offload | 8 | 340.65 s | 1831.7 | 34.0 GB **per card** |
| ZeRO-3 + CPU param offload | 8 | 356.10 s | 1752.2 | 38.0 GB per card |
| layer streaming bf16 | 1 | 579.30 s | 1077.1 | **2.9 GB** |

SM clock 1980 MHz median-while-busy in every row.

**The headline is not the one the question expects: eight cards of ZeRO-3 are
slower than one card training resident** — 340.65 s against 235.84 s. For a model
that fits on a single card, sharding it across eight buys nothing and costs an
all-gather per layer per step over PCIe. The 8-GPU runs use an effective batch of
64 against the single-GPU 8, which is how anyone would actually use them, and
they still lose.

Against streaming, eight ZeRO-3 cards are **1.70x** faster than streaming on one
(1831.7 / 1077.1) — not the 10x the question anticipated, and they spend 34 GB on
each of eight cards to do it, against 2.9 GB on one.

### What that means for positioning, stated plainly

The right reading is **not** "streaming beats ZeRO-3". It is that this whole
comparison is being run on hardware where the premise of streaming does not
apply. An 8B model fits comfortably in one H100, so the correct choice on this
box is the top row — resident on one card — and both sharding and streaming are
worse than doing nothing clever at all.

Layer streaming is for the case where **the one card you have cannot hold the
model**. That case cannot be constructed on an 80 GB card with an 8B model, which
is exactly why the feature's real evidence is on a 4 GB laptop. The honest claim
is therefore the narrow one: *not "we are faster than DeepSpeed", but "we are for
a different situation — one card, and it is too small"*.

Two caveats that cut in ZeRO-3's favour and are stated rather than omitted: this
box is **PCIe, PIX between all pairs, no NVLink**, and ZeRO-3's all-gather traffic
is exactly what NVLink exists for — on an NVLink node these ZeRO-3 rows would
improve and the streaming row would not. And 623,973 tokens over ~140 optimizer
steps is a short run; a longer one amortises DeepSpeed's startup better.

### One discarded measurement, kept

The first 8-GPU attempt used the original 64-row benchmark set (`b_ds.yaml`,
4 epochs, 8 optimizer steps) and produced ZeRO-3 **with** CPU offload at 86.29 s
against ZeRO-3 **without** offload at 102.56 s — offloading to CPU apparently
*faster* than keeping parameters on the GPUs. That is not a real effect; at eight
optimizer steps the run is entirely startup and communication setup, and the
measurement says nothing about throughput. Re-run on the 3000-row dataset the
ordering reverses and becomes sensible. Left here because it is exactly the kind
of number that would have looked publishable.

## STEP 13 — is variant 2 numerically clean? (a probe, and a vacuous first answer)

Both repairs measured in STEP 9 are dead for one shared reason: bitsandbytes holds
its reference from forward to backward, so *any* de-aliasing keeps a copy of every
layer alive across that span and costs **O(model)**, not O(window). STEP 9's
measurement on real 32B put the number on it: peak VRAM 4 220 -> 19 720 MiB.

Variant 2 is the remaining candidate implementable inside Soup: do not send the
streamed NF4 weight through `MatMul4Bit` at all. Dequantise inside the checkpointed
region and use a native matmul. That is O(window) by construction — the dequantised
weight is a transient inside the recomputed block, not a reference held across it.

Its risk is numerical. `matmul_4bit` is a fused kernel; dequantise-then-matmul is two
operations with a different accumulation order. There is no reason to expect the two
to be bit-identical, and if they are not, then "streamed NF4 == resident NF4" stops
holding *by construction rather than because of a bug* — which is a question about
what the project's reference standard means, not a question about the repair.

Forward and gradient are asked separately throughout and never averaged, because
bitsandbytes is asymmetric and the asymmetry is the whole point:

```
MatMul4Bit.forward   -> torch.ops.bitsandbytes.gemm_4bit          (fused)
MatMul4Bit.backward  -> F.dequantize_4bit(B, ctx.state) @ ...     (NOT fused)
```

The gradient path upstream is *already* dequantise-then-matmul. So the two arms can
agree on the gradient while disagreeing on the forward, and the gradient is the half
that matters here — it is what the defect corrupts.

### The first answer was vacuous, and the path control did not catch it

First run: real Qwen2.5-32B projection shapes, real `quant_state` (blocksize 64, nf4,
double_quant — what `layer_shard._quantize_nf4` writes), 5 seeds x 5 shapes at
M = 2048 tokens. Result: forward **and** gradient bit-exact on 25/25, worst
`max_abs` exactly 0.0, with determinism and accumulation-order controls holding.

All true and all beside the point. `bitsandbytes::gemm_4bit` dispatches on M:

```python
_gemm_4bit_custom_max_m = 1536   # CUDA
if M > _gemm_4bit_custom_max_m:
    use_custom = False           # -> _dequant_linear_fallback
```

At M = 2048 bitsandbytes itself takes `_dequant_linear_fallback`. **Both arms were
running the same code.** The run compared variant 2 against variant 2.

The control that should have caught this did not. It counted calls to
`bitsandbytes.functional.dequantize_4bit` and saw zero in the forward, which was read
as "the fused kernel ran". The fallback lives in `backends/cuda/ops.py` and reaches
the dequantiser through a registered torch op, never touching the `functional`
wrapper that was patched. A path control has to observe the path, not a proxy for it.

An earlier control in the same run failed loudly and was fixed before this one was
found: the negative control perturbed a weight element by `max|W| * 1e-3`, about
4e-3 relative on an element of size ~2e-2 — which is bf16's own resolution (2^-8).
It rounded away, so the "control" compared a tensor against itself and reported that
`torch.equal` could not detect a difference. Replaced with a sign flip plus a whole
unit, and an assertion that the perturbation survived rounding.

### The second answer, with the dispatch read out rather than assumed

Rewritten to (a) record the dispatcher's actual per-M decision, (b) **force** the
fused kernel on at training-shaped M by patching the two module globals the kernel
reads at call time, and (c) count `_dequant_linear_fallback` directly, so a forcing
that silently failed is distinguishable from a kernel that genuinely agreed.

Forcing verified: **24/24 forced rows ran with 0 fallback calls.** The forced rows
are real agreements or real disagreements, not failed patches.

| | rows | forward bit-exact | gradient bit-exact |
|---|---|---|---|
| default dispatch, M in 1..4096, 6 shapes, 5 seeds | 300 | no (only where fused ran) | **yes, 300/300** |
| fused kernel forced on, M = 512 and 2048 | 60 | no, 3 of 6 shapes | **yes, 60/60** |
| fallback-counter cross-check, M in 8..2048 | 48 | as above | **yes, 48/48** |
| shipped CI fixture shape | 15 | no at M = 8, 16 | **yes, 15/15** |

**Gradient: bit-exact in 423 of 423 rows, worst `max_abs` exactly 0.0**, across every
shape, every M, forced and unforced. This is not luck; it is construction. Variant 2's
backward *is* bitsandbytes' backward — the same `dequantize_4bit` followed by the same
matmul. Nothing about the gradient changes.

**Forward: differs only where the fused kernel genuinely runs, and then at bf16 noise.**
Worst `max_abs` 6.25e-2, worst relative-to-scale **3.95e-3**, against bf16's own
resolution of 2^-8 = 3.9e-3. One unit in the last place. Structurally the same number,
not a different one.

And the window where the fused kernel runs does not overlap real training shapes:

| shape | fused kernel runs at | falls back at |
|---|---|---|
| Llama-3.1-8B `q_proj` [4096x4096] | never, in M = 8..512 | every M tested |
| Qwen2.5-32B `q_proj` / `down_proj` | never, in M = 8..2048 | every M tested |
| Qwen2.5-32B `gate_proj` [27648x5120] | M <= 32 | M >= 512 |
| CI fixture [64x64] | M <= 16 | M >= 64 |
| CI fixture [256x64] | every M tested (8..512) | never |

At real training shapes bitsandbytes is **already doing variant 2 internally**. So for
the configuration this feature ships for — 8B / 32B NF4, batch x seq >= 128 — variant 2
is not a numerics change at all. It makes explicit what the library already does, and
moves it inside the checkpointed region where the aliasing cannot happen.

The heuristic was also queried for arches this box does not have, by substituting
`_gpu_dispatch_props`: sm86 (RTX 3050 Laptop, the shipped 4 GB configuration), sm89
and sm90 all agree — over the 60 (M, shape) combinations swept, the largest M at which
any of them selects the fused kernel is **32**. The sm86 answer is a query of the
shipped heuristic, not a measurement on that card.

### The one place it does bite: the CI fixture

`tests/test_v07202.py` builds `hidden_size=64, intermediate_size=64, vocab_size=64`
with sequences of 64-128. That shape sits **inside** the fused window — 7 of 10 fixture
rows use the fused kernel, and at M = 8 and 16 the forward differs by up to 3.9e-3.

So the tests that assert streamed-NF4 == resident-NF4 bit-exactly are running in
precisely the regime where variant 2 changes the streamed arm, while the real models
they stand in for are not. The gradient stays exact there too (15/15), so what would
move is the forward assertion, on fixtures, at one bf16 ulp.

That is a decision about what the reference standard means, and it is deliberately not
taken here. Recorded, and left open.

Scripts: `numerics.py` (the vacuous first run, kept), `numerics2.py` (dispatch sweep
plus forced fused kernel), `forcecheck.py` (fallback counter), `fixtureshape.py`.
Results: `/root/results/numerics_bf16.json`, `numerics_v3.json`, `forcecheck.json`,
`fixtureshape.json`. torch 2.13.0+cu130, bitsandbytes 0.50.0, H100 sm90, 132 SMs.

## STEP 14 — variant 2 implemented, and the gate it had to pass (FIX PHASE)

STEP 13 established that dequantise-then-matmul is bit-exact to `MatMul4Bit` on the
gradient in 423/423 rows, by construction. So the repair is: never send a streamed
NF4 weight through that autograd Function. `install_dequant_forward` swaps the
forward of every `bnb.nn.Linear4bit` inside a streamed layer for
`dequantize_4bit` + `F.linear`, with the dtype handling mirroring
`Linear4bit.forward` line for line so nothing else moves.

`F.linear` then saves the dense weight through the ordinary mechanism, which
checkpointing DOES discard and recompute, and the transient lives only inside the
recomputed block — O(window), which is what both rejected repairs could not be.

### The gate, on real Qwen2.5-32B

Both arms in ONE process against ONE deterministic resident NF4 reference. The
`control` arm neuters `install_dequant_forward` before the model is built, so it
must reproduce the defect; without that, a passing `variant2` arm is equally
consistent with "this configuration never triggered it".

64 layers, NF4, `stream_buffers=2`, `pin=True`, seq 128, 5 repeats.

| | control (repair off) | variant 2 | resident reference |
|---|---|---|---|
| gradients exact | 8–12 / 256 | **256 / 256**, all 5 reps | — |
| wrong layers (of 64) | 61–62 | **0** | — |
| worst abs | 7.743083e-01 | **0.0** | — |
| loss | 13.058376312 | 13.058376312 | 13.058376312 |
| peak VRAM | 22 413.3 MiB | 23 064.2 MiB | 30 504.0 MiB |
| median tok/s | 208.7 | 198.6 | — |
| Linear4bit modules rewired | 0 | 448 | — |

**Cost: +650.9 MiB (1.029x) and −4.8% throughput (0.951x).** Against the criteria
set before the work: gradients exact at `pin=True, buffers=2` — yes, 256/256 five
times over; peak VRAM within a couple of layers rather than O(model) — yes, +650.9
MiB against a per-layer packed size of 239.8 MiB, i.e. **2.71 layer-equivalents**;
throughput no worse than 1.5x — yes, 1.05x.

For scale, the de-aliasing repair rejected in STEP 9 cost 4 220 -> 19 720 MiB. That
figure comes from STEP 9's own harness and configuration, not from this gate's
control, so the two peak columns are not directly subtractable — what transfers is
the shape of the cost: O(model) there, a bounded transient here.

**Both arms produced a loss identical to the resident reference, to every digit.**
That is not an aside; it is the whole reason this defect was silent for three
releases. A run whose gradients are wrong on 62 of 64 layers reports a healthy,
bit-exact loss curve. Only a gradient comparison against a resident reference can
see it, which is why that comparison is the gate.

### The first run of this gate was vacuous, and the control is what caught it

Run 1 reported `grads exact 0/0` on every repetition and a verdict of
`variant2_gradients_all_exact: true`, because `n_exact == n_compared` is trivially
satisfied by `0 == 0`. The cause: a streamed model's `named_parameters()` still
carries the wrapper's `.inner.` segment — v0.72.1's fix was serialisation-only — so
intersecting streamed names with resident names gives the EMPTY SET. The gate was
comparing nothing and calling it a pass.

What exposed it was not the green verdict but the line under it: the control had
not broken. A control that cannot fail is the tell for a measurement that cannot
measure. Fixed by keying gradients canonically (`.replace(".inner.", ".")`, as
`repeat_backward.py` already did) and by making an empty intersection a hard
`SystemExit` rather than a pass. That guard is now in the harness permanently.

This is the third time in this session a control caught a false positive: the
vacuous M=2048 numerics run, the negative control that rounded away, and this.

### Caveats on these numbers

- `sm_clock` was sampled at the END of each arm, not while busy, and both arms read
  345 MHz — an idle reading. It is recorded for completeness but is NOT a
  while-busy clock and no fraction-of-ceiling is computed from it.
- Peak VRAM is `torch.cuda.max_memory_allocated`, per arm, after
  `reset_peak_memory_stats`.
- The control's repetition 1 differs from repetitions 2–5 (12/256 exact,
  worst_abs 7.099325e-03, against 8/256 and 7.743083e-01). That matches STEP 9's
  finding that the corruption depends on state carried between backward passes, and
  it is why a single repetition is not a measurement here.
- Measured at seq 128 on one model. The repair is shape-independent by
  construction, but only 32B NF4 was gated end to end.

Script: `variant2_gate.py`. Result: `/root/results/variant2_gate_32b.json`.
Shipped as `install_dequant_forward` in `src/soup_cli/utils/layer_stream_runtime.py`,
with a CI test asserting the streamed path makes zero `MatMul4Bit` calls and a
resident control asserting the counter can see such calls at all.

## STEP 15 — #328 diagnosed and fixed: nobody was setting `gradient_checkpointing`

STEP 1 recorded five failures in `tests/test_v07204.py` on this box's CUDA stack —
all four preference losses dead with `RuntimeError: Tensor on device cuda:0 is not
on the expected device meta!`, SFT passing — and left it as "known #328, torch 2.13
only". It has a cause, and the cause is in Soup.

### Where the meta tensor enters

The failing op is `LlamaRMSNorm.forward`'s `self.weight * hidden_states`.
Instrumenting that method over a DPO step: RMSNorm is called 13 times and **exactly
one** call has `weight.device == meta` against `hidden_states.device == cuda:0` — the
last one, under `grad_enabled=True`, i.e. a checkpoint recompute. The SFT control
makes 9 RMSNorm calls with **zero** meta weights.

Three independent lines say the checkpoint doing that recompute is **not Soup's**:

1. the frame is `CheckpointFunction.backward`, the *reentrant* path, while
   `StreamedDecoderLayer.forward` calls `checkpoint(..., use_reentrant=False)`;
2. `ctx.run_function` lands on `LlamaDecoderLayer.forward` through
   `nn.Module.__call__` with **no `_body` frame**, so what is being recomputed is
   `self.inner` itself, not the `functional_call` closure that substitutes the
   streamed weights;
3. at step-begin the inner raw `LlamaDecoderLayer` — a `GradientCheckpointingLayer`
   in transformers 4.57 — already has `gradient_checkpointing=True` and
   `_gradient_checkpointing_func` set, with `input_layernorm.weight.is_meta` True.

So HF's `gradient_checkpointing_enable()` reaches *inside* the wrapper and makes the
inner layer checkpoint its own forward. That inner checkpoint is created while
`functional_call`'s reparametrisation context is open (weights real) and recomputed
during backward long after it has exited and restored the `meta` placeholders.

### Why SFT passed — established by control, not left as a coincidence

`args.gradient_checkpointing` is `False` for SFT and `True` for all four preference
losses, although `training.gradient_checkpointing` is `False` in the config for both.

`should_enable_hf_gradient_checkpointing` exists precisely to stop HF double-
checkpointing a streamed model. **Only `sft.py` ever called it.** `dpo.py`,
`orpo.py`, `simpo.py` and `kto.py` did not mention the flag at all, so its value was
decided entirely by TRL's default.

| arm | `args.gradient_checkpointing` | result |
|---|---|---|
| sft, default | false | **OK** |
| dpo, default | true | FAIL `expected device meta` |
| **dpo, forced false** | false | **OK — the failure disappears** |
| sft, forced true | true | FAIL, but a *different* error |

The load-bearing row is dpo-forced-false: with the wrapper's own `use_reentrant=False`
checkpoint still active, removing HF's checkpointing removes the failure entirely.
Reported honestly, the sft-forced-true row is **not** a clean symmetric control —
flipping the flag after `setup()` skips the input-require-grads wiring that path
would otherwise do, so it fails earlier for an unrelated reason and neither confirms
nor refutes sufficiency.

### One omission, two opposite symptoms

TRL's default is not stable across versions. Measured directly:

| trl | `DPOConfig(...).gradient_checkpointing` | symptom |
|---|---|---|
| 0.19.1 (dev box) | `False` | a user's explicit `gradient_checkpointing: true` was **silently dropped** for all four preference tasks |
| 0.26.2 (this box) | `True` | HF's checkpointing arrived **uninvited** and killed every streamed preference run on torch 2.13 |

That is why it survived: on the machine where the tests are usually run, the bug's
sign is inverted and produces no crash at all.

### The fix, and what the failing test had to be

All four wrappers now pass the flag explicitly through the shared guard, so it
tracks the config and never TRL's default.

The TDD step is worth recording because the obvious test does not fail. On trl
0.19.1 the *streamed* assertion (`args.gradient_checkpointing is False`) **already
passed**, since that version's default is False. The test that was red locally is
the CONTROL: a NON-streaming run that asks for checkpointing must GET it. Written
parametrized over both `True` and `False`, and deliberately NOT asserting TRL's own
default — an assertion on the default would be red on one supported stack and green
on the other while the real bug went unnoticed on both.

Verified on this box, torch 2.13.0+cu130 / trl 0.26.2: `tests/test_v07204.py` goes
from **5 failed, 62 passed** to **67 passed**.

Reproducer and instrumentation: `/root/issue328_min.py` (six arms in one process —
SFT control, the four preference losses, SFT control again; the control passes both
before and after, so this is not ordering or poisoned CUDA state),
`/root/issue328_probe.py`, `/root/issue328_control.py`. Results under
`/root/results/issue328_*.json`.

## STEP 16 — #327: the VRAM pre-flight over-predicts, and not for the reason assumed

72 runs, 3 repeats per cell, Qwen2.5-0.5B-Instruct (vocab 151936 — a real
vocabulary, which is what the two earlier attempts lacked). Spread is **exactly
0.00 MiB in every cell**: `max_memory_allocated` is bit-reproducible for a fixed
shape, so the repeats confirm determinism rather than estimate noise.

Fixture validity was recorded per run, because that is what invalidated the earlier
attempts — DPO measured an identical 63.43 MiB at seq 128, 256 and 512, an
independent variable that never moved. Here `observed_rows x observed_seq` equals
`2·batch x max_length` for DPO and `batch x max_length` for SFT in all 72 runs.

**The bug in one row:** DPO batch 1 / seq 768 is refused (predicted 3541.84 MiB
against a 3166.20 MiB budget at the 3.32 GB free figure from the v0.72.4 notes) and
measures **3092.46 MiB — it fits, with 73.74 MiB to spare.**

**But it is not a preference-loss problem.** The SFT control over-predicts by the
same amount at the same *effective* row count: DPO b1 against SFT b2, and DPO b2
against SFT b4, agree to **+0.09% … +0.51%** (a flat ~6.5 MiB) across all 8
comparable shapes. So `_STREAM_ROWS_PER_EXAMPLE = 2` is **correct** — a streamed DPO
step costs what a streamed SFT step at twice the rows costs.

What is wrong is one coefficient on the shared logits term. The ratio is not
constant (DPO 1.107 -> 1.167, SFT 0.987 -> 1.163); it rises and asymptotes, the
signature of a slope error plus a small offset. A least-squares fit of
`measured = a + b·(vocab x seq x rows)` gives **b = 12.311 bytes/element**,
a = 353.0 MiB, worst residual 0.15% over 12 DPO cells, against the shipped
`LOGITS_BYTES_PER_ELEMENT = 14`.

**The honest statement is that the constant is stack-dependent, not that 14 is
wrong.** 14 was measured on RTX 3050 / Windows / an older torch; 12.3 is this
stack. Both are measurements, and v0.72.3's own grid — 10 real runs, worst error
0.85%, never under-predicting — still passes unchanged. Changing 14 on the strength
of one stack would trade a documented over-prediction for an undocumented
under-prediction, which is the worse failure: on Windows/WDDM it does not raise, it
silently spills.

Two robustness checks: 30 cells re-run against the tree carrying the #331 fix gave
**byte-identical** peaks (the repair does not move VRAM), and the SFT 1x256 cell
reproduces the -1.3% under-prediction that v0.72.4 already documents as a known
-1.7% SFT miss.

Method note worth keeping: peak must be reset AFTER `setup()`. A peak spanning
`setup()` charges the pre-flight's own 4096³ GEMM probe to the training step and
reports an under-prediction that does not exist — recorded in v0.72.4 as one of
three invalid measurement attempts, and re-encountered here.

Raw: `/root/results/issue327b/*.json` (72 records), `/root/results/issue327_summary.json`.

## STEP 17 — #78: the first measurement of FlashAttention and Liger, and what it found

Neither had ever been measured, because neither installs on the maintainer's
Windows box. Real `soup train`, Llama-3.1-8B + LoRA r=16, bf16, batch 4, seq 1024,
80 steps, 5 repeats **interleaved** A/B/C/D so clock drift cannot favour whichever
ran first.

| arm | median tok/s | spread | peak VRAM | SM clock (busy) | ratio |
|---|---|---|---|---|---|
| baseline (SDPA) | 9621.1 | 0.83% | 70.99 GiB | 1965 MHz | 1.000x |
| FlashAttention 2 | 9764.8 | 0.92% | 70.99 GiB | 1965 MHz | **1.015x** |
| Liger (HF flag) | 10115.3 | 0.22% | 61.81 GiB | 1905 MHz | **1.051x**, VRAM -12.9% |
| FA2 + Liger | 10256.0 | 0.53% | 61.81 GiB | 1920 MHz | **1.066x**, VRAM -12.9% |

Activation was verified rather than assumed — a counter on PYTHONPATH in dedicated
short runs (timing runs had counters off): baseline 256 SDPA calls / 0 FA2, FA2 256
FA2 calls / **0 SDPA**, Liger with the module classes actually swapped to
`LigerRMSNorm` / `LigerSwiGLUMLP`. Baseline's real kernel, from `torch.profiler`,
is `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16`.

**Against what Soup documented:** FlashAttention claimed "2-4x speedup and
significant memory savings" and measures **1.015x with zero memory saving**; Liger
claimed "20-60% memory and 20-40% throughput" and measures **12.9% and 5.1%**. Both
claims are now corrected in the source docstrings and docs, with the measurement
conditions attached.

FA2 has little to beat here: the baseline is already a Hopper flash kernel, so this
is flash against flash, which is also why the VRAM columns match to four figures.
That is a property of the comparison, not a criticism of FA2, and the gap would
widen with sequence length.

**Four defects the benchmark found before it found numbers:**

1. **`training.use_liger: true` crashed at step 0.** Soup patched the model but
   never set `TrainingArguments.use_liger_kernel`, the flag TRL reads to know the
   fused path returns `logits=None`; its entropy guard could not fire and
   `entropy_from_logits(None)` raised. Reproduced on trl 0.26.2 **and** 0.19.1, so
   the feature had never worked anywhere in the supported pin. The Liger rows above
   use the HF-native flag, i.e. what a fixed Soup does. **Fixed.**
2. **`data.max_length` above 1024 was silently ignored on every SFT run** —
   `max_length=512` gave 512 tokens/sample, `max_length=4096` gave **1024**. Soup
   passes a `TrainingArguments`, and `SFTTrainer` converts it with
   `SFTConfig(**args.to_dict())`, where `max_length` is an SFT-only field that does
   not survive the round trip. **Fixed.** This also bounds the FA result: 1024 was
   the longest sequence Soup could produce, so the benchmark ran at the operating
   point users actually got.
3. **The FlashAttention 3 branch is unreachable.** Soup selects FA3 when
   `import flash_attn` reports major >= 3, but Dao-AILab ships FA3 as package
   `flash_attn_3` / module `flash_attn_interface`, and transformers gates on
   `_is_package_available("flash_attn_3")`. Verified by spoofing `flash_attn` to
   3.0.0: Soup requests `flash_attention_3` while
   `transformers.is_flash_attn_3_available()` is False. Not fixed here — filed.
4. FA3 could not be measured on this box for an independent reason (no `nvcc` to
   build `hopper/`, PyPI `flash-attn-3` is a 0.0.0 placeholder, and the `kernels`
   route needs `huggingface_hub>=1.10` against transformers 4.57.6's `<1.0` pin —
   tried, broke transformers, reverted). So "Hopper now or never" did not rescue
   it, and the honest status is unmeasured.

## STEP 18 — #76: vLLM works, SGLang was broken for every request

Both backends are documented and neither had ever been run, because neither
supports Windows.

**vLLM works.** `soup serve --backend vllm` came up in 38–118 s and returned 200 on
`/v1/chat/completions`, on SSE streaming, and on `/v1/messages`. `--prefix-cache`
verified through the engine's own config log. A LoRA adapter passed as `--model`
loaded with the base auto-resolved.

**SGLang was 100% broken.** `utils/sglang.py` did `response["text"]` on the return
of `Runtime.generate`, which in sglang 0.5.16 ends `return json.dumps(...)` — a
string. Every request raised `TypeError: string indices must be integers`, in both
the streaming and non-streaming path. **Fixed**, accepting both shapes.

**Install hazard confirmed, and it is the one the project was already burned by:**
`pip install vllm` into the training venv resolves torch **2.13.0 -> 2.11.0** and
transformers **4.57.6 -> 5.14.1**, past Soup's own `<5.0.0` cap. Separate venvs were
used; the training venv was verified unperturbed afterwards.

Three more defects, filed rather than fixed:

- **vLLM ignores the model's chat template.** `utils/vllm.py::_build_prompt`
  hand-rolls `"User: …\nAssistant:"` while the transformers backend uses
  `apply_chat_template`. A/B inside one engine, same model and sampling params,
  only the prompt differing: on Llama-3.1-8B + LoRA the hand-rolled prompt produces
  a run-on loop that burns all 64 tokens. This degrades every vLLM user's output.
- **`finish_reason` is hardcoded `"stop"`** even when length-truncated — observed
  `"stop"` with `completion_tokens == max_tokens == 64`.
- **`--dashboard` silently no-ops** on vLLM and SGLang (`/metrics` -> 404) although
  `docs/commands.md` advertises it; SGLang additionally lacks `/v1/messages`, which
  `docs/serving-and-export.md` claims for both.

Also recorded because the record is the working one: a suspicion that Soup leaked
`VLLM::EngineCore` processes holding 51 GB was **disproved** by a dedicated SIGINT
test — the orphans were the harness `SIGKILL`ing after 5 s, and Soup's shutdown is
clean in ~6 s.

## STEP 19 — convergence, second arm: NF4

STEP 11 compared streamed against resident in **bf16**. Users and the preprint are
about **NF4**, so the quality claim rested on the wrong quantisation. 8B sits below
the ~165 MiB/layer defect threshold, so this arm is clean by construction — and the
run's own panel confirms it: 2 x 113 MB buffers against a 3.60 GB pinned store over
32 layers.

Same protocol: 5 paired training subsets, one held-out set of 300 rows, judged by
Soup's own `soup ship --task-mode metric`.

| pair | resident NF4 | streamed NF4 | difference |
|---|---|---|---|
| s0 | 0.8766666666666667 | 0.8733333333333333 | +0.0033333333333334103 |
| s1 | 0.8833333333333333 | 0.8766666666666667 | +0.006666666666666599 |
| s2 | 0.89 | 0.8633333333333333 | +0.026666666666666727 |
| s3 | 0.8566666666666667 | 0.8633333333333333 | −0.006666666666666599 |
| s4 | 0.88 | 0.8833333333333333 | −0.0033333333333332993 |

**Mean paired difference +0.005333333333333368** (sum of five, divided by five).
Signs split 3 positive / 2 negative.

The control is the within-arm spread, which is what that difference has to beat:

| arm | mean | spread (max − min) |
|---|---|---|
| resident NF4 | 0.8773333333333333 | **0.033333333333333326** |
| streamed NF4 | 0.8719999999999999 | **0.020000000000000018** |

The between-arm difference is 0.16 of the resident spread and 0.27 of the streamed
spread (both divisions), i.e. **1.6 items out of 300** (multiplication). Both arms
clear the 0.42 task base and the 0.3333 majority-class floor.

**Verdict: at NF4 the two are indistinguishable**, reproducing the bf16 result
(+0.0060 against a 0.0133 spread) on the quantisation that actually ships.

Leg 2 points the same way and is worth stating because it points *against* the
convenient direction: 3 of 5 **resident** runs came back DON'T SHIP against 1 of 5
streamed, and the single worst regression in the matrix is a resident run
(−0.208333 on `mini_common_sense`). Every flagged regression is on a 24- or 40-item
suite where one item is 4.2% / 2.5%. No streaming-specific failure mode.

Judge control: all 10 runs report task base 0.42 and an identical benchmark base
vector, and 0.42 matches the bf16 arm exactly — the judge is deterministic and
unchanged across all four arms in this record.

**The caveat that matters most, stated rather than buried:** the bf16 arm ran at
`9117da1` and this arm at `a14ea3b`, and `985d6fc` (#331) landed between them,
rewriting the streamed NF4 forward. So this arm measures the **post-#331** path.
Both NF4 arms ran at the same revision, so the pairing is internally valid, but
these numbers are **not revision-comparable to the bf16 arm's**.

Two more limits on what this says: the judge loads a **bf16** base
(`live_eval.load_model_and_tokenizer` takes no quantisation argument), so both arms
are scored as an NF4-*trained* adapter on a bf16 base — identical for both, so the
paired comparison holds, but the absolute accuracies are not NF4-inference numbers.
And adapter-init seed is uncontrolled, which is exactly why the design is paired by
subset; all 10 adapters have distinct md5s, so the pairing did its job.

Also kept: a monitor loop reported a false `ABORTED` because `grep -c` emits
`file:count` per file and broke a `paste -sd+ | bc` sum — a monitoring bug that
looks identical to a real abort, and touched no measurement. And every SSH session
dropped simultaneously mid-run while box uptime was unchanged; all 10 runs
completed `rc=0`, which is the entire reason the harness runs under `setsid`.

Wall-clock averaged 339.8 s resident against 336.0 s streamed, including model load
and shard-cache read. That is **not** a throughput measurement and no tok/s claim is
made from it.

Raw: `/root/results/convergence_nf4_summary.json` and the per-run files.

## What was not done, and what these numbers do not say

- **The RAM-vs-disk gap is still unmeasured.** No NVMe on this box; see the DISK
  section for the measurement taken and the decision not to bypass the guard.
- **The NF4 defect has a diagnosis but no repair.** The mechanism is aliasing
  (STEP 9), and both obvious workarounds were measured and rejected: de-aliasing
  costs 8x peak VRAM, `pin=False` costs 7.41x throughput. Filed as #331 with the
  synthetic reproducer.
- **Why the defect is NF4-only, and why its boundary is so sharp, is not
  explained.** The bf16 path aliases the pool identically and is exact at 3.9x
  the bytes. Seven hypotheses were tested and rejected; none replaced them.
- **The threshold is a bracket, not a number** — 163.8 MiB/layer exact, 171.5
  broken, 4.7% wide.
- **The defect is not demonstrated end-to-end through `soup train`.** Both
  attempts were confounded because the CLI does not pin the adapter-init seed.
  The evidence is the controlled harness.
- **The quality result resolves ~1pp, not less.** 300 held-out items, five
  paired subsets, and an uncontrolled adapter-init seed. "No difference detected"
  is not "no difference".
- **`soup ship`'s leg-2 suites are coarse.** `mini_common_sense` has 24 items, so
  one item is 4.2% against a 0.05 threshold. The verdicts are directionally right
  and numerically blunt.
- **The 8-GPU comparison is on hardware where streaming's premise does not
  apply**, and on a PCIe box with no NVLink, which is the interconnect ZeRO-3
  most depends on.
- **Preference losses were not benchmarked at all** — `dpo`/`orpo`/`simpo`/`kto`
  streamed all fail on this torch before training starts (FINDING 2), so
  v0.72.4's claims are untested here.
- **Every number is one machine, one session**, on a stack (torch 2.13.0+cu130,
  bnb 0.50.0, trl 0.26.2, peft 0.20.0) that differs from the published records in
  everything but `transformers`. SM clock was 1980 MHz median-while-busy in every
  timed run, so no clock correction is applied anywhere.
- **Post-fix numbers are marked with the revision they were taken at**; see the
  measurement/fix boundary above.

---

## Does any of this change the preprint?

The preprint (*Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB
Laptop GPU*, DOI [10.5281/zenodo.21771064](https://doi.org/10.5281/zenodo.21771064))
measures Llama-3.1-8B NF4 on a 4 GB card. A published DOI cannot be quietly
corrected, so this is answered explicitly rather than left implicit.

**No measured number in it changes, and nothing in it is invalidated.**

- Its headline configuration is 8B NF4 at **105 MiB per layer**, comfortably
  below the 163.8–171.5 MiB boundary found here, and it survives a 50-backward
  soak with `worst_abs = 0.0`.
- Its throughput claim is independently reproduced on different hardware, a
  different OS and a much newer stack: 119.6 tok/s / 3.32 GB there against a
  median 113.00 tok/s / 3.32 GB here.
- Its exactness claim is *strengthened*, not weakened: what it could only assert
  on 3-layer from-config checkpoints is now shown against resident references at
  8B, 14B, 32B and 72B.

What the session adds is **scope beyond** what the preprint claims — behaviour at
32B and above, which it never asserted — plus the defect that lives there. If a
future version widens its scope past 14B, this record's threshold and the
pinning result have to go with it.

## Reproducing

Harness scripts live in the session scratchpad, not in the repo. Each is small
and self-contained:

| script | what it does |
|---|---|
| `bitexact.py` | shard -> stream -> compare logits/gradients/loss curve against a resident reference of matching numerics |
| `graddiff.py` | gradients after one backward + each model's own curve twice |
| `determinism.py` | forward, backward and curve reproducibility of one model |
| `repeat_backward.py` | N streamed backwards against one deterministic resident reference; `--pin`, `--buffers`, `--order` |
| `layercount.py` / `depth_vs_bytes.py` | synthetic Llamas sweeping depth, per-layer bytes and quantisation |
| `ckpt_hypothesis.py` | flips `StreamedDecoderLayer.use_checkpoint` at runtime, both arms |
| `mechanism.py` / `mechanism_cost.py` | `sync` vs `clone` vs control, and what each costs |
| `pincost.py` | pinned vs pageable throughput, correctness asserted in the same process |
| `prep_convergence.py` | the emotion-classification subsets and held-out set |
| `runbench.sh` / `variance.sh` / `runbench8.sh` | one `soup train` with VRAM and SM-clock sampling; n repeats; 8-GPU variant under torchrun |

All of them import `soup_cli` unmodified: every measurement above the
measurement/fix boundary was taken against untouched Soup. The two `src/` changes
made afterwards (STEP 10's guard, STEP 12's `device_map` fix) are commits in
their own right, each with tests, and neither is exercised by the harnesses
above.
