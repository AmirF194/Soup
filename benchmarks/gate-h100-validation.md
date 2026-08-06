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

**Status: IN PROGRESS.** This file is written as the work happens and committed
after each measurement. Sections appear in the order they were run, not in the
order they will eventually read best.

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
auto -> 0, `stage3_prefetch_bucket_size` auto -> 5e6). Result below.
