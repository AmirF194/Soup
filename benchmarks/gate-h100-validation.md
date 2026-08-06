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
