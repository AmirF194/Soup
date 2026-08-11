# Measurement records

Raw gate records for Soup's layer-streaming feature, published as written.

These are not a report assembled after the fact. They are the working records
kept while each item was built and verified, so they contain the failures, the
assumptions that turned out wrong, and the numbers that were measured and then
discarded — in the order those things happened.

They are the evidence behind the preprint:

> Makazhan, A. (2026). *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model
> on a 4 GB Laptop GPU.* Zenodo.
> [10.5281/zenodo.21771064](https://doi.org/10.5281/zenodo.21771064)

| File | What it gates | Headline |
|---|---|---|
| [`gate-v0.72.0-layer-streaming.md`](gate-v0.72.0-layer-streaming.md) | The streaming path itself | Bit-exactness vs a resident reference; 3B bf16 trained on a 4 GB card |
| [`gate-v0.72.2-nf4.md`](gate-v0.72.2-nf4.md) | NF4 quantised streaming | Llama-3.1-8B at 119.6 tok/s in a 3.32 GB peak |
| [`gate-v0.72.3-breadth.md`](gate-v0.72.3-breadth.md) | Nine architectures, batching, accumulation, resume, disk tier | Peak-VRAM predictor at 0.85% worst-case error; accumulation is per-token I/O-neutral |
| [`gate-v0.72.4-preference-losses.md`](gate-v0.72.4-preference-losses.md) | DPO / ORPO / SimPO / KTO over the streaming engine | DPO's reference model costs no extra weights — 0.914x the SFT peak, against +730.44 MB for a real second instance |
| [`probe-v0.73.0-what-bounds-streaming.md`](probe-v0.73.0-what-bounds-streaming.md) | What the streamed step is actually bound by, and Cut Cross-Entropy on top of it | **Not** transfer-bound: 71.3% of the card's same-session GEMM ceiling, and deleting every host-to-device byte buys 1.4%. CCE triples the usable microbatch for +9.6% |
| [`gate-h100-validation.md`](gate-h100-validation.md) |  The method on someone else's hardware: bit-exactness at real sizes, convergence quality, DeepSpeed, variance | **Forward** bit-exact to 72B; **backward** bit-exact to 14B NF4 pre-repair, re-gated after the STEP 14 fix at 32B (256/256) **and at 72B (320/320, the size where the defect was worst)**; 2.93x DeepSpeed ZeRO-3 offload in 9.7x less VRAM; and the silent wrong-gradient defect that fix repairs |

## Hardware

Every number in the four `gate-v0.72.*` records was measured on one machine:

- **GPU** — RTX 3050 Laptop, 4 GB (4.29 GB usable)
- **Host** — 16.9 GB RAM, NVMe
- **OS** — Windows 11

`gate-h100-validation.md` is the exception and the reason it exists: 8x H100
80 GB, 503 GB RAM, Ubuntu 24.04, on a much newer torch/bitsandbytes/trl/peft
stack. It is the first record from hardware other than the laptop, and the first
able to hold a *resident* reference for an 8B–72B model — which is what turns
"bit-exact on a 3-layer toy" into "bit-exact on real models".

Windows/WDDM matters for reading these: it spills into shared host memory rather
than raising `CUDA out of memory`, so a run completing is not evidence that its
configuration fits. That is why peak VRAM is reported alongside every throughput
figure, and why the fit decision refuses rather than warns.

## Reading the numbers

- **Throughput is quoted with the SM clock it was taken at.** This card's boost
  clock varies about 13% between sessions, so a fraction-of-ceiling stated
  without its clock is not meaningful. Where a GEMM ceiling is compared against,
  it was measured in the same session.
- **The correctness reference always matches the numerics under test** — a
  streamed NF4 run is compared against a *resident NF4* run, never against
  resident bf16, which would hide a real defect inside quantisation error.
- **"Bit-exact" is always two claims, never one.** The **forward** (logits,
  `torch.equal`) and the **backward** (every LoRA gradient tensor) are measured
  independently and do not always agree: in `gate-h100-validation.md` the forward
  is exact at every size up to 72B while the backward, pre-repair, was wrong above
  ~165 MiB per NF4 layer. So "bit-exact at 72B" on its own is not a statement this
  record makes — check which half, at which quantisation, at which MiB/layer. That
  file opens with a per-model ledger giving all four for every row, and marks
  anything unmeasured "not tested" rather than leaving it blank.
- **Derived figures are labelled as arithmetic.** Where a line says "1M tokens =
  2.3 h", that is division, not a measured wall-clock run.

## Reproducing

The implementation ships in Soup under Apache-2.0. Reproduction commands are in
Appendix A of the paper; the correctness protocol runs as part of the project's
test suite, so a regression in bit-exactness fails CI rather than reaching a
user.

```bash
pip install "soup-cli[train]"
```
