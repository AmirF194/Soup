# Measurement records

Raw gate records for Soup's layer-streaming feature, published as written.

These are not a report assembled after the fact. They are the working records
kept while each item was built and verified, so they contain the failures, the
assumptions that turned out wrong, and the numbers that were measured and then
discarded — in the order those things happened. They are the evidence behind
the paper *Exact Layer Streaming: LoRA Fine-Tuning of an 8B Model on a 4 GB
Laptop GPU*.

| File | What it gates | Headline |
|---|---|---|
| [`gate-v0.72.0-layer-streaming.md`](gate-v0.72.0-layer-streaming.md) | The streaming path itself | Bit-exactness vs a resident reference; 3B bf16 trained on a 4 GB card |
| [`gate-v0.72.2-nf4.md`](gate-v0.72.2-nf4.md) | NF4 quantised streaming | Llama-3.1-8B at 119.6 tok/s in a 3.32 GB peak |
| [`gate-v0.72.3-breadth.md`](gate-v0.72.3-breadth.md) | Nine architectures, batching, accumulation, resume, disk tier | Peak-VRAM predictor at 0.85% worst-case error; accumulation is per-token I/O-neutral |

## Hardware

Every number was measured on one machine:

- **GPU** — RTX 3050 Laptop, 4 GB (4.29 GB usable)
- **Host** — 16.9 GB RAM, NVMe
- **OS** — Windows 11

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
- **Derived figures are labelled as arithmetic.** Where a line says "1M tokens =
  2.3 h", that is division, not a measured wall-clock run.

## Reproducing

The implementation ships in Soup under Apache-2.0. Reproduction commands are in
Appendix A of the paper; the correctness protocol runs as part of the project's
test suite, so a regression in bit-exactness fails CI rather than reaching a
user.

```bash
pip install 'soup-cli[train]'
```
