# End to end: the collective was not the bottleneck until I fixed the staging

The microbenchmarks say ring beats the parameter server by up to **1.78×** on a
44.7 MB gradient buffer. Inside a real ResNet-18 training step that advantage was
initially worth **3.2%**, because something else dominated.

## What dominated

First measurement, p=4, global batch 128, α=200 µs, β=0.4:

| collective | step | compute | all-reduce | gather+scatter | comm share |
|---|---|---|---|---|---|
| ring | 349.6 ms | 168.7 ms | 73.6 ms | **107.3 ms** | 51.7% |
| ps | 352.1 ms | 155.7 ms | 99.0 ms | **97.3 ms** | 55.8% |

Flattening gradients into the contiguous buffer and writing them back cost *more
than the all-reduce itself*, and that cost is identical for every algorithm — so
it diluted the entire comparison.

## Diagnosis

Packing 62 parameter tensors into a host-side buffer issues **one device-to-host
transfer per tensor**. Isolating it:

| | per-tensor loop | single fused transfer |
|---|---|---|
| CPU | 0.9 ms | 1.2 ms |
| MPS | **23.4 ms** | **7.8 ms** |

On CPU both are free, so the cost is entirely the device boundary crossing, not
the packing arithmetic — each small transfer pays its own launch and
synchronisation overhead.

## Fix and effect

Two buffers, one per memory space: pack into a device-side flat buffer with
on-device copies, then cross to the host **once**.

| | before | after | |
|---|---|---|---|
| staging (pack + transfer) | 107.3 ms | 16.7 ms | **6.4× faster** |
| step time (ring) | 349.6 ms | 250.0 ms | **−28.5%** |
| ring vs PS end to end | 1.01× | **1.11×** | |
| value of picking the right collective | 3.2% | **9.7%** | |

After the fix:

| collective | step | compute | all-reduce | pack | transfer | comm share |
|---|---|---|---|---|---|---|
| ring | 250.0 ms | 157.9 ms | 75.4 ms | 10.2 ms | 6.5 ms | 36.8% |
| recursive | 259.9 ms | 165.8 ms | 75.6 ms | 10.8 ms | 7.7 ms | 36.2% |
| ps | 276.9 ms | 160.8 ms | 98.9 ms | 11.2 ms | 6.0 ms | 41.9% |

## Confirmed on real CIFAR-10

The numbers above use synthetic CIFAR-shaped data. Repeated on the real dataset
(p=4, global batch 128, α=200 µs, β=0.4):

| collective | step | compute | all-reduce | pack | transfer | comm share |
|---|---|---|---|---|---|---|
| ring | 257.6 ms | 158.8 ms | 80.2 ms | 11.3 ms | 7.4 ms | 38.4% |
| recursive | 258.4 ms | 159.9 ms | 81.1 ms | 10.3 ms | 7.1 ms | 38.1% |
| ps | 271.3 ms | 153.7 ms | 100.3 ms | 11.1 ms | 6.2 ms | 43.3% |

Same structure; ring beats PS by 1.05×, worth 5.0% of a step (lower than
synthetic's 9.7% because real data loading adds fixed per-step cost).

Training converges and all three collectives agree — 80 steps, lr=0.01,
GroupNorm, global batch 128:

| | loss trajectory (20-step means) | max \|diff\| vs 1 worker |
|---|---|---|
| 1 worker | 2.367 → 2.218 → 2.093 → 1.901 | — |
| 4 workers, ring | 2.367 → 2.220 → 2.098 → 1.894 | 4.9e-2 |
| 4 workers, ps | 2.367 → 2.219 → 2.097 → 1.902 | 2.1e-2 |
| 4 workers, recursive | 2.367 → 2.217 → 2.098 → 1.903 | 4.3e-2 |

(Divergence accumulates with run length: it is 5e-5 at 5 steps, which is what
the test suite gates on.)

## Why this is the useful part

The microbenchmark was not wrong; it was answering a narrower question than the
one that mattered. A 1.78× advantage on the collective was worth 3.2% of a step
because an algorithm-independent cost sat beside it, and no amount of refining
the collective comparison would have surfaced that. It took measuring the whole
step and attributing every phase.

The ordering also matters: the staging fix was worth **28.5%** of step time,
roughly three times what choosing the best collective was worth even after the
fix. The cheapest win was not in the algorithm under study.

## Caveat

gloo is CPU-only, so this device-boundary crossing is forced by the backend
choice. NCCL on CUDA all-reduces device tensors directly and would not pay it at
all — which means this particular bottleneck is an artefact of the gloo + MPS
combination, not a general property of data-parallel training. The measurement
method transfers; the specific number does not.
