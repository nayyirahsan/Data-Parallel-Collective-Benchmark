# At the real gradient size: ring wins where bandwidth is scarce, PS where it isn't

ResNet-18's flattened gradient buffer is **44.7 MB** (11.17 M parameters, fp32) —
roughly 50× above every crossover measured in the H1 grid, so ring should be
well clear of the parameter server. It is, but only in the regime the crossover
analysis says it should be.

Median all-reduce time, 44.7 MB:

| p | β (ns/B) | ring | recursive | PS | winner | PS / winner |
|---|---|---|---|---|---|---|
| 2 | 0.05 | 11.1 ms | 10.6 ms | 14.8 ms | recursive | 1.40× |
| 2 | 0.40 | 34.6 ms | 34.2 ms | 56.4 ms | recursive | 1.65× |
| 4 | 0.05 | 36.0 ms | 41.0 ms | 34.8 ms | **PS** | 1.00× |
| 4 | 0.40 | 64.7 ms | 68.0 ms | 103.3 ms | ring | 1.60× |
| 8 | 0.05 | 101.0 ms | 125.0 ms | 76.9 ms | **PS** | 1.00× |
| 8 | 0.40 | 110.9 ms | 153.6 ms | 197.3 ms | ring | 1.78× |

(α = 200 µs.)

**The headline is the β dependence, not the message size.** At β = 0.4 ns/byte
(≈20 Gbit/s) ring beats PS by up to **1.78×** at p=8. At β = 0.05 ns/byte (a fast
intra-node link) **the parameter server wins at p ≥ 4**, by up to 1.31×, despite
the message being 50× past the crossover measured at β = 0.4.

This is the practical form of the H1 result. "Ring wins above N\*" is only true
for a given β; when bandwidth is plentiful the per-step software floor dominates
and PS's p steps beat ring's 2(p−1). A crossover quoted as a single message size,
which is how it is usually reported, is underspecified.

## Open question: recursive halving/doubling is slower than ring at p ≥ 4

Recursive moves **identical total bytes** to ring — 2N(1−1/p) = 78 MB at p=8 — in
6 steps rather than 14, and pays the same injected link cost. It should win. It
loses by 1.4× at p=8.

Two explanations were proposed and **both were tested and refuted**:

1. **Allocation overhead.** Recursive allocated a receive buffer per step (up to
   22 MB) while ring allocated once. Fixing this appeared to make recursive 10–30%
   *slower* across separate runs — but an interleaved, trial-paired A/B showed
   **no difference at any configuration** (all CIs span zero). The apparent
   regression was between-run machine drift, and PyTorch's caching allocator makes
   `torch.empty` effectively free. The preallocated version was kept as the
   cleaner code, not as an optimisation.

2. **Cache behaviour at large transfers.** Recursive's first step moves one 22 MB
   block where ring moves 5.6 MB chunks, so the per-byte cost might worsen beyond
   cache capacity. Measured directly, effective bandwidth **plateaus at ~0.19
   ns/byte (41 Gbit/s) from 2 MB to 64 MB** with no turnaround. Refuted.

The gap does widen with β (25 ms at β=0.05, 50 ms at β=0.4) even though both
algorithms are charged identical total injected cost, which points at the
*distribution* of spin time — recursive's concentrated 8.8 ms spins versus ring's
14 spread 2.2 ms ones — interacting with 8 co-located spinning processes. That is
a hypothesis, not a finding; it has not been tested and is recorded as open.
