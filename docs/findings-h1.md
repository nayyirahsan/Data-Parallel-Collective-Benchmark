# H1 result: the α-β model's error is governed by α/α_sw

**H1 as stated is refuted at the margin. Its mechanism is confirmed decisively,
and the refined finding is stronger than the original claim.**

H1 predicted the textbook α-β model would underestimate the ring/PS crossover by
**≥5×**. Measured maximum is **4.43×** — close, but under the threshold, so the
literal hypothesis loses.

What the sweep shows instead is that the error is not a constant at all. It is a
monotone function of how large the injected network latency is relative to the
independently measured software floor:

| p | α | α/α_sw | measured N* | textbook N* | textbook error |
|---|---|---|---|---|---|
| 8 | 50 µs | 0.20 | 518.7 KB | 117.2 KB | **4.43×** |
| 4 | 50 µs | 0.36 | 431.6 KB | 97.7 KB | **4.42×** |
| 8 | 200 µs | 0.78 | 874.6 KB | 468.8 KB | 1.87× |
| 4 | 200 µs | 1.43 | 574.1 KB | 390.6 KB | 1.47× |
| 8 | 1000 µs | 3.91 | 2.48 MB | 2.29 MB | 1.09× |
| 4 | 1000 µs | 7.14 | 1.91 MB | 1.91 MB | **1.00×** |

(β = 0.4 ns/byte throughout; α_sw is the per-p floor from `experiments/floor.yaml`.)

The model becomes **exact** once α ≫ α_sw, and degrades to ~4.4× once α ≪ α_sw.
That is the expected shape: the textbook α is network latency, and the error is
roughly the factor by which the true per-step cost (α + α_sw) exceeds it.

### Why this is the more useful result

It explains *when to trust the model rather than whether to*. The α-β model is
trusted in HPC because MPI collectives in C over InfiniBand sit at α/α_sw ≫ 1,
where this data says it is accurate to 1.00×. Python-level frameworks sit at
α/α_sw < 1, where the same model misprices the crossover by over 4×. Both
observations come from one sweep.

### Second finding: at β=0 the model does not just mispredict, it predicts the wrong sign

With no injected bandwidth term, the textbook model says ring can **never** beat
PS — ring pays more steps for no bandwidth benefit, so PS wins at every message
size. Measured, ring wins above **1.85 MB at p=4** and **3.72 MB at p=8**.

The model is not slightly off here; it is qualitatively wrong, because it treats
the link as having zero per-byte cost while the machine itself has one
(b_sw = 0.09–0.20 ns/byte, i.e. 40–87 Gbit/s effective). Any real system has a
bandwidth term whether or not the model includes one.

### Corrected model

Refitting with the independently measured per-p floor — (a_sw, b_sw) from the
α=β=0 experiment, never tuned to this data — brings predictions to within
**8–60%, median 22%**, versus up to 4.43× for the textbook form, and recovers
the correct sign in every β=0 case.

The residual ~22% is not noise; it is the α-β model's linearity assumption
breaking down. Effective bandwidth is itself size-dependent (a 512 B step costs
219 ns/byte, a 4 MB step 0.16 ns/byte), so a single β cannot describe both ends
of the range. Capturing that would need a three-parameter per-step model.

### Caveats stated plainly

- Crossovers marked `*` in the analysis output lie beyond the largest measured
  size (8 MB) and are extrapolations from the linear fit; they are reported as
  bounds, not measurements.
- The p=2, β=0 case is degenerate: ring and PS both reduce to exactly 2α, so no
  crossover exists in either model or measurement.
- The software floor grows with p through resource contention between
  co-located processes (see `floor.md`), which is an artefact of simulating
  workers on one machine. It is absorbed per-p rather than left in the
  comparison.
