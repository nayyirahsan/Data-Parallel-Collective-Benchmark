# The software overhead floor, and why it is fitted per worker count

Measured with α = β = 0, so every nanosecond is Python dispatch, buffer
management, and the gloo substrate.

| p | a_sw (per step) | b_sw | effective bandwidth | R² |
|---|---|---|---|---|
| 2 | 84.6 µs | 0.092 ns/B | 87.2 Gbit/s | 0.85 |
| 4 | 140.1 µs | 0.150 ns/B | 53.2 Gbit/s | 0.80 |
| 8 | 255.5 µs | 0.198 ns/B | 40.4 Gbit/s | 0.54 |

A single global fit across all p gives R² = 0.65 with residuals to 213%, because
both terms grow with p. Two causes were candidates, and the per-rank data
separates them:

| | contribution |
|---|---|
| slowest-of-p gating (max over ranks vs mean) | **1.02–1.11×**, barely grows with p |
| per-rank mean growth from p=2 to p=8 | **1.55–1.99×** |

So the p-dependence is **resource contention between co-located processes**
sharing memory bandwidth and cores — not order statistics. The floor is
therefore a property of the machine *at a given co-location level*, and is
fitted independently for each p rather than treated as a machine constant.

### Two consequences

**For validity.** This contention is an artefact of simulating workers on one
machine and would not appear on a real cluster. Absorbing it into a per-p floor
keeps it out of the injected-α comparison that H1 and H2 actually test.

**For H2.** Baseline slowest-of-p amplification is only 1.02–1.11×, so the
clean benchmark carries very little natural tail noise. Injected jitter will
therefore show up as signal rather than being swamped — H2 is measurable on this
apparatus. Had this factor been large, it would have been a blocker.

### Residual misfit

R² falls to 0.54 at p=8 because effective bandwidth is size-dependent: a 512 B
step costs 219 ns/byte while a 4 MB step costs 0.16 ns/byte, a ~1400× spread. A
two-parameter linear-in-N model cannot span that, which is itself the α-β
model's linearity assumption failing.
