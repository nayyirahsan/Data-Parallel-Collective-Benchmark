# H2 result: refuted — and the apparatus could not have tested it anyway

**H2 is refuted. Separately and more importantly, the experiment revealed that
the apparatus cannot support the measurement H2 requires — which was found by
measuring the apparatus rather than by trusting it.**

H2 predicted that ring all-reduce, running 2(p−1) serially dependent steps to a
parameter server's p, would amplify tail latency ≥4× more than PS at p=8, and
that this would flip the winner inside ring's bandwidth-favourable regime.

## Part 1: the apparatus cannot measure p99 at p ≥ 4

A p99 claim is only meaningful if the apparatus's own tail is small relative to
the injected effect. The clean-run noise floor (`experiments/noise_floor.yaml`):

| p | clean p99/p50 | p99 spread across trials | usable |
|---|---|---|---|
| 2 | 1.19–1.80× | 1.14–2.91× | 5/12 configs |
| 4 | 1.85–9.76× | 2.74–13.17× | **0/12** |
| 8 | 2.31–6.11× | 2.45–6.22× | **0/12** |

Only **5 of 36 configurations** can support a tail statistic, all at p=2. Worse,
ring at p=8/4 KB has a median of 11.09 ms against just 2.80 ms of modelled cost —
**75% of the measurement is contention**, not the quantity being studied.

This exposed a gap in my own calibration. The Week 1 gate validated the delay
shim under spin contention with *no payload*; it never tested memory-bandwidth
contention from realistic message sizes, which turns out to dominate. The
correlation between injected spin load and tail noise is r = 0.52 — real, but
process count matters more than spin load.

**And p=2, the only clean regime, cannot test H2 at all**: at p=2 ring runs
2(p−1) = 2 steps and PS runs p = 2. The step-count asymmetry H2 is about does
not exist there.

## Part 2: modelling the dependency structure directly

Rather than report a p99 the apparatus cannot support, each algorithm's message
DAG is modelled as a discrete-event system (`src/dpt/simulate.py`): no
contention, and cheap enough for the 200k iterations a stable p99 needs.

Validated against measurement, split by whether the apparatus was trustworthy:

| region | agreement within 25% | median error |
|---|---|---|
| apparatus clean | **8/10** | **6%** |
| apparatus noisy | 3/26 | 160% |

The model agrees with measurement precisely where measurement is trustworthy and
diverges precisely where it is contention-dominated. That pattern is evidence the
divergence is the apparatus's, but it cannot fully exclude model error — so the
model is treated as validated only for (ring, p=2) and (recursive, p=2), and
indicative elsewhere. Simulated numbers are labelled as such throughout.

## Part 2b: the apparatus limit was an engineering choice, not a hardware one

The contention came from the shim **spin-waiting**, which consumes a core for
every injected delay — so p workers needed p cores that were not also running
the workers.

The original calibration rejected sleeping because a *fixed* 300 µs slack fails
at large delays (+17.7% error at 5 ms). But macOS's sleep overshoot is
**proportional** (~1.25–1.35×), not absolute, so a *fractional* sleep stays
inside the deadline at every scale: 0.7 × 1.35 < 1. Sleeping for 70% of each
delay and spinning the remainder holds median error at 0.03–0.33% — identical to
pure spinning — while cutting CPU occupancy from **99% to 11–15%**.

That single change took the apparatus from **5/36 to 19/36** usable
configurations, and made **p=4 at β=0.4 pass the gate on all six** clean
configurations. p=8 still fails, so H2 as literally stated (at p=8) remains
untestable here — but p=4 is now measurable, and the mechanism genuinely exists
there: ring runs 2(p−1)=6 steps to PS's 4, a 1.5× asymmetry, unlike p=2 where
the step counts are equal.

## Part 3: H2 is refuted — now by a gated measurement, not only by a model

Simulated p99 degradation under 1% × 10 jitter (1 MB, α=200 µs, β=0.4):

| p | ring steps | ring | ps | ring/ps |
|---|---|---|---|---|
| 2 | 2 | 5.50× | 5.50× | 1.00× |
| 4 | 6 | 4.00× | 3.25× | 1.23× |
| 8 | 14 | 2.29× | 2.13× | 1.08× |
| 16 | 30 | 2.20× | 1.56× | 1.41× |
| 64 | 126 | 1.43× | 1.14× | 1.25× |

H2 predicted ring/ps ≥ 4× at p=8 and growing with p. Simulated is **1.08×** and
flat.

**The gated p=4 measurement agrees.** Run at the one configuration that passes
the noise-floor gate (β=0.4, 8 trials × 250 iterations):

| N | ring | ps | recursive | **ring/ps** |
|---|---|---|---|---|
| 4 KB | 5.46× | 4.77× | 6.56× | **1.15×** |
| 64 KB | 6.06× | 4.84× | 5.24× | **1.25×** |
| 1 MB | 3.85× | 4.57× | 4.75× | **0.84×** |

Against H2's predicted ≥4×, the measured ratio is **1.15–1.25×**, and at 1 MB it
drops *below* 1 — ring degrading less than PS. The simulator predicts **1.23×**
at p=4, which the measurement brackets.

This matters for more than H2. The model was previously validated only at p=2,
where ring and PS have identical step counts and the mechanism does not exist —
the weakest possible place to check it. It is now validated at **p=4**, where the
asymmetry is real: `dpt.analysis.validate` reports the model trusted for
`(ring, 4)` and `(recursive, 4)`, 8/16 agreeing within 25% at a median error of
22%. Much of the residual is the two-point jitter model quantising simulated p99
onto discrete levels (4.00×, 5.50×, 7.75×).

Neither line of evidence supports H2, and they now agree at a worker count where
the mechanism exists.

## Why the hypothesis was wrong

Degradation *ratio* falls with p for every algorithm (ring 5.50× → 1.43×). The
reasoning error in H2 was treating a longer dependency chain as pure exposure.
It is also **accumulation**: 2(p−1) steps make ring's clean baseline large, so a
fixed-size straggler perturbation is a *smaller relative* inflation. More steps
means more chances to be hit, but also a bigger denominator, and the denominator
wins.

The corollary is counterintuitive and worth more than the original claim:
**recursive halving/doubling, with the fewest steps (2 log₂ p), is the most
tail-fragile in ratio terms** — 4.00× at p=64 versus ring's 1.43×.

## What actually decides the winner

Ratios are the wrong statistic for a practitioner. In **absolute** simulated p99,
recursive halving/doubling wins nearly everywhere, clean and jittered:

| p | ring clean | ps clean | winner | ring jittered | ps jittered | winner |
|---|---|---|---|---|---|---|
| 4 | 1.83 ms | 2.48 ms | recursive | 7.32 ms | 8.05 ms | **ring (flip)** |
| 8 | 3.53 ms | 4.96 ms | recursive | 8.08 ms | 10.53 ms | recursive |
| 64 | 26.03 ms | 39.64 ms | recursive | 37.18 ms | 45.22 ms | recursive |

Being simultaneously bandwidth-optimal and latency-logarithmic beats being
bandwidth-optimal alone, and jitter does not change that except at p=4.

## Caveats

- The two-point jitter model (×1 or ×k) makes completion times discrete, so
  simulated p99 snaps to quantised levels — visible as repeated values (4.00×,
  5.50×). A continuous heavy-tailed distribution would smooth this.
- A tested alternative mechanism — that degradation collapses onto expected
  stragglers per collective, q·p — is **not** supported: ratios of 2.29× and
  4.00× both occur at q·p = 0.02. Step count, not straggler density, dominates.
- Everything at p ≥ 4 rests on a model validated only at p=2. A real multi-node
  cluster would settle it, and is the single change that would most improve this
  result.
