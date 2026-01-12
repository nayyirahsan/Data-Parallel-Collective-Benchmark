# Calibration gate: delay injection fidelity

The (α, β) axis is only a controlled variable if the shim can deliver a requested
delay faithfully. This measurement runs before anything else and is reproducible
via `python -m dpt.bench.calibrate`. Numbers below are from the development
machine: Apple M4 Pro, 12 cores, 24 GB unified memory, macOS 25.6, Python 3.13.1.

## Finding 1 — `time.sleep` carries a proportional bias, not a granularity floor

| target | median | error |
|---|---|---|
| 10 µs | 15.9 µs | +59% |
| 100 µs | 130.5 µs | +31% |
| 1 ms | 1269.9 µs | +27% |
| 10 ms | 12071.5 µs | +21% |

The error stays near +25% at *every* scale rather than shrinking as the target
grows. This rules out the correction a granularity floor would permit (subtract a
fixed offset) and rules out sleep entirely for this use.

## Finding 2 — pure spin-wait beats the hybrid strategy

The original design specified spin below 1 ms, sleep above. Measured, that hybrid
is **worse than pure spin above 1 ms**, because the sleep leg drags its
proportional bias back in:

| target | spin error | hybrid error |
|---|---|---|
| 100 µs | +0.1% | +0.1% |
| 1 ms | +0.0% | +0.0% |
| 5 ms | +0.0% | **+17.7%** |
| 10 ms | +0.0% | **+21.3%** |

**Design changed: pure spin-wait, no sleep leg.** Pure spin holds median error
≤1.7% from 10 µs up and 0.0% from 50 µs up.

## Finding 3 — the usable worker count is p ≤ 8, and this bounds H2

Spin-waiting consumes a core, so concurrent workers contend. Median accuracy is
unaffected (0.0–0.4% error at every p and target), but the **tail** degrades once
the machine is saturated:

| p | α=50 µs p99 err | α=200 µs p99 err | α=1 ms p99 err |
|---|---|---|---|
| 1 | +17.8% | +6.0% | +0.4% |
| 2 | +6.8% | +0.3% | +0.9% |
| 4 | +14.7% | +2.8% | +1.4% |
| 8 | **+33.1%** | +3.8% | +1.0% |
| 12 | +69.2% | +112.2% | +58.4% |

At p=12 (spinners = cores) the scheduler is starved and the tail blows up, with
outliers to 1.9 ms. **p ≤ 8 is therefore an empirical constraint, not a guess.**

### Consequence for H2

H2's claim is about **p99** degradation under injected tail jitter. The apparatus
itself has a p99 noise floor of +33% at (p=8, α=50 µs) — in that corner, injected
tail latency is confounded with scheduler tail latency and the result would be
uninterpretable.

**Enforced in the sweep config: H2 runs at α ≥ 200 µs**, where the p=8 tail error
is +3.8%. The low-α corner remains valid for H1, which is a median-statistic
claim and is unaffected (median error 0.4% at p=8, α=50 µs).

This is the first real finding of the project: the measurement apparatus has a
regime of validity, and it was located before it could contaminate a result.
