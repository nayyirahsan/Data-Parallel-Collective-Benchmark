# Settling H2: what actually needs to change

The instinct is "get a real cluster." That is the more expensive answer and not
the one the data points to.

## The diagnosis

H2's measurement failed at p ≥ 4 because of **core contention, not network
realism**. The evidence is in `floor.md`: the p-dependence of the software floor
decomposes into slowest-of-p gating (1.02–1.11×, negligible) and per-rank mean
growth (1.55–1.99×, dominant). The second is co-located processes competing for
cores and memory bandwidth.

The transport shim spin-waits, because `time.sleep` on macOS overshoots by
+20–30% at every scale (`calibration.md`). Spinning consumes a core for the
duration of every injected delay, so **p workers need p cores that are not also
running the workers**. On 12 cores at p=8 the scheduler starves: p99 error
reaches +112% at p=12, and the apparatus's own clean p99/p50 hits 2–10× at p ≥ 4.

That is a core-count problem. A 32-vCPU machine fixes it. A multi-node cluster
also fixes it, but incidentally and at far greater cost and setup effort.

**Rule of thumb from this machine: budget ~3× cores per worker.** p=8 wants
24–32 cores; p=16 wants 48–64.

## The cheap path: one big-core-count VM

| provider | instance | vCPU | rough cost | good for |
|---|---|---|---|---|
| Hetzner | CCX43 | 16 dedicated | ~€0.25/hr | p=4, cheapest |
| Hetzner | CCX53 | 32 dedicated | ~€0.50/hr | p=8 |
| AWS | c7i.8xlarge | 32 | ~$1.43/hr on-demand, ~$0.45 spot | p=8 |
| GCP | c3-standard-44 | 44 | ~$2/hr | p=8–16 |

The full H2 run takes well under two hours, so this is a **few dollars**. Use
dedicated/compute-optimised instances — shared-tenancy burstable types (AWS `t`
family, most "shared vCPU" tiers) reintroduce exactly the scheduling noise being
eliminated, and will fail the gate.

Prefer a bare-metal or dedicated-CPU instance over a shared one even at equal
vCPU count. Verify before spending time:

```bash
nproc
lscpu | grep -E 'Model name|Socket|Thread|NUMA'
```

## Running it

```bash
git clone <your repo> && cd dis-par-training
pip install -e .
./scripts/settle_h2.sh 8        # the world size to settle
```

The script runs four stages and **stops if the machine cannot support the
measurement**:

1. Timer fidelity at this core count (`dpt.bench.calibrate`).
2. Apparatus tail-noise floor (`experiments/noise_floor.yaml`).
3. **Gate** — scoped to the operating point `h2_jitter.yaml` actually uses
   (β=0.4, N ≥ 1 MB). It fails loudly and says which criterion failed:
   p99/p50 too high, or p99 unstable across process launches.
4. Only on a pass: the H2 experiment, the degradation analysis, and simulator
   validation against the new measurements.

Stage 3 is the point of the script. Running H2 on a machine that fails the gate
produces numbers that look fine and mean nothing — which is exactly the trap the
original run fell into.

## Reading the result

`dpt.analysis.jitter_impact` prints a **ring/ps** column. That is the H2 answer:

- **≥ 4× and growing with p** — H2 was right after all, and the refutation in
  `findings-h2.md` was an artefact of the 12-core machine. Say so plainly.
- **≈ 1× and flat** — confirms the current refutation, but now as a
  *measurement* at p=8 rather than a model prediction validated only at p=2.
  That is a substantially stronger claim and `findings-h2.md` should be rewritten
  to lead with it.

Either way, run `dpt.analysis.validate` afterwards: with the apparatus clean at
p=8, its "apparatus clean" row finally covers p ≥ 4, and the simulator's
credibility can be stated for the regime the findings actually rely on — which
is currently the biggest hole in the H2 write-up.

## If you do want genuine multi-node

Worth doing only after the core-count version, because it answers a different
question: whether the shim's α-β abstraction matches a *physical* network
(that is `netem-validation.md`'s job at one node), and whether cross-machine
effects — NIC queuing, switch contention, clock skew — change the picture.

Minimum useful setup is 4 machines on one switch. Two is not enough: at p=2 ring
and PS have identical step counts, so H2's mechanism does not exist there. Set
`MASTER_ADDR`/`MASTER_PORT` and use the `env://` init method that
`GlooTransport` already supports, then set the shim to α=β=0 and let the real
network provide the cost.

The honest cost/benefit: this is days of setup for a result the 32-vCPU run gets
in an afternoon. The single-machine limitation is worth removing only once the
core-contention one is gone.

---

# What was tried locally first (2026-09-14)

Four cheaper routes were tested. **One worked** and moved H2 from a model
prediction to a gated measurement at p=4 without spending anything. The others
failed, which is what makes the remaining core-count recommendation
load-bearing rather than assumed.

## 0. Make delay injection stop burning a core — WORKED

The apparatus was capped because the shim spin-waits: p workers need p cores that
are not also running the workers. Sleeping for a **fraction** of each delay
(0.7) rather than a fixed slack holds identical accuracy at **11–15% CPU instead
of 99%** — see `decisions.md` #12.

| | before | after |
|---|---|---|
| usable configs for tail statistics | 5/36 | **19/36** |
| p=4, β=0.4 gate | fail | **pass (6/6)** |
| p=8 gate | fail | fail (1/6 at β=0) |

This yielded the p=4 H2 measurement in `findings-h2.md` (ring/PS = 1.15–1.25×
against a predicted ≥4×) and validated the simulator at a worker count where the
mechanism exists. **p=8 still needs more cores** — everything below still
applies for settling H2 as originally stated.

## 1. Sleeping for the *whole* delay instead of spinning — refuted

Before the fractional-sleep fix above, the question was whether Linux's
`nanosleep` was accurate enough to replace spinning outright. It is not — which
is why the eventual fix keeps a spin tail rather than dropping spinning.

Measured in a Docker Linux VM, sleep is **far worse**:

| target | macOS sleep error | Linux VM sleep error |
|---|---|---|
| 50 µs | +59% | **+1898%** (999 µs actual) |
| 200 µs | +31% | **+399%** (999 µs actual) |
| 1 ms | +27% | **+100%** (1996 µs actual) |

Everything quantizes to ~1 ms multiples — the **same ~1 ms timer quantum** that
makes `tc netem` unusable below 1 ms in this VM (`netem-validation.md`). One root
cause, two symptoms. Spin-wait held 0.0–0.4% median error on both platforms, so
decision #1 stands, now confirmed on a second platform for a different reason.

## 2. Smaller worker counts — refuted

At p=3, ring runs 2(p−1)=4 steps to PS's 3, so H2's asymmetry exists (ratio 1.33
against 1.75 at p=8). Tested at the best operating point (1 MB, β=0.4) with
**10 trials × 400 iterations**:

| | p99/p50 | p99 spread across launches | usable |
|---|---|---|---|
| ring p=3 | passes | **1.88×** | no |
| ps p=3 | passes | **2.14×** | no |

The tail *ratio* passes with enough sampling, but p99 remains unstable
**between cold process launches** — which more iterations cannot fix, because it
is a property of how each launch gets scheduled, not of sample size. p=2 is the
only stable configuration, and there ring and PS have identical step counts, so
the mechanism under test does not exist.

**After the fractional-sleep fix, p=4 became usable** and produced the measurement
in `findings-h2.md`. p=3 remains unusable and p=8 still fails the gate, so
settling H2 at the worker count it was stated for still needs more cores.

## 3. A methodology trap worth knowing

The first p=3 run reported ring at **7.23×** and was ready to be written up as a
failure. The cause was Docker Desktop running in the background — a 12-CPU Linux
VM on a 12-core host, competing with the benchmark. With it stopped and the load
average settled below 2, the same configuration measured **1.44×**, a 5×
difference from background load alone.

**Run these measurements on an idle machine and check `uptime` first.** The
tooling used to investigate the apparatus can contaminate the apparatus.
