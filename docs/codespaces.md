# Settling H2 on GitHub Codespaces (Student Developer Pack)

Codespaces is the lowest-friction way to get more cores: no VM provisioning, no
credit card, and the repo is already configured. The Student Pack includes
GitHub Pro, which carries **180 core-hours/month free**. A full H2 run is under
two hours, so a 32-core machine costs ~64 core-hours — about a third of a
month's allowance.

## The catch, and why the tooling checks for it

Codespaces runs in a VM, and **VM timer granularity determines how many cores
you need**. The shim's core-yielding delay (sleep 70% of each delay, spin the
tail) only works if the platform can deliver a sub-200 µs sleep:

| platform | sleep granularity | core-yielding at α=200 µs | cores per worker |
|---|---|---|---|
| macOS (M4 Pro) | 5 µs | active | ~0.33 |
| Docker Desktop (linuxkit) | **997 µs** | inactive, falls back to spin | ~0.6 |

If Codespaces has a coarse timer like Docker Desktop's, each worker spins a full
core and p=8 needs ~48 cores rather than ~24. The shim detects this
automatically and degrades to pure spinning rather than silently overrunning
deadlines — but it changes what machine size you need.

**So check first.** It takes seconds:

```bash
./scripts/preflight.sh
```

It reports measured timer granularity, cores, and the world size to expect —
extrapolated from gate outcomes actually observed on the reference machine, not
from a contention model.

## Steps

1. On the repo page: **Code → Codespaces → ... → New with options**
2. Machine type: **32-core** if offered, otherwise 16-core. (Personal accounts
   often cap at 16; larger types may need enabling in
   Settings → Codespaces.) The devcontainer requests 16 as a floor.
3. Wait for `postCreateCommand` to install CPU torch and `iproute2`.
4. Then:

```bash
./scripts/preflight.sh          # what can this machine support?
./scripts/settle_h2.sh 8        # gated: refuses if it cannot
```

If the gate refuses at p=8, fall back to the configuration that is known to pass
on 12 fine-timer cores:

```bash
./scripts/settle_h2.sh 4
```

## Expected outcomes

| cores | timer | expected | notes |
|---|---|---|---|
| 16 | fine | p≈5 | p=8 probably refused; run p=4 |
| 32 | fine | p≈10 | **p=8 reachable — the target** |
| 16 | coarse | p≈2 | not worth it |
| 32 | coarse | p≈5 | p=8 refused |

The 32-core machine with a fine timer is the combination that settles H2 as
originally stated. Everything else still improves on the current p=4 result only
marginally, so check `preflight.sh` before burning core-hours.

## Reading the result

`dpt.analysis.jitter_impact` prints a **ring/ps** column.

- Current gated measurement at p=4: **1.15–1.25×** (0.84× at 1 MB).
- Simulator predicts **1.08×** at p=8, flat in p.
- H2 predicted **≥4×, growing with p**.

If p=8 lands near 1.1× and flat, H2 is refuted by direct measurement at the
worker count it was stated for, and the simulator is validated there too —
which would let `findings-h2.md` drop its "model validated only at p≤4" caveat
entirely. That is the single biggest remaining weakness in the write-up.

If it lands at ≥4×, the refutation was an artefact of small p and the finding
inverts. Say so plainly; both outcomes are publishable.

## Also in the Student Pack, if Codespaces is capped

- **DigitalOcean, $200** — a 32-vCPU droplet is ~$0.48/hr; hours of headroom.
- **Azure for Students, $100, no credit card** — `Standard_F32s_v2` is 32 vCPU.

Both need `pip install -e .` and give a plain Linux box, usually with a finer
timer than a nested container. `./scripts/preflight.sh` answers that in seconds.

## The other thing worth doing there

A real Linux box with a high-resolution timer also closes the `tc netem`
limitation: netem quantises to ~1 ms in Docker Desktop, so the shim could only
be validated against it at 4–32 ms, not at the 50 µs–1 ms the experiments use
(`netem-validation.md`). On a machine where `preflight.sh` reports a fine timer:

```bash
sudo apt-get install -y iproute2
sudo python -m dpt.bench.netem --delays-us 0 100 250 500 1000
```

If netem tracks linearly down to 100 µs there, the shim is validated directly in
its operating range and that caveat disappears too.
