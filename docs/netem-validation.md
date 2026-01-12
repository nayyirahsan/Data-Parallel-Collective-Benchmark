# Validating the transport shim against a real shaped link

Every result in this project rests on `ShapedTransport` charging α + N·β being a
faithful stand-in for a network that actually has that latency and bandwidth.
Until it is compared against a real shaped link, that is an argument rather than
a measurement — and it is the main open item in the README's limitations.

## Why it cannot run on macOS

`tc netem` is a Linux traffic-control feature. macOS has `dnctl`/`pfctl`
(dummynet), but its semantics differ enough that a comparison would be measuring
the difference between two shapers rather than validating the shim. Running a
Linux kernel locally via Docker is simpler and gives the canonical tool.

## Running it

```bash
open -a Docker              # macOS: start Docker Desktop and wait for it
./scripts/netem_validate.sh
```

Takes a few minutes. Results land in `results/netem_validation.json`.

`--cap-add=NET_ADMIN` is required to attach a qdisc. It is scoped to the
container's own network namespace and does not alter the host's networking.

To vary the sweep:

```bash
./scripts/netem_validate.sh --delays-us 0 100 250 500 1000 --iters 60
./scripts/netem_validate.sh --skip-bandwidth     # latency only, faster
```

## What it measures, and why the mapping is fitted

`tc netem delay D` delays every packet on egress. On loopback both the data
packet **and its TCP ACK** cross that qdisc, so the per-message cost may come out
as D or 2D depending on whether send completion waits on the ACK. Guessing the
factor would bake an assumption into the validation, so the script instead:

1. Sweeps the **shim's** α with no netem, fitting per-step cost against α.
2. Sweeps **netem's** delay with the shim disabled, fitting the same slope.
3. Reports the ratio of the two slopes.

Ring at p=2 is used as the probe because it runs exactly 2 steps, so per-step
cost divides out with no cost model in between. Small messages (4 KB) isolate the
latency term; the bandwidth pass uses 256 KB–4 MB to isolate β.

## Interpreting the verdict

| ratio | meaning |
|---|---|
| ≈ 1.0 | The shim's α maps 1:1 onto a netem delay. Nothing to change. |
| ≈ 2.0 | Each configured delay is paid twice per message (data + ACK). The shim stays faithful under a factor-of-2 rescaling — record it in `decisions.md` and restate α in the findings as "equivalent to a netem delay of α/2". |
| anything else | The two are not related by a constant factor. Inspect the per-point table before trusting the shim's absolute α semantics. |

A ratio of 1 or 2 both **validate** the shim — they differ only in what physical
link a given α corresponds to. Only the third case is a problem, and it would
mean the α-β abstraction does not describe this transport, which would be a
finding worth reporting in its own right.

Bandwidth is checked the same way: nominal β (from netem's `rate`) against the β
recovered by fitting per-step cost versus message size, for both the shim and
netem. Note that netem's rate limiter loses accuracy above a few Gbit/s, so the
sweep defaults to 1–2 Gbit/s (β = 8 and 4 ns/byte) where it is trustworthy —
deliberately slower links than the main experiments use, since the goal is to
validate the *model*, not to reproduce a specific operating point.

## If Docker is unavailable

Any Linux box works — a spare machine, a VM, or a cloud instance:

```bash
sudo apt-get install -y iproute2
pip install -e .
sudo python -m dpt.bench.netem
```

Google Colab is a weaker option: its runtimes usually lack NET_ADMIN, so `tc`
fails when attaching the qdisc. The script detects this and exits with a clear
message rather than silently reporting unshaped numbers.

---

# Results (run 2026-09-14, Docker Desktop, kernel 6.12.54-linuxkit)

**The shim is validated.** Both cost curves are linear in configured delay, and
the two instruments differ by a single constant factor that appears
independently in latency *and* bandwidth.

## The shim delivers what it claims

Measured against wall clock, across the range the experiments actually use:

| configured α | measured per-step |
|---|---|
| 0 | 76.2 µs |
| 250 µs | 321.3 µs |
| 500 µs | 576.0 µs |
| 1000 µs | 1100.5 µs |
| 2000 µs | 2098.2 µs |

**slope 1.014 µs/µs, R² = 0.9999**, intercept 73.6 µs — which is the software
floor, independently consistent with `floor.md`. Repeated at 4–32 ms: slope
1.004, R² = 1.0000. Fitted β came within 3–10% of nominal at every rate tested.

## netem applies its shaping exactly twice

| | shim | netem | ratio |
|---|---|---|---|
| latency slope | 1.004 | 2.090 | **2.08×** |
| β at 500 Mbit/s | +3% | +105% | **2.05×** |
| β at 1 Gbit/s | +5% | +113% | **2.13×** |

A factor of ~2 in latency and ~2 in bandwidth, arrived at by separate
measurements, is a mechanism rather than a coincidence: on loopback a packet
crosses the egress qdisc on both the send and the receive path, so netem's delay
and rate limiter each apply twice.

**Conversion: 1 µs of shim α ≡ 2.08 µs of configured `tc netem delay`.** The
findings' α values restate as equivalent netem delays by dividing by 2.08 — the
results themselves are unaffected, since the shim is internally consistent and
validated against wall clock.

## netem cannot shape below ~1 ms in this environment

The first run failed its linearity check (netem R² = 0.9474) and the raw numbers
showed why — `delay 250us` and `delay 500us` produced **identical** results.
A direct RTT probe confirms quantization:

| configured delay | measured RTT |
|---|---|
| 100 µs | 2000.1 µs |
| 250 µs | 1999.4 µs |
| 500 µs | 2015.7 µs |
| 1 ms | 4000.1 µs |
| 2 ms | 6004.9 µs |
| 8 ms | 20488.2 µs |

Everything at or below 500 µs collapses onto the same 2 ms floor. The kernel is
Docker Desktop's `linuxkit` VM and `/proc/timer_list` is unavailable, so hrtimer
resolution cannot be confirmed — but the behaviour is a virtualized-timer
limitation, not a property of the shim.

**This is the load-bearing caveat.** The main experiments run at α = 50 µs–1 ms,
which is *below* netem's usable resolution here. So the shim is validated against
netem at **4–32 ms** and against **wall clock** at 250 µs–2 ms (R² = 0.9999), and
the claim that it behaves the same at 50 µs rests on that linearity rather than
on a direct netem comparison at 50 µs. Confirming it directly would need bare-metal
Linux with a high-resolution timer, not a VM.

## What this changes

The README previously listed the shim's network fidelity as *argued rather than
demonstrated*. It is now demonstrated, with a stated conversion factor and a
stated resolution limit on the reference instrument.
