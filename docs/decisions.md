# Design decisions to defend

Running log of non-obvious choices, each with the alternative rejected and why.

### 1. Spin-wait, not sleep, for delay injection
`time.sleep` overshoots by a consistent +20–30% at every scale on macOS; the
hybrid strategy originally planned inherits that bias above 1 ms (+17.7% at
5 ms). Pure spin holds ≤0.4% median error. Cost: a consumed core per worker,
which bounds the experiment at p ≤ 8. Full data in `calibration.md`.

### 2. Point-to-point is borrowed; collectives are not
`isend`/`irecv`/`barrier` come from `torch.distributed`'s gloo backend. Every
collective algorithm is implemented from scratch on top. Reimplementing a wire
protocol would have cost weeks and measured nothing either hypothesis asks
about. The line is drawn at the algorithm boundary, which is where the claims
live.

### 3. Jitter keys on (worker, step), not message index
H2 requires replaying an identical jitter realisation against each algorithm.
Message-index keying cannot do this — ring sends 2(p−1) messages per collective
and PS sends 2, so message *k* is not comparable across arms. Keying on
"rank *r* is transiently slow during step *s*" is algorithm-independent,
physically meaningful, and makes the comparison properly paired. Draws are a
pure function of (seed, rank, step), so no trace needs storing.

### 4. Link cost is charged at send-initiate
This models a store-and-forward link where the sender pays serialisation plus
propagation. Delaying receiver completion instead would model cut-through, and
would let a sender race ahead issuing messages it had not paid for — decoupling
injected cost from the algorithm's dependency structure. Since H2 *is* a claim
about dependency structure, the delay must sit where the algorithm actually
serialises. Consequence to state plainly: a broadcast to p receivers costs p
times, as on a real shared uplink — which is what makes the PS bottleneck
appear. Validated against `tc netem` in the Colab notebook.

### 5. Zero-cost shaping is the same code path
`ShapedTransport` with α=β=0 is used for the software-overhead floor run that H1
depends on, rather than bypassing the wrapper. This guarantees the floor
measurement includes exactly the same Python dispatch and buffer overheads as
the shaped runs, so subtracting it is meaningful.

### 6. Recursive halving/doubling refuses non-power-of-two
Supporting arbitrary p needs an extra pre/post reduction phase that changes the
step count — precisely the quantity under comparison. Refusing is honest;
approximating would silently corrupt the comparison.

### 7. A/B comparisons run interleaved inside one sweep, never across runs
Comparing two implementations by running one sweep, changing the code, and
running another attributes machine drift to the change. Measured concretely:
a recursive-collective variant looked 10–30% slower that way, but an
interleaved, trial-paired A/B in a single sweep found **no difference at any
configuration**. Variant selection is therefore a parameter inside the config
grid (see `recursive_fresh` in `dpt.collectives`), so both arms share cold
launches and shuffled ordering.

### 8. Hypotheses about my own code get tested, not assumed
Two explanations for recursive halving/doubling's unexpected slowness —
per-step allocation, and cache behaviour at large transfers — were both
plausible and both refuted by direct measurement (see
`findings-workload.md`). The residual gap is recorded as an open question
rather than attributed to an untested third mechanism.

### 9. Shim α relates to `tc netem delay` by a measured factor of 2.08
Validated on Linux (`docs/netem-validation.md`): the shim is linear in
configured α with slope 1.004 and R² = 1.0000, and netem is linear with slope
2.090. The ~2× gap appears independently in bandwidth (netem's effective β is
2.05–2.13× nominal), which identifies it as double traversal of the loopback
egress qdisc rather than an error in either instrument. **1 µs of shim α ≡
2.08 µs of configured netem delay.** Findings are unaffected; only the mapping
to a physical link changes.

The verdict deliberately tests *linearity of both curves*, not agreement on a
particular factor. An earlier version hard-coded an expected ratio of 1 or 2 and
would have reported a spurious failure — the slope ratio is a unit conversion,
whereas linearity is what determines whether the α-β abstraction holds at all.

### 10. Noise-floor measurements require a verified-idle machine
A p=3 configuration measured 7.23× tail ratio with Docker Desktop running (a
12-CPU VM on a 12-core host) and 1.44× with it stopped — a 5× difference from
background load alone. Every tail measurement now checks `uptime` first and
waits for load average below 2. This was found by noticing that a
previously-measured p=2 point had drifted from 1.49× to 3.10× between runs;
without that cross-check the contaminated numbers would have been written up as
a finding.

### 11. Spin-wait is required on Linux too, for a different reason
On macOS `time.sleep` overshoots 20–30% proportionally. In a Docker Linux VM it
quantizes to ~1 ms: a 50 µs sleep takes 999 µs (+1898%). That is the same ~1 ms
timer quantum that makes `tc netem` unusable below 1 ms here
(`netem-validation.md`) — one root cause, two symptoms. Decision #1 holds on
both platforms.

### 12. Delay injection yields its core: fractional sleep, spin the tail
Pure spinning is accurate but consumes a core for every injected delay, which is
what capped the experiment at p ≤ 8 and made every p ≥ 4 tail measurement
unusable. Decision #1 rejected sleeping on evidence from a *fixed* 300 µs slack,
which fails at large delays — but the overshoot is **proportional** (~1.25–1.35×),
not absolute, so a *fractional* sleep is safe at every scale: 0.7 × 1.35 < 1.

Measured: sleeping 70% of each delay then spinning to the deadline holds median
error at 0.03–0.33%, indistinguishable from pure spinning, at **11–15% CPU
instead of 99%**. At 0.8 the sleep leg overruns the deadline (+9.2% at 50 µs),
so 0.7 is the usable maximum. Delays under 20 µs skip the sleep entirely.

Effect: usable configurations for tail statistics went from **5/36 to 19/36**,
and p=4 at β=0.4 now passes the gate — which turned H2 from a model prediction
into a measurement. `sleep_fraction=0` on `ShapedTransport` restores pure
spinning.

The general lesson: decision #1 was right about `time.sleep` being unusable and
wrong about the remedy. Re-testing a settled decision when its *consequence*
became the binding constraint was worth more than any tuning elsewhere.
