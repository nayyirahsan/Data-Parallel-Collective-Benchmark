"""Figure generation.

    python -m dpt.analysis.figures

Every figure is generated from the committed CSVs in results/, so the plots and
the numbers in docs/ cannot drift apart.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..simulate import SimConfig, degradation, simulate  # noqa: E402
from .crossover import analyse  # noqa: E402
from .fit import fit_floor, step_bytes, steps  # noqa: E402
from .noise import TAIL_THRESHOLD, profile  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("figures")
ALGO_COLOURS = {"ring": "#1f77b4", "ps": "#d62728", "recursive": "#2ca02c"}
ALGO_LABELS = {"ring": "Ring", "ps": "Parameter server", "recursive": "Recursive h/d"}


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_floor() -> None:
    """Per-step cost vs per-step transfer size, one line per worker count."""
    path = RESULTS / "floor_critpath.csv"
    if not path.exists():
        return
    rows = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            p, n, algo = int(row["world_size"]), int(row["nbytes"]), row["algo"]
            rows[(p, algo, n)].append(float(row["duration_ns"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fits = fit_floor(path)
    for p in sorted({k[0] for k in rows}):
        xs, ys = [], []
        for (pp, algo, n), vals in rows.items():
            if pp != p:
                continue
            xs.append(step_bytes(algo, n, p))
            ys.append(float(np.median(vals)) / steps(algo, p) / 1000)
        order = np.argsort(xs)
        ax.scatter(np.array(xs)[order], np.array(ys)[order], s=22, alpha=0.75, label=f"p={p}")
        fit = fits[p]
        grid = np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)
        ax.plot(grid, (fit.a_sw_ns + grid * fit.b_sw_ns_per_byte) / 1000, lw=1.2, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bytes per step")
    ax.set_ylabel("cost per step (µs)")
    ax.set_title("Software overhead floor (α = β = 0)\nfixed cost and bandwidth both degrade with worker count")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save(fig, "floor.png")


def figure_h1_model_error() -> None:
    """Textbook model error as a function of injected latency over the floor."""
    critpath, floor_path = RESULTS / "h1_grid_critpath.csv", RESULTS / "floor_critpath.csv"
    if not (critpath.exists() and floor_path.exists()):
        return
    floors = fit_floor(floor_path)
    results = analyse(critpath, floors)

    pts = [
        (r.alpha_ns / floors[r.world_size].a_sw_ns, r.textbook_ratio, r.corrected_ratio, r.world_size)
        for r in results
        if r.textbook_ratio and not r.extrapolated
    ]
    if not pts:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, idx, colour, marker in (
        ("textbook α-β model", 1, "#d62728", "o"),
        ("corrected (fitted floor)", 2, "#2ca02c", "s"),
    ):
        xs = [p[0] for p in pts if p[idx]]
        ys = [p[idx] for p in pts if p[idx]]
        ax.scatter(xs, ys, c=colour, marker=marker, s=55, label=label, zorder=3)

    grid = np.logspace(np.log10(0.15), np.log10(10), 100)
    ax.plot(grid, 1 + 1 / grid, "k--", lw=1, alpha=0.6, label=r"$1+\alpha_{sw}/\alpha$")
    ax.axhline(1.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha\,/\,\alpha_{sw}$  (injected latency over software floor)")
    ax.set_ylabel("measured N* / predicted N*")
    ax.set_title("H1: the α-β model's error is governed by α/α_sw\nexact once network latency dominates software overhead")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    _save(fig, "h1_model_error.png")


def figure_crossover_surface() -> None:
    """Measured vs predicted crossover across worker count and latency."""
    critpath, floor_path = RESULTS / "h1_grid_critpath.csv", RESULTS / "floor_critpath.csv"
    if not (critpath.exists() and floor_path.exists()):
        return
    floors = fit_floor(floor_path)
    results = [r for r in analyse(critpath, floors) if r.beta_ns_per_byte > 0]

    by_p = defaultdict(list)
    for r in results:
        if r.measured_ns:
            by_p[r.world_size].append(r)

    fig, axes = plt.subplots(1, len(by_p), figsize=(4.2 * len(by_p), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, sorted(by_p)):
        rs = sorted(by_p[p], key=lambda r: r.alpha_ns)
        alphas = [r.alpha_ns / 1000 for r in rs]
        measured = [r.measured_ns / 1024 for r in rs]
        lo = [(r.ci_low or np.nan) / 1024 for r in rs]
        hi = [(r.ci_high or np.nan) / 1024 for r in rs]
        ax.fill_between(alphas, lo, hi, alpha=0.2, color="#1f77b4", label="95% CI")
        ax.plot(alphas, measured, "o-", color="#1f77b4", label="measured")
        tb = [(r.textbook_ns / 1024 if r.textbook_ns else np.nan) for r in rs]
        cr = [(r.corrected_ns / 1024 if r.corrected_ns else np.nan) for r in rs]
        ax.plot(alphas, tb, "s--", color="#d62728", label="textbook")
        ax.plot(alphas, cr, "^--", color="#2ca02c", label="corrected")
        ax.set_xlabel("injected α (µs)")
        ax.set_title(f"p = {p}")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("crossover N*  (KB)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Ring overtakes parameter server: measured vs predicted crossover (β = 0.4 ns/byte)")
    _save(fig, "crossover_surface.png")


def figure_noise_floor() -> None:
    """Where the apparatus can and cannot support a tail statistic."""
    path = RESULTS / "noise_floor_critpath.csv"
    if not path.exists():
        return
    profiles = profile(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in sorted({p.algo for p in profiles}):
        for beta, style in ((0.0, "o-"), (0.4, "s--")):
            subset = sorted(
                [p for p in profiles if p.algo == algo and p.beta == beta],
                key=lambda p: (p.world_size, p.nbytes),
            )
            if not subset:
                continue
            by_p = defaultdict(list)
            for pr in subset:
                by_p[pr.world_size].append(pr.tail_ratio)
            ps = sorted(by_p)
            ax.plot(
                ps, [np.mean(by_p[p]) for p in ps], style,
                color=ALGO_COLOURS[algo], alpha=0.85,
                label=f"{ALGO_LABELS[algo]}, β={beta:g}",
            )
    ax.axhline(TAIL_THRESHOLD, color="k", ls=":", lw=1.2)
    ax.text(2.1, TAIL_THRESHOLD * 1.05, "usable for tail statistics", fontsize=8)
    ax.set_xlabel("workers (p)")
    ax.set_ylabel("clean p99 / p50")
    ax.set_xticks([2, 4, 8])
    ax.set_title("Apparatus tail-noise floor\nonly p=2 supports a p99 claim on this machine")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, "noise_floor.png")


def figure_h2_simulated() -> None:
    """Simulated degradation and absolute p99 against worker count."""
    ps = [2, 4, 8, 16, 32, 64]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    for algo in ("ring", "ps", "recursive"):
        ratios, absolute = [], []
        for p in ps:
            cfg = SimConfig(algo, p, 1048576, 200_000, 0.4, 0.01, 10.0)
            ratios.append(degradation(cfg, 200_000)["p99_ratio"])
            absolute.append(np.percentile(simulate(cfg, 200_000), 99) / 1e6)
        ax1.plot(ps, ratios, "o-", color=ALGO_COLOURS[algo], label=ALGO_LABELS[algo])
        ax2.plot(ps, absolute, "o-", color=ALGO_COLOURS[algo], label=ALGO_LABELS[algo])

    ax1.axhline(1.0, color="k", lw=0.8, alpha=0.4)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ps)
    ax1.set_xticklabels(ps)
    ax1.set_xlabel("workers (p)")
    ax1.set_ylabel("p99 degradation vs clean")
    ax1.set_title("H2 refuted: degradation falls with p\n(ring is not more fragile)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(ps)
    ax2.set_xticklabels(ps)
    ax2.set_xlabel("workers (p)")
    ax2.set_ylabel("absolute p99 under jitter (ms)")
    ax2.set_title("What actually decides the winner\nrecursive h/d dominates in absolute terms")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig.suptitle("Simulated, 1 MB, α=200 µs, β=0.4, jitter q=1% k=10 — model validated only at p=2", fontsize=9)
    _save(fig, "h2_simulated.png")


def figure_e2e_breakdown() -> None:
    """Where a training step's time goes, before and after the staging fix."""
    # Measured with `python -m dpt.train.compare`; the before column is the
    # per-tensor device transfer, the after column one fused transfer.
    before = {"compute": 168.7, "all-reduce": 73.6, "pack + transfer": 107.3}
    after = {"compute": 157.9, "all-reduce": 75.4, "pack + transfer": 16.7}
    phases = list(before)
    colours = ["#9467bd", "#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (label, data) in enumerate((("before", before), ("after", after))):
        left = 0.0
        for phase, colour in zip(phases, colours):
            ax.barh(i, data[phase], left=left, color=colour, height=0.55,
                    label=phase if i == 0 else None)
            if data[phase] > 12:
                ax.text(left + data[phase] / 2, i, f"{data[phase]:.0f}",
                        ha="center", va="center", fontsize=8, color="white")
            left += data[phase]
        ax.text(left + 4, i, f"{left:.0f} ms", va="center", fontsize=9)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["per-tensor\ntransfer", "one fused\ntransfer"])
    ax.set_xlabel("time per training step (ms)")
    ax.set_xlim(0, 400)
    ax.set_title("ResNet-18 step attribution, ring all-reduce at p=4\n"
                 "staging cost exceeded the collective until it was fused")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              frameon=False)
    ax.grid(alpha=0.3, axis="x")
    _save(fig, "e2e_breakdown.png")


def figure_bandwidth_curve() -> None:
    """Effective per-byte cost against transfer size."""
    path = RESULTS / "bandwidth_curve_critpath.csv"
    if not path.exists():
        return
    vals = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            vals[int(row["nbytes"])].append(float(row["duration_ns"]))

    sizes = sorted(vals)
    xs = [n / 2 for n in sizes]
    ys = [float(np.median(vals[n])) / 2 / (n / 2) for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bytes per transfer")
    ax.set_ylabel("cost per byte (ns)")
    ax.set_title("Effective per-byte cost plateaus above ~2 MB\n"
                 "no turnaround: refutes the cache explanation for recursive's slowness")
    ax.grid(alpha=0.3, which="both")
    _save(fig, "bandwidth_curve.png")


def main() -> None:
    print("generating figures:")
    figure_e2e_breakdown()
    figure_bandwidth_curve()
    figure_floor()
    figure_h1_model_error()
    figure_crossover_surface()
    figure_noise_floor()
    figure_h2_simulated()


if __name__ == "__main__":
    main()
