"""
Memory and timing test for the MEP model evaluated by pygpc (pygpc 0.4.4).

Calls run_model() repeatedly with random parameter samples drawn from the same
distributions as main.py, and reports for every call:
  - wall time of the call
  - current memory (RSS) after the call, and after garbage collection
  - peak memory so far

At the end it estimates whether memory leaks between calls and how much
walltime / memory to request for the full pygpc run.

Run it in an interactive job from the MEPmodeling directory, e.g.:
    qsub -I -l select=1:ncpus=1:mem=8gb:scratch_local=2gb -l walltime=1:00:00
    (load python module and set PYTHONUSERBASE/PYTHONPATH as in qsub_main.sh)
    python test_model_memory.py --n-calls 20
"""

import argparse
import gc
import resource
import time

import numpy as np

from pygpc_main import run_model

SUBJ = 2  # same subject as in muap_gpc_model.py


def rss_mb():
    """Current resident memory of this process in MB (Linux only)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def peak_mb():
    """Peak resident memory of this process so far in MB (ru_maxrss is in KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def sample_parameters(rng):
    """Draw one parameter set from the distributions defined in main.py."""
    u = rng.uniform
    return dict(
        b=u(45, 425),
        lam=rng.normal(3.619, 0.774),
        R1=u(0, 10), R2=u(0, 10), R3=u(0, 10), R4=u(0, 10), R5=u(0, 10),
        E1=u(0, 20), E2=u(0, 20),
        I1=u(0, 10), I2=u(0, 10),
        RC_th=u(1, 10),
        MU_T=u(5, 10),
        AMPAw=u(0, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-calls", type=int, default=20,
                        help="number of model evaluations to test")
    parser.add_argument("--seed", type=int, default=1)
    # planned pygpc run, used for the walltime/memory estimate
    parser.add_argument("--n-grid", type=int, default=3000,
                        help="training samples (n_grid)")
    parser.add_argument("--n-val-plot", type=int, default=51 * 51,
                        help="original-model runs in validate_gpc_plot")
    parser.add_argument("--n-mc", type=int, default=1000,
                        help="original-model runs in validate_gpc_mc")
    parser.add_argument("--n-cpu", type=int, default=1,
                        help="parallel pygpc workers planned (n_cpu)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gc.collect()
    rss_start = rss_mb()
    print(f"Memory before first call: {rss_start:.0f} MB\n")
    print(f"{'call':>4} {'time [s]':>9} {'RSS [MB]':>9} {'RSS after gc':>13} "
          f"{'peak [MB]':>10} {'R2':>10}")

    times, rss_after, rss_gc = [], [], []
    for i in range(args.n_calls):
        p = sample_parameters(rng)

        t0 = time.perf_counter()
        r2 = run_model(SUBJ, **p)
        dt = time.perf_counter() - t0

        rss_call = rss_mb()
        gc.collect()
        rss_collected = rss_mb()

        times.append(dt)
        rss_after.append(rss_call)
        rss_gc.append(rss_collected)

        flag = "" if np.isfinite(r2) else "  <-- not finite!"
        print(f"{i:>4} {dt:>9.2f} {rss_call:>9.0f} {rss_collected:>13.0f} "
              f"{peak_mb():>10.0f} {r2:>10.4f}{flag}")

    # ---------------- summary ----------------
    times = np.array(times)
    rss_gc = np.array(rss_gc)
    print("\n================ Summary ================")
    print(f"First call (incl. warm-up/loading): {times[0]:.2f} s")
    steady = times[1:] if len(times) > 1 else times
    print(f"Later calls: median {np.median(steady):.2f} s, "
          f"mean {steady.mean():.2f} s, max {steady.max():.2f} s")
    print(f"Peak memory: {peak_mb():.0f} MB")

    # memory growth per call (after gc), ignoring the first call
    if len(rss_gc) >= 4:
        calls = np.arange(1, len(rss_gc))
        growth = np.polyfit(calls, rss_gc[1:], 1)[0]
    else:
        growth = 0.0
    print(f"Memory growth per call (after gc): {growth:.2f} MB/call")

    n_total = args.n_grid + args.n_val_plot + args.n_mc
    t_total_h = n_total * np.median(steady) / max(args.n_cpu, 1) / 3600
    print(f"\nPlanned original-model runs: {n_total} "
          f"(training {args.n_grid} + plot validation {args.n_val_plot} "
          f"+ MC validation {args.n_mc})")
    print(f"Estimated model time with n_cpu={args.n_cpu}: {t_total_h:.1f} h")
    print(f"Suggested walltime (+50% margin): {np.ceil(t_total_h * 1.5):.0f} h")

    # memory estimate: steady per-process peak, plus leak over the run
    per_proc = peak_mb()
    leak_total = max(growth, 0) * n_total / max(args.n_cpu, 1)
    mem_gb = (per_proc + leak_total) * args.n_cpu / 1024 + 1.5
    if growth > 1.0:
        print(f"\nWARNING: memory grows by ~{growth:.1f} MB per call. Over the full run "
              f"each process would reach ~{(per_proc + leak_total) / 1024:.1f} GB. "
              f"This is a leak in the model code and should be fixed rather than "
              f"covered by requesting more memory.")
    print(f"Suggested mem= (incl. ~1.5 GB overhead): {np.ceil(mem_gb):.0f}gb")


if __name__ == "__main__":
    main()
