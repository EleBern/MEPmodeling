"""
test_optimizer_scalar.py
------------------------
Validates ga_run() on standard scalar benchmark functions with known global
minima.  Each test minimises f(p) over a bounded domain and checks that the
recovered minimum is within tolerance of the known solution.

Benchmark functions
-------------------
  TEST 1 – Sphere          f(p) = sum(p²)                    min = 0  at origin
  TEST 2 – Rosenbrock      f(p) = sum(100(p[i+1]-p[i]²)² + (1-p[i])²)
                                                              min = 0  at (1,...,1)
  TEST 3 – Rastrigin       f(p) = An + sum(p²-A·cos(2π·p))  min = 0  at origin
  TEST 4 – Ackley          f(p) = -a·exp(-b·sqrt(mean(p²)))
                                  - exp(mean(cos(c·p))) + a + e
                                                              min = 0  at origin

All functions are parameterised as objective_function(p, ref) → (error, value)
where error is a 1-element array [f(p)] so ga_evaluation computes fit = f(p)².

Usage
-----
    python test_optimizer_scalar.py                  # all tests
    python test_optimizer_scalar.py --test 2         # only Rosenbrock
    python test_optimizer_scalar.py --N1 40 --tg 30
    python test_optimizer_scalar.py --ndim 5         # 5-D problems (tests 1-3)
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

import unittest.mock as mock
sys.modules.setdefault('Utils', mock.MagicMock())

from Optimizer import ga_run


# =============================================================================
# Shared helpers
# =============================================================================

def make_ref(boundary):
    """
    Minimal ref dict for scalar optimisation.
    y0 is a dummy 1-element array; it is only used by fitness_function()
    inside ga_run for the printed R² — it has no effect on the optimisation.
    """
    return {
        'boundary': np.asarray(boundary, dtype=float),
        'y0':       np.array([0.0]),
    }


def wrap_scalar(fn):
    """
    Wrap a plain scalar function f(p) -> float into the objective interface:
        objective(p, ref) -> (np.array([f(p)]), f(p))
    """
    def objective(p, ref):
        val = float(fn(p))
        return np.array([val]), val
    return objective


def print_result(name, param_names, true_min_params, true_min_val,
                 recovered_params, recovered_val, passed, tol):
    print()
    print('=' * 64)
    print(f'  RESULT: {name}')
    print('=' * 64)
    print(f'  {"Parameter":<14}  {"True minimum":>13}  {"Recovered":>13}  {"Δ":>10}')
    print('  ' + '-' * 56)
    for name_p, true, rec in zip(param_names, true_min_params, recovered_params):
        print(f'  {name_p:<14}  {true:>13.6f}  {rec:>13.6f}  {abs(rec-true):>10.2e}')
    print('  ' + '-' * 56)
    print(f'  f(p*)  true    = {true_min_val:.6e}')
    print(f'  f(p*)  found   = {recovered_val:.6e}')
    print(f'  |f_found - f*| = {abs(recovered_val - true_min_val):.2e}   '
          f'(tol = {tol:.1e})   {"PASS ✓" if passed else "FAIL ✗"}')
    print('=' * 64)
    print()


def save_loss_plot(KS_arr, filename, title):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(KS_arr, lw=1.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best fit  (= f(p)²)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f'  Loss curve → {filename}')


def save_surface_plot(fn, bounds_2d, p_best, filename, title):
    """2-D contour of the function with the found minimum marked."""
    lo, hi = bounds_2d[0]
    g = np.linspace(lo, hi, 200)
    X, Y = np.meshgrid(g, g)
    Z = np.array([[fn(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    fig, ax = plt.subplots(figsize=(6, 5))
    cp = ax.contourf(X, Y, Z, levels=40, cmap='viridis')
    plt.colorbar(cp, ax=ax)
    ax.scatter(*p_best[:2], color='red', s=80, zorder=5,
               label=f'GA min ({p_best[0]:.3f}, {p_best[1]:.3f})')
    ax.set_xlabel('p[0]')
    ax.set_ylabel('p[1]')
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f'  Surface plot → {filename}')


# =============================================================================
# TEST 1 – Sphere   f(p) = sum(p²)
# =============================================================================

def sphere(p):
    return np.sum(p ** 2)


def run_test_1(ndim, N1, tg, tol):
    name = f'Sphere ({ndim}-D)'
    print('\n' + '#' * 64)
    print(f'  TEST 1 – {name}')
    print('#' * 64)

    bounds   = [[-5.12, 5.12]] * ndim
    ref      = make_ref(bounds)
    true_p   = np.zeros(ndim)
    true_val = 0.0

    p_post, _, KS_arr, _ = ga_run(
        ref, wrap_scalar(sphere),
        N1=N1, N2=N1, N3=N1, tg=tg, op=-1, single_run_tol=tol,
    )

    recovered_val = sphere(p_post)
    passed = abs(recovered_val - true_val) < tol * 10   # generous: tol is early-stop
    param_names = [f'p[{i}]' for i in range(ndim)]
    print_result(name, param_names, true_p, true_val, p_post, recovered_val, passed, tol)
    save_loss_plot(KS_arr, 'test1_sphere_loss.png', f'TEST 1 – {name}  loss')
    if ndim == 2:
        save_surface_plot(sphere, bounds, p_post, 'test1_sphere_surface.png',
                          f'TEST 1 – {name}')
    return passed, recovered_val


# =============================================================================
# TEST 2 – Rosenbrock   f(p) = sum(100(p[i+1]-p[i]²)² + (1-p[i])²)
# =============================================================================

def rosenbrock(p):
    p = np.asarray(p, dtype=float)
    return np.sum(100.0 * (p[1:] - p[:-1] ** 2) ** 2 + (1.0 - p[:-1]) ** 2)


def run_test_2(ndim, N1, tg, tol):
    name = f'Rosenbrock ({ndim}-D)'
    print('\n' + '#' * 64)
    print(f'  TEST 2 – {name}')
    print('#' * 64)

    bounds   = [[-2.048, 2.048]] * ndim
    ref      = make_ref(bounds)
    true_p   = np.ones(ndim)
    true_val = 0.0

    p_post, _, KS_arr, _ = ga_run(
        ref, wrap_scalar(rosenbrock),
        N1=N1, N2=N1, N3=N1, tg=tg, op=-1, single_run_tol=tol,
    )

    recovered_val = rosenbrock(p_post)
    passed = abs(recovered_val - true_val) < 0.1   # Rosenbrock is hard; use fixed tol
    param_names = [f'p[{i}]' for i in range(ndim)]
    print_result(name, param_names, true_p, true_val, p_post, recovered_val, passed, tol=0.1)
    save_loss_plot(KS_arr, 'test2_rosenbrock_loss.png', f'TEST 2 – {name}  loss')
    if ndim == 2:
        save_surface_plot(rosenbrock, bounds, p_post, 'test2_rosenbrock_surface.png',
                          f'TEST 2 – {name}')
    return passed, recovered_val


# =============================================================================
# TEST 3 – Rastrigin   f(p) = A·n + sum(p² - A·cos(2π·p))
# =============================================================================

def rastrigin(p, A=10.0):
    p = np.asarray(p, dtype=float)
    return A * len(p) + np.sum(p ** 2 - A * np.cos(2.0 * np.pi * p))


def run_test_3(ndim, N1, tg, tol):
    name = f'Rastrigin ({ndim}-D)'
    print('\n' + '#' * 64)
    print(f'  TEST 3 – {name}')
    print('#' * 64)

    bounds   = [[-5.12, 5.12]] * ndim
    ref      = make_ref(bounds)
    true_p   = np.zeros(ndim)
    true_val = 0.0

    p_post, _, KS_arr, _ = ga_run(
        ref, wrap_scalar(rastrigin),
        N1=N1, N2=N1, N3=N1, tg=tg, op=-1, single_run_tol=tol,
    )

    recovered_val = rastrigin(p_post)
    passed = abs(recovered_val - true_val) < 1.0   # highly multi-modal; accept near-zero
    param_names = [f'p[{i}]' for i in range(ndim)]
    print_result(name, param_names, true_p, true_val, p_post, recovered_val, passed, tol=1.0)
    save_loss_plot(KS_arr, 'test3_rastrigin_loss.png', f'TEST 3 – {name}  loss')
    if ndim == 2:
        save_surface_plot(rastrigin, bounds, p_post, 'test3_rastrigin_surface.png',
                          f'TEST 3 – {name}')
    return passed, recovered_val


# =============================================================================
# TEST 4 – Ackley (fixed 2-D)
# =============================================================================

def ackley(p, a=20.0, b=0.2, c=2.0 * np.pi):
    p = np.asarray(p, dtype=float)
    n = len(p)
    return (-a * np.exp(-b * np.sqrt(np.sum(p**2) / n))
            - np.exp(np.sum(np.cos(c * p)) / n)
            + a + np.e)


def run_test_4(N1, tg, tol):
    ndim = 2
    name = f'Ackley ({ndim}-D)'
    print('\n' + '#' * 64)
    print(f'  TEST 4 – {name}')
    print('#' * 64)

    bounds   = [[-5.0, 5.0]] * ndim
    ref      = make_ref(bounds)
    true_p   = np.zeros(ndim)
    true_val = 0.0

    p_post, _, KS_arr, _ = ga_run(
        ref, wrap_scalar(ackley),
        N1=N1, N2=N1, N3=N1, tg=tg, op=-1, single_run_tol=tol,
    )

    recovered_val = ackley(p_post)
    passed = abs(recovered_val - true_val) < 0.5
    param_names = [f'p[{i}]' for i in range(ndim)]
    print_result(name, param_names, true_p, true_val, p_post, recovered_val, passed, tol=0.5)
    save_loss_plot(KS_arr, 'test4_ackley_loss.png', f'TEST 4 – {name}  loss')
    save_surface_plot(ackley, bounds, p_post, 'test4_ackley_surface.png',
                      f'TEST 4 – {name}')
    return passed, recovered_val


# =============================================================================
# Summary
# =============================================================================

def print_summary(results):
    print()
    print('=' * 64)
    print('  SUMMARY')
    print('=' * 64)
    print(f'  {"Test":<35}  {"f(p*) found":>12}  {"Pass":>6}')
    print('  ' + '-' * 58)
    all_pass = True
    for label, passed, val in results:
        all_pass = all_pass and passed
        mark = '✓' if passed else '✗'
        print(f'  {label:<35}  {val:>12.4e}  {mark:>6}')
    print('  ' + '-' * 58)
    print(f'  Overall: {"PASS ✓" if all_pass else "FAIL ✗"}')
    print('=' * 64)
    print()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Test ga_run on standard scalar benchmark functions'
    )
    p.add_argument('--test',  type=int,   default=0,
                   help='Run only this test (1/2/3/4). 0 = all (default).')
    p.add_argument('--N1',   type=int,   default=40,
                   help='Population size (default 40).')
    p.add_argument('--tg',   type=int,   default=30,
                   help='Max generations (default 30).')
    p.add_argument('--ndim', type=int,   default=2,
                   help='Problem dimensionality for tests 1-3 (default 2).')
    p.add_argument('--tol',  type=float, default=1e-6,
                   help='Early-stop tolerance passed to ga_run (default 1e-6).')
    return p.parse_args()


def main():
    args = parse_args()
    ndim = max(2, args.ndim)

    print()
    print('╔════════════════════════════════════════════════════════════╗')
    print('║        ga_run  –  Scalar Benchmark Test Suite              ║')
    print('╠════════════════════════════════════════════════════════════╣')
    print(f'║  Population N1 = {args.N1:<4}   Generations tg = {args.tg:<4}            ║')
    print(f'║  Problem ndim  = {ndim:<4}   Early-stop tol = {args.tol:<10.1e}      ║')
    print('╚════════════════════════════════════════════════════════════╝')

    runners = {
        1: lambda: run_test_1(ndim, args.N1, args.tg, args.tol),
        2: lambda: run_test_2(ndim, args.N1, args.tg, args.tol),
        3: lambda: run_test_3(ndim, args.N1, args.tg, args.tol),
        4: lambda: run_test_4(args.N1, args.tg, args.tol),
    }
    labels = {
        1: f'TEST 1 – Sphere ({ndim}-D)',
        2: f'TEST 2 – Rosenbrock ({ndim}-D)',
        3: f'TEST 3 – Rastrigin ({ndim}-D)',
        4:  'TEST 4 – Ackley (2-D)',
    }

    to_run = [args.test] if args.test in runners else [1, 2, 3, 4]

    results = []
    for idx in to_run:
        passed, val = runners[idx]()
        results.append((labels[idx], passed, val))

    if len(results) > 1:
        print_summary(results)


if __name__ == '__main__':
    main()