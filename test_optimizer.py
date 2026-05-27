"""
test_optimizer.py
-----------------
Validates ga_run() against three time-series benchmarks with known ground-truth
parameters.  Each test:
  1. Generates a noise-free target signal with fixed TRUE_PARAMS.
  2. Adds a small amount of Gaussian noise so the problem is realistic.
  3. Runs ga_run() with tight parameter bounds centred on the ground truth.
  4. Reports recovered parameters, % error, and a fit metric (R²).

Benchmark models
----------------
  TEST 1 – Damped sinusoid         y = A · exp(-d·t) · sin(2π·f·t + φ)
  TEST 2 – Sum of two sinusoids    y = A1·sin(2π·f1·t) + A2·sin(2π·f2·t + φ)
  TEST 3 – Logistic growth         y = K / (1 + ((K-y0)/y0) · exp(-r·t))

Usage
-----
    python test_optimizer.py [--test 1] [--N1 30] [--tg 20] [--noise 0.02]

The script can be run from the directory containing Optimizer.py and ga_helpers.py.
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless – swap to 'TkAgg' for interactive use
import matplotlib.pyplot as plt

# ── import the optimizer ──────────────────────────────────────────────────────
sys.path.insert(0, '.')        # ensure local Optimizer.py / ga_helpers.py are found
from Optimizer import ga_run, ga_plot_fit


# =============================================================================
# Shared helpers
# =============================================================================

def r_squared(y_true, y_pred):
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def make_ref(y0, boundary):
    """
    Build the minimal ref dict expected by ga_run.

    Parameters
    ----------
    y0       : np.ndarray  target time series  [nData,]
    boundary : np.ndarray  [nParams x 2]  [[lo, hi], ...]

    Returns
    -------
    ref : dict
    """
    return {
        'y0':       y0,
        'boundary': np.asarray(boundary, dtype=float),
    }


def print_result(test_name, param_names, true_params, recovered_params, r2):
    """Pretty-print a test result."""
    print()
    print('=' * 62)
    print(f'  RESULT: {test_name}')
    print('=' * 62)
    print(f'  {"Parameter":<18}  {"True":>10}  {"Recovered":>10}  {"Error %":>9}')
    print('  ' + '-' * 56)
    for name, true, rec in zip(param_names, true_params, recovered_params):
        pct = abs(rec - true) / (abs(true) + 1e-12) * 100
        print(f'  {name:<18}  {true:>10.4f}  {rec:>10.4f}  {pct:>8.2f}%')
    print('  ' + '-' * 56)
    print(f'  R²  = {r2:.6f}')
    print('=' * 62)
    print()


def save_plot(t, y_true, y_noisy, y_pred, title, filename):
    """Save a 2-panel comparison figure."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, y_noisy, 'k.', ms=2, alpha=0.4, label='noisy target')
    axes[0].plot(t, y_true,  'b-', lw=1.5, label='true signal')
    axes[0].plot(t, y_pred,  'r--', lw=1.5, label='GA fit')
    axes[0].set_ylabel('Signal')
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)

    residuals = y_noisy - y_pred
    axes[1].plot(t, residuals, 'g-', lw=1)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_ylabel('Residual')
    axes[1].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f'  Plot saved → {filename}')


# =============================================================================
# TEST 1 – Damped sinusoid
# =============================================================================
#   y(t) = A · exp(-d·t) · sin(2π·f·t + φ)
#   Parameters: A (amplitude), d (decay), f (frequency), φ (phase)

TRUE_PARAMS_1   = [2.5, 0.3, 1.2, 0.8]        # A, d, f, phi
PARAM_NAMES_1   = ['A (amplitude)', 'd (decay)', 'f (frequency)', 'phi (phase)']
# Bounds: ±50 % around the true value (generous but bounded)
BOUNDS_1 = [
    [1.0,  4.0],    # A
    [0.05, 0.8],    # d
    [0.5,  2.5],    # f
    [0.0,  1.8],    # phi
]


def model_damped_sinusoid(p, t):
    A, d, f, phi = p
    return A * np.exp(-d * t) * np.sin(2 * np.pi * f * t + phi)


def objective_damped_sinusoid(p, ref):
    """
    Returns
    -------
    error   : np.ndarray  absolute residuals |y_noisy - y_pred|
    y_pred  : np.ndarray  model output
    """
    y_pred = model_damped_sinusoid(p, ref['t'])
    error  = np.abs(ref['y0'] - y_pred)
    return error, y_pred


def run_test_1(N1, tg, noise_std, single_run_tol):
    print('\n' + '#' * 62)
    print('  TEST 1 – Damped sinusoid')
    print('#' * 62)

    t      = np.linspace(0, 5, 300)
    y_true = model_damped_sinusoid(TRUE_PARAMS_1, t)
    rng    = np.random.default_rng(0)
    y_noisy = y_true + rng.normal(0, noise_std * np.std(y_true), size=t.shape)

    ref = make_ref(y_noisy, BOUNDS_1)
    ref['t'] = t

    p_post, KP_arr, KS_arr, _ = ga_run(
        ref, objective_damped_sinusoid,
        N1=N1, N2=max(20, N1), N3=max(20, N1),
        tg=tg, op=-1, single_run_tol=single_run_tol,
    )

    y_pred = model_damped_sinusoid(p_post, t)
    r2 = r_squared(y_noisy, y_pred)
    print_result('Damped sinusoid', PARAM_NAMES_1, TRUE_PARAMS_1, p_post, r2)
    save_plot(t, y_true, y_noisy, y_pred,
              'TEST 1 – Damped sinusoid', 'test1_damped_sinusoid.png')

    ga_plot_fit_to_file(KS_arr, 'test1_loss.png', 'TEST 1 – loss curve')
    return r2, p_post


# =============================================================================
# TEST 2 – Sum of two sinusoids
# =============================================================================
#   y(t) = A1·sin(2π·f1·t) + A2·sin(2π·f2·t + φ)
#   Parameters: A1, f1, A2, f2, phi

TRUE_PARAMS_2  = [1.5, 0.8, 0.7, 2.5, 1.1]    # A1, f1, A2, f2, phi
PARAM_NAMES_2  = ['A1', 'f1', 'A2', 'f2', 'phi']
BOUNDS_2 = [
    [0.3, 3.0],    # A1
    [0.2, 2.0],    # f1
    [0.1, 2.0],    # A2
    [1.0, 4.0],    # f2
    [0.0, 2.5],    # phi
]


def model_two_sinusoids(p, t):
    A1, f1, A2, f2, phi = p
    return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t + phi)


def objective_two_sinusoids(p, ref):
    y_pred = model_two_sinusoids(p, ref['t'])
    error  = np.abs(ref['y0'] - y_pred)
    return error, y_pred


def run_test_2(N1, tg, noise_std, single_run_tol):
    print('\n' + '#' * 62)
    print('  TEST 2 – Sum of two sinusoids')
    print('#' * 62)

    t      = np.linspace(0, 4, 400)
    y_true = model_two_sinusoids(TRUE_PARAMS_2, t)
    rng    = np.random.default_rng(1)
    y_noisy = y_true + rng.normal(0, noise_std * np.std(y_true), size=t.shape)

    ref = make_ref(y_noisy, BOUNDS_2)
    ref['t'] = t

    p_post, KP_arr, KS_arr, _ = ga_run(
        ref, objective_two_sinusoids,
        N1=N1, N2=max(20, N1), N3=max(20, N1),
        tg=tg, op=-1, single_run_tol=single_run_tol,
    )

    y_pred = model_two_sinusoids(p_post, t)
    r2 = r_squared(y_noisy, y_pred)
    print_result('Sum of two sinusoids', PARAM_NAMES_2, TRUE_PARAMS_2, p_post, r2)
    save_plot(t, y_true, y_noisy, y_pred,
              'TEST 2 – Sum of two sinusoids', 'test2_two_sinusoids.png')

    ga_plot_fit_to_file(KS_arr, 'test2_loss.png', 'TEST 2 – loss curve')
    return r2, p_post


# =============================================================================
# TEST 3 – Logistic growth
# =============================================================================
#   y(t) = K / (1 + ((K - y0) / y0) · exp(-r·t))
#   Parameters: K (carrying capacity), y0 (initial value), r (growth rate)

TRUE_PARAMS_3  = [10.0, 0.5, 1.2]             # K, y0, r
PARAM_NAMES_3  = ['K (capacity)', 'y0 (initial)', 'r (growth rate)']
BOUNDS_3 = [
    [5.0, 20.0],   # K
    [0.1,  2.0],   # y0
    [0.3,  3.0],   # r
]


def model_logistic(p, t):
    K, y0, r = p
    return K / (1.0 + ((K - y0) / (y0 + 1e-12)) * np.exp(-r * t))


def objective_logistic(p, ref):
    y_pred = model_logistic(p, ref['t'])
    error  = np.abs(ref['y0'] - y_pred)
    return error, y_pred


def run_test_3(N1, tg, noise_std, single_run_tol):
    print('\n' + '#' * 62)
    print('  TEST 3 – Logistic growth')
    print('#' * 62)

    t      = np.linspace(0, 8, 200)
    y_true = model_logistic(TRUE_PARAMS_3, t)
    rng    = np.random.default_rng(2)
    y_noisy = y_true + rng.normal(0, noise_std * np.std(y_true), size=t.shape)

    ref = make_ref(y_noisy, BOUNDS_3)
    ref['t'] = t

    p_post, KP_arr, KS_arr, _ = ga_run(
        ref, objective_logistic,
        N1=N1, N2=max(20, N1), N3=max(20, N1),
        tg=tg, op=-1, single_run_tol=single_run_tol,
    )

    y_pred = model_logistic(p_post, t)
    r2 = r_squared(y_noisy, y_pred)
    print_result('Logistic growth', PARAM_NAMES_3, TRUE_PARAMS_3, p_post, r2)
    save_plot(t, y_true, y_noisy, y_pred,
              'TEST 3 – Logistic growth', 'test3_logistic.png')

    ga_plot_fit_to_file(KS_arr, 'test3_loss.png', 'TEST 3 – loss curve')
    return r2, p_post


# =============================================================================
# Utility: save loss curve to file (headless alternative to ga_plot_fit)
# =============================================================================

def ga_plot_fit_to_file(errors, filename, title='Loss curve'):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.asarray(errors))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best fit (SSE)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f'  Loss curve saved → {filename}')


# =============================================================================
# Summary table
# =============================================================================

def print_summary(results):
    print()
    print('=' * 62)
    print('  SUMMARY')
    print('=' * 62)
    print(f'  {"Test":<35}  {"R²":>8}  {"Pass":>6}')
    print('  ' + '-' * 54)
    all_pass = True
    for name, r2 in results:
        passed = r2 >= 0.95
        all_pass = all_pass and passed
        mark = '✓' if passed else '✗'
        print(f'  {name:<35}  {r2:>8.4f}  {mark:>6}')
    print('  ' + '-' * 54)
    print(f'  Overall: {"PASS ✓" if all_pass else "FAIL ✗"}'
          f'  (threshold R² ≥ 0.95)')
    print('=' * 62)
    print()


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Test ga_run optimizer on synthetic benchmarks')
    p.add_argument('--test',  type=int, default=0,
                   help='Run only this test (1/2/3).  0 = run all (default).')
    p.add_argument('--N1',   type=int,   default=30,
                   help='Population size (default 30).')
    p.add_argument('--tg',   type=int,   default=20,
                   help='Max generations (default 20).')
    p.add_argument('--noise', type=float, default=0.02,
                   help='Noise level as fraction of signal std (default 0.02).')
    p.add_argument('--tol',  type=float, default=1e-5,
                   help='Early-stop tolerance (default 1e-5).')
    return p.parse_args()


def main():
    args = parse_args()

    print()
    print('╔══════════════════════════════════════════════════════════╗')
    print('║          ga_run  –  Benchmark Test Suite                 ║')
    print('╠══════════════════════════════════════════════════════════╣')
    print(f'║  Population N1 = {args.N1:<4}   Generations tg = {args.tg:<4}          ║')
    print(f'║  Noise level   = {args.noise:<6.3f}  Tolerance  = {args.tol:<10.2e}     ║')
    print('╚══════════════════════════════════════════════════════════╝')

    runners = {
        1: run_test_1,
        2: run_test_2,
        3: run_test_3,
    }
    names = {
        1: 'TEST 1 – Damped sinusoid',
        2: 'TEST 2 – Sum of two sinusoids',
        3: 'TEST 3 – Logistic growth',
    }

    to_run = [args.test] if args.test in runners else [1, 2, 3]

    results = []
    for idx in to_run:
        r2, _ = runners[idx](args.N1, args.tg, args.noise, args.tol)
        results.append((names[idx], r2))

    if len(results) > 1:
        print_summary(results)


if __name__ == '__main__':
    main()