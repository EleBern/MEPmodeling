"""
Biological MEP model entry point.

Example usage:
    from ga_MEPmodel_bio import ga_MEPmodel_bio
    ga_MEPmodel_bio(subj=1, withRC=1, AMPAweight=None, reRun=0)

    subj        : subject number
    withRC      : include recurrent connections (default 1)
    AMPAweight  : fixed AMPA weight value, or None to fit freely
    reRun       : 0 – load fitted result and plot simulated MEP
                  1 – re-run model fitting (backs up previous fitted result)
"""

import os
import sys
import h5py
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from load_h5            import load_h5_to_dict
from MEPmodel_bio       import MEPmodel_bio
from config_model_bio   import config_model_bio

# GA toolbox
from GA.ga_toolbox.population       import population
from GA.ga_toolbox.selection_uniq   import selection_uniq
from GA.ga_toolbox.crossover        import crossover
from GA.ga_toolbox.mutation         import mutation
from GA.ga_toolbox.mutationV        import mutationV
from GA.ga_toolbox.mutation_single  import mutation_single
from GA.ga_toolbox.fitness_function import fitness_function
from GA.ga_toolbox.gradient_search import gradient_search

# Gradient toolbox
from GA.gradient_toolbox.evaluation       import evaluation
from GA.gradient_toolbox.selection_best   import selection_best


# ==========================================================================
def objective_function(p, ref):
    """
    Objective function: runs the biological MEP model and returns the residual.
    """
    _, ref_updated = MEPmodel_bio(p, ref)
    error = ref_updated['error']
    return error, ref_updated


# ==========================================================================
def _to_h5_compatible(value):
    """
    Convert *value* to something h5py can write as a dataset.

    Resolution order
    ----------------
    1. None              → store as empty bytes
    2. dict              → signal caller to recurse (returns None sentinel)
    3. str/bytes         → np.bytes_
    4. bool              → int (must come before int check)
    5. int / float       → as-is
    6. np.ndarray
       a. 0-d object     → unwrap and recurse
       b. object dtype   → convert element-wise to str, store as bytes array
       c. Unicode dtype  → encode each element to bytes
       d. numeric/bool   → as-is
    7. list / tuple      → convert to np.ndarray; fall back to bytes on failure
    8. everything else   → str repr stored as bytes
    """
    if value is None:
        return np.bytes_(b'')
    if isinstance(value, dict):
        return None                          # sentinel: caller must recurse
    if isinstance(value, (bytes, np.bytes_)):
        return np.bytes_(value)
    if isinstance(value, str):
        return np.bytes_(value.encode('utf-8'))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_h5_compatible(value.item())
        if value.dtype == object:
            flat = [str(v).encode('utf-8') for v in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        if value.dtype.kind == 'U':
            flat = [s.encode('utf-8') for s in value.ravel()]
            return np.array(flat, dtype='S').reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        try:
            arr = np.array(value)
            if arr.dtype == object:
                raise ValueError('object array')
            if arr.dtype.kind == 'U':
                flat = [s.encode('utf-8') for s in arr.ravel()]
                return np.array(flat, dtype='S').reshape(arr.shape)
            return arr
        except (ValueError, TypeError):
            return np.bytes_(str(value).encode('utf-8'))
    return np.bytes_(str(value).encode('utf-8'))


def _save_dict_to_h5(h5file, data):
    """
    Recursively write a dict to an open h5py.File or h5py.Group.

    Tuple keys (used for nested bio-model fields) are serialised as
    their string representation so they round-trip safely.
    """
    for key, value in data.items():
        safe_key = str(key)          # h5py requires string keys; tuple → "('a','b')"
        if isinstance(value, dict):
            grp = h5file.require_group(safe_key)
            _save_dict_to_h5(grp, value)
        else:
            converted = _to_h5_compatible(value)
            if converted is None:
                grp = h5file.require_group(safe_key)
                _save_dict_to_h5(grp, value)
            else:
                h5file.create_dataset(safe_key, data=converted)


# ==========================================================================
def ga_MEPmodel_bio(subj, withRC=1, AMPAweight=None, reRun=0):
    """
    Main entry point for biological MEP model fitting using a Genetic Algorithm.
    """
    root = os.getcwd()

    # ----- model setting -----
    ref = config_model_bio(subj, withRC, AMPAweight)

    # ----- derive h5 result path (replace any existing extension) -----
    resultname_h5 = os.path.splitext(ref['resultname'])[0] + '.h5'
    result_path   = os.path.join(root, resultname_h5)

    # ----- run GA or load fitted result -----
    if os.path.isfile(result_path) and not reRun:
        print(f'Use fitted result: \n{resultname_h5}')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        p_post = tmp['p_post'].flatten()
    elif not os.path.isfile(result_path):
        print(f'Fitted result \n{resultname_h5} does not exist. Start running optimization')
        p_post = _run_and_save(ref, root, result_path)
    else:
        p_post = _run_and_save(ref, root, result_path)

    # ----- show result -----
    plotOn = 0
    MEPmodel_bio(p_post, ref, plotOn)
    print("R2 ", ref["R2"])

# ==========================================================================
def _run_and_save(ref, root, result_path):
    """
    Inline translation of MATLAB run_ga.  Runs the full GA loop, saves the
    result as an HDF5 file, and returns the best parameter set.

    Parameters
    ----------
    ref         : dict   model configuration from config_model_bio
    root        : str    working directory
    result_path : str    full path to the .h5 output file

    Returns
    -------
    p_post : np.ndarray  [nParams,]  — parameters from the last generation
    """
    # ------------------------------------------------------------------
    # 0.  Hyperparameters  (mirrors MATLAB run_ga)
    # ------------------------------------------------------------------
    op = -1          # -1: minimise,  1: maximise

    N1 = 60          # population size
    N2 = 100         # crossover: number of pairs
    N3 = 100         # mutation:  number of pairs
    tg = 1           # total generations

    # Gradient-search configuration
    conf = {
        'gLoop': 10,   # iterations per gradient search
        'gL':    -12,
        'gU':    12,
        'gTol':  0.01,
        'op':    op,
        'myfunc': objective_function,
    }
    conf['gT'] = abs(conf['gU'] - conf['gL']) + 1

    # ------------------------------------------------------------------
    # 1.  Boundaries
    # ------------------------------------------------------------------
    LR      = ref['model']['boundary'][:, 0]
    UR      = ref['model']['boundary'][:, 1]
    nParams = len(LR)

    # Expose boundary at top level (MEPmodel_bio may need it)
    ref['boundary'] = ref['model']['boundary']

    # conf also needs the bounds and the full ref for gradient_search
    conf['LR']     = LR
    conf['UR']     = UR
    conf['y_goal'] = ref

    # ------------------------------------------------------------------
    # 2.  Collect previous solutions to seed the population
    #     Mirrors MATLAB: primary result file + fixed-AMPAweight loop
    # ------------------------------------------------------------------
    solution_ini = np.empty((0, nParams))

    # 2a. Primary result file for this subject
    primary_path = result_path          # already the .h5 path for this subject
    if os.path.isfile(primary_path):
        print(f'{primary_path} found.')
        with h5py.File(primary_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        solution_ini = np.vstack([solution_ini, np.atleast_2d(tmp['p_post'])])

    # 2b. Fixed-AMPAweight loop (AMPAw = 0.2, 0.3, …, 0.8)
    #     Mirrors MATLAB: for AMPAw=0.2:0.1:0.8 … load & optionally fix p(12)
    ampa_fixed = ref['model'].get('AMPAweight')   # None / [] → free;  scalar → fixed
    ampa_is_fixed = (ampa_fixed is not None and
                     not (isinstance(ampa_fixed, (list, np.ndarray))
                          and len(np.atleast_1d(ampa_fixed)) == 0))

    fixed_seed_dir = os.path.join(root, 'fitted_results', 'bio', 'fixed_AMPAweight')
    for AMPAw in np.arange(0.2, 0.85, 0.1):       # 0.2 … 0.8 inclusive
        AMPAw = round(AMPAw, 1)
        tmpname = os.path.join(
            fixed_seed_dir,
            f"result_bio_s{ref['subj']}[{AMPAw:g}].h5"
        )
        if os.path.isfile(tmpname):
            print(f'{tmpname} found.')
            with h5py.File(tmpname, 'r') as f:
                tmp = load_h5_to_dict(f)
            p_tmp = np.atleast_1d(tmp['p_post']).ravel()
            if ampa_is_fixed:
                # fix AMPA weight parameter (0-indexed: param index 11 = MATLAB 12)
                p_tmp = p_tmp.copy()
                p_tmp[11] = float(np.atleast_1d(ampa_fixed).ravel()[0])
            solution_ini = np.vstack([solution_ini, p_tmp])

    # 2c. Clip seeded solutions to parameter bounds
    if solution_ini.size > 0:
        for i in range(nParams):
            solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # ------------------------------------------------------------------
    # 3.  Initialisation
    # ------------------------------------------------------------------
    print('======== Initialization ========')
    P = population(N1, nParams, LR, UR)           # [N1 x nParams] random solutions
    if solution_ini.size > 0:
        P = np.vstack([P, solution_ini])           # append seeded solutions

    E, R, _ = evaluation(P, objective_function, ref)  # E: [nSolutions,]  R: [T x nSolutions]
    P, E, R = selection_best(P, E, R, N1, op)      # keep best N1
    R1 = R[:, 0]                                   # residual of current best

    print('done')
    print(f'Minimum cost: {E[0]}')
    print('================================')
    E_crit = E[0]

    # History accumulators  (row w holds generation w data)
    K          = []   # [[avg_cost, best_cost], …]
    KP         = []   # [best_params, …]
    KS         = []   # [best_cost, …]
    GA_counter = []   # [0/1 per generation]

    # ------------------------------------------------------------------
    # 4.  Online-plot setup
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    plt.ion()
    plt.show()

    # ------------------------------------------------------------------
    # 5.  Main GA loop
    # ------------------------------------------------------------------
    w = 0   # generation index (0-based, matching list append)
    j = 1   # generation counter for stopping (mirrors MATLAB j=1 before loop)

    while True:

        # ---- 5a. Gradient search on current best ----
        print('======= Gradient search ========')
        Para_E_grd, E_grd, R_grd = gradient_search(P[0:1, :], R1, conf, E_crit)
        # Replace best if gradient improved it
        if op * E_grd[0] > op * E[0]:
            P[0, :]  = Para_E_grd[0, :]
            E[0]     = E_grd[0]
            R[:, 0]  = R_grd[:, 0]
        print('done')

        # ---- 5b. Single-parameter mutation of current best ----
        print('======= single-parameter mutation ========')
        P_ = mutation_single(P[0:1, :], LR, UR)   # [nParams x nParams]
        E_, R_ = evaluation(P_, objective_function, ref)
        print('done')

        # ---- 5c. Gradient search on each single-param mutant ----
        print('======= Gradient search ========')
        Para_E_grd = np.empty_like(P_)
        E_grd      = np.empty(len(E_))
        R_grd      = np.empty_like(R_)
        for i in range(len(E_)):
            print(f'[{i+1}/{len(E_)}] cost: {E_[i]:.6f}')
            pg, eg, rg = gradient_search(P_[i:i+1, :], R_[:, i:i+1], conf, E_crit)
            Para_E_grd[i, :] = pg[0, :]
            E_grd[i]         = eg[0]
            R_grd[:, i]      = rg[:, 0]

        # Replace mutants where gradient improved them
        index = op * E_grd > op * E_
        P_[index, :]   = Para_E_grd[index, :]
        E_[index]      = E_grd[index]
        R_[:, index]   = R_grd[:, index]

        # Append mutants to population
        P = np.vstack([P, P_])                     # [(N1 + nParams) x nParams]
        E = np.concatenate([E, E_])
        R = np.hstack([R, R_])
        print('done')

        # Track best-after-gradient for GA-effectiveness check
        _, E_show, _ = selection_best(P, E, R, 1, op)
        print(f'best after gradient: {E_show[0]}')

        # ---- 5d. GA operators ----
        print('GA search...')
        P_mutV     = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)          # N1 solutions
        P_cross    = crossover(P, N2)                                   # 2*N2 solutions
        P_mut      = mutation(P, N3)                                    # 2*N3 solutions
        P_new      = np.vstack([P_mutV, P_cross, P_mut])

        # Evaluate only the newly generated solutions
        E_new, _, _ = evaluation(P_new, objective_function, ref)

        # Merge all solutions (mirrors MATLAB indexing: P(N1+nParams+1:end) is P_new)
        P = np.vstack([P, P_new])
        E = np.concatenate([E, E_new])

        # Selection: keep N1 unique best solutions
        P, E = selection_uniq(P, E, N1, N1, op, LR, UR)

        # Re-evaluate residual of new best solution
        _, R1_arr, _ = evaluation(P[0:1, :], objective_function, ref)
        R1 = R1_arr[:, 0]
        print('done')

        # ---- 5e. Record history ----
        avg_cost = E.sum() / N1
        K.append([avg_cost, E[0]])
        KP.append(P[0, :].copy())
        KS.append(E[0])
        E_crit = E[0]

        print('========')
        print(f'current best Loss: {KS[w]}')
        print('========')

        gof = fitness_function(ref['y0'].ravel(), R1)
        print('========')
        print(f'current best R2: {gof}')
        print('========')

        # GA-effectiveness flag
        if E_show[0] > E[0]:
            print('GA works')
            GA_counter.append(1)
        else:
            print("GA doesn't work")
            GA_counter.append(0)

        # ---- 5f. Online plot ----
        _, houtput = objective_function(KP[-1], ref)
        K_arr  = np.array(K)
        GA_arr = np.array(GA_counter)

        for ax in axes:
            ax.cla()

        axes[0].plot(K_arr[:, 1], 'b.')
        axes[0].plot(K_arr[:, 0], 'r.')
        axes[0].set_title('Blue - Best            Red - Average')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Loss function')
        axes[0].set_yscale('log')
        axes[0].grid(True)

        axes[1].plot(E, 'b.')
        axes[1].set_xlabel('Chromosomes')
        axes[1].set_ylabel('Loss function')
        axes[1].set_yscale('log')
        axes[1].grid(True)

        axes[2].plot(KP[-1], '-ko')
        axes[2].set_title('parameter')

        axes[3].plot(ref['y0'].flatten(order='F'), 'k', linewidth=1.5)
        axes[3].plot(
            houtput['sim']['simMEP2'].flatten(order='F'), 'r', linewidth=1.0
        )
        axes[3].set_title('target & best fit')

        axes[4].plot(GA_arr, 'b.')
        axes[4].set_xlabel('Generations')
        suc_rate = GA_arr.sum() / len(GA_arr) if len(GA_arr) else 0
        axes[4].set_title(
            f'0--not work, 1--work, total success rate: {suc_rate:.2f}'
        )

        plt.pause(0.01)
        fig.canvas.draw()

        # ---- 5g. Stopping criteria ----
        w += 1
        j += 1

        if j > tg:          # max generations reached
            break
        if KS[-1] < 0.01:   # good-enough fit
            break

    # ------------------------------------------------------------------
    # 6.  Extract best overall result
    #     MATLAB: p_post = KP(end,:)  — last generation's best
    #     (The overall minimum across all generations is also reported.)
    # ------------------------------------------------------------------
    KS_arr = np.array(KS)
    KP_arr = np.array(KP)

    if op == -1:
        best_idx = int(np.argmin(KS_arr))
        print(f'minimum: {KS_arr[best_idx]}')
    else:
        best_idx = int(np.argmax(KS_arr))
        print(f'maximum: {KS_arr[best_idx]}')

    # p_post follows MATLAB convention: last generation's best
    p_post = KP_arr[-1, :]

    # ------------------------------------------------------------------
    # 7.  Backup previous result (if any) then save
    # ------------------------------------------------------------------
    if os.path.isfile(result_path):
        timestamp   = datetime.now().strftime('%Y-%m%d-%H%M')
        backup_path = os.path.splitext(result_path)[0] + f'_backup-{timestamp}.h5'
        shutil.copyfile(result_path, backup_path)
        print(f'Previous result backed up to: {backup_path}')

    _, ref_final = MEPmodel_bio(p_post, ref, 0)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with h5py.File(result_path, 'w') as f:
        f.create_dataset('p_post', data=p_post)
        f.create_dataset('KP',     data=KP_arr)
        f.create_dataset('KS',     data=KS_arr)
        f.create_dataset('P',      data=P)
        grp = f.create_group('ref')
        _save_dict_to_h5(grp, ref_final)

    print('fitted result saved:')
    print(result_path)

    return p_post


# ==========================================================================
if __name__ == '__main__':
    subj       = int(sys.argv[1])   if len(sys.argv) > 1 else 1
    withRC     = int(sys.argv[2])   if len(sys.argv) > 2 else 1
    AMPAweight = float(sys.argv[3]) if len(sys.argv) > 3 else None
    reRun      = int(sys.argv[4])   if len(sys.argv) > 4 else 0
    ga_MEPmodel_bio(subj, withRC, AMPAweight, reRun)