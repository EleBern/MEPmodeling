import os
import sys
import h5py
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from MEPmodel_pheno       import MEPmodel_pheno
from load_h5            import load_h5_to_dict
from config_model_pheno   import config_model_pheno
from objective_function import objective_function_pheno as objective_function

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

def ga_MEPmodel_pheno(subj, reRun=0):
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(root, 'GA', 'ga_toolbox'))
    sys.path.append(os.path.join(root, 'GA', 'gradient_toolbox'))

    # ----- model setting -----
    ref = config_model_pheno(subj)

    # ----- run GA -----
    result_path = os.path.join(root, ref['resultname'])
    if os.path.exists(result_path) and not reRun:
        print(f'Use fitted result: \n{ref["resultname"]}')
        with h5py.File(result_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        p_post = tmp['p_post'].flatten()
    else:
        p_post = run_ga(ref)

    # ----- show result -----
    plotOn = 1
    MEPmodel_pheno(p_post, ref, plotOn)

# ==========================================================================
def run_ga(ref):
    myfunc = objective_function
    op = -1  # -1: find minimum
             #  1: find maximum

    LR = np.asarray(ref['boundary'])[:, 0]
    UR = np.asarray(ref['boundary'])[:, 1]
    nParams = len(LR)

    conf = {
        'UR': UR,
        'LR': LR,
        'op': op,
        'myfunc': myfunc,
        'y_goal': ref,
        'gLoop': 10,   # for gradient search
        'gL': -12,
        'gU': 12,
    }
    conf['gT'] = abs(conf['gU'] - conf['gL']) + 1
    conf['gTol'] = 0.01

    # ------------------------------------------------------------------
    N1 = 60   # population size
    N2 = 100  # crossover, number of pairs to crossover
    N3 = 100  # mutation, number of pairs to mutate
    tg = 5    # total generations

    # rng(200)  # not set, matching commented-out MATLAB line
    # warnings ignored, matching commented-out MATLAB line

    K = np.zeros((0, 2))     # history of [average cost, best cost]
    KP = np.zeros((0, nParams))  # history of [best solution]
    KS = np.zeros(0)         # history of [best cost]
    w = 0                    # counter (0-based to match python indexing)
    j = 1                    # counter

    fig = plt.figure()

    # %%%%%%%%%%collect previous solutions%%%%%%%%%%%%%
    root = os.path.dirname(os.path.abspath(__file__))
    tmpname = os.path.join(root, ref['resultname'])
    solution_ini = np.zeros((0, nParams))
    if os.path.exists(tmpname):
        print(f'{tmpname} found.')
        with h5py.File(tmpname, 'r') as f:
            tmp = load_h5_to_dict(f)
        solution_ini = np.vstack([solution_ini, np.atleast_2d(tmp['p_post'])])


    # rectify min max
    for i in range(nParams):
        if solution_ini.shape[0] > 0:
            solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # -----initialization-----
    print('======== Initialization ========')
    P = population(N1, nParams, LR, UR)  # generate [60 x nParams] random solutions
    if solution_ini.shape[0] > 0:
        P = np.vstack([P, solution_ini])  # add pre-selected solutions
    E, R, _ = evaluation(P, myfunc, ref)  # E: evaluation fitness, R: residual, error
    P, E, R = selection_best(P, E, R, N1, op)
    R1 = R[:, 0]
    print('done')
    print(f'Minimum cost: {E[0]}')
    print('================================')
    E_crit = E[0]

    GA_counter = []

    # -----loop-----
    while True:
        print('======= Gradient search ========')
        Para_E_grd, E_grd, R_grd = gradient_search(P[0:1, :], R1, conf, E_crit)
        # replace
        if op * E_grd > op * E[0]:
            P[0, :] = Para_E_grd
            E[0] = E_grd[0]
            R[:, 0] = R_grd[:, 0]
        print('done')

        print('======= single-parameter mutation ========')
        P_ = mutation_single(P[0, :], LR, UR)

        E_, R_, _ = evaluation(P_, myfunc, ref)
        print('done')

        print('======= Gradient search ========')
        Para_E_grd = np.zeros((len(E_), nParams))
        E_grd = np.zeros(len(E_))
        R_grd = np.zeros((R_.shape[0], len(E_)))
        for i in range(len(E_)):
            print(f'[{i + 1}/{len(E_)}] cost: {E_[i]:f}')
            pg, eg, rg = gradient_search(P_[i:i+1, :], R_[:, i:i+1], conf, E_crit)
            Para_E_grd[i, :] = pg[0, :]
            E_grd[i]         = eg[0]
            R_grd[:, i]      = rg[:, 0]

        # replace
        index = op * E_grd > op * E_
        P_[index, :] = Para_E_grd[index, :]  # update gradient mutation
        E_[index] = E_grd[index]
        R_[:, index] = R_grd[:, index]
        P = np.vstack([P, P_])  # [(60 + nParams) x nParams] solutions
        E = np.concatenate([E, E_])  # [1 x (60 + nParams)] cost
        R = np.hstack([R, R_])  # [timepoints x (60 + nParams)] residual
        print('done')

        # %%%%%%%%%%
        # add to show, delete later
        _, E_show, _ = selection_best(P, E, R, 1, op)
        print(f'best after gradient: {E_show}')
        # %%%%%%%%%%

        # GA
        print('GA search...')
        P_mutV     = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)          # N1 solutions
        P_cross    = crossover(P, N2)                                   # 2*N2 solutions
        P_mut      = mutation(P, N3)                                    # 2*N3 solutions
        P_new = np.vstack([P_mutV, P_cross, P_mut])

        E_, _, _ = evaluation(P_new, myfunc, ref)

        P = np.vstack([P, P_new])
        E = np.hstack([E, E_])  # cost [1 x (N1+N2+N3)*2+nParams]

        # selection
        P, E = selection_uniq(P, E, N1, N1, op, LR, UR)  # select N1 solutions
        _, R1, _ = evaluation(P[0:1, :], myfunc, ref)  # R1: residual of best solution
        print('done')

        # grow histories
        K = np.vstack([K, [np.sum(E) / N1, E[0]]])  # average cost, best cost (for plot)

        KP_row = P[0, :nParams]
        KP = np.vstack([KP, KP_row])  # save best
        KS = np.append(KS, E[0])      # save best
        E_crit = E[0]
        print('========')
        print(f'current best Loss: {KS[w]}')
        print('========')
        gof = fitness_function(np.ravel(ref['y0']), R1)

        print('========')
        print(f'current best R2: {gof}')
        print('========')

        # %%%%%%%
        # add to show, delete later
        if E_show > E[0]:
            print('GA works')
            GA_counter.append(1)
        else:
            print("GA doesn't work")
            GA_counter.append(0)
        # %%%%%%%%%%
        w = w + 1  # update generation
        j = j + 1  # update generation counter

        # online plot
        _, houtput = myfunc(KP[-1, :], ref)
        plt.clf()

        plt.subplot(5, 1, 1)
        plt.plot(K[:, 1], 'b.')
        plt.plot(K[:, 0], 'r.')
        plt.title('Blue - Best            Red - Average')
        plt.xlabel('Generation')
        plt.ylabel('Loss function')
        plt.grid(True)
        plt.yscale('log')

        plt.subplot(5, 1, 2)
        plt.plot(E, 'b.')
        plt.xlabel('Cromosomes')
        plt.ylabel('Loss function')
        plt.grid(True)
        plt.yscale('log')

        plt.subplot(5, 1, 3)
        plt.plot(KP[-1, :], '-ko')
        plt.title('parameter')

        plt.subplot(5, 1, 4)
        plt.plot(np.ravel(ref['y0']), 'k', linewidth=1.5)
        plt.plot(np.ravel(houtput['sim']['simMEP2']), 'r', linewidth=1)
        plt.title('target & best fit')

        plt.subplot(5, 1, 5)
        plt.plot(GA_counter, 'b.')
        plt.xlabel('Generations')
        gatoshow = GA_counter.count(1) / len(GA_counter)
        plt.title(f'0--not work,1--work, total succeful rate: {gatoshow}')
        plt.draw()
        plt.pause(0.001)

        # stop: number of generations
        if j > tg:
            break

        # stop: good fit
        if KS[-1] < 0.01:
            j = tg + 1
            break

    # get final result
    if op == -1:
        idx = int(np.argmin(KS))
        minimum = KS[idx]
        find_parameter = KP[idx, :]
        print(minimum)

    if op == 1:
        idx = int(np.argmax(KS))
        maximum = KS[idx]
        find_parameter = KP[idx, :]
        print(maximum)

    # make a copy of previous fitted result
    result_path = os.path.join(root, ref['resultname'])
    if os.path.exists(result_path):
        base, ext = os.path.splitext(ref['resultname'])
        timestamp = datetime.now().strftime('%Y-%m%d-%H%M')
        filename_backup = f'{base}_backup-{timestamp}.mat'
        shutil.copyfile(result_path, os.path.join(root, filename_backup))

    # save fitted result
    p_post = KP[-1, :]
    _, ref = MEPmodel_pheno(p_post, ref, 0)  # update ref
    with h5py.File(result_path, 'w') as f:
        f.create_dataset('p_post', data=p_post)
        f.create_dataset('KP',     data=KP)
        f.create_dataset('KS',     data=KS)
        f.create_dataset('P',      data=P)
        grp = f.create_group('ref')
        _save_dict_to_h5(grp, ref)
    print('fitted result saved:')
    print(ref['resultname'])

    return p_post


if __name__ == '__main__':
    subj       = int(sys.argv[1])   if len(sys.argv) > 1 else 1
    reRun      = int(sys.argv[2])   if len(sys.argv) > 2 else 0
    ga_MEPmodel_pheno(subj, reRun)
    pass