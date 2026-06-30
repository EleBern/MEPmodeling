import numpy as np


def selection_best(P, E_, R, p, op):
    """
    Select the top-p solutions by fitness.

    Parameters
    ----------
    P  : np.ndarray  [n_pop, n_parameter]  population
    E  : np.ndarray  [n_pop,]               fitness values
    R  : np.ndarray  [n_data_sample, n_pop] residuals (y - h_output)
    p  : int         number of solutions to return
    op : int         -1 → select minimum fitness, +1 → select maximum fitness

    Returns
    -------
    YY1 : np.ndarray  [p, n_parameter]      selected population
    YY2 : np.ndarray  [p,]                  fitness of YY1
    YY3 : np.ndarray  [n_data_sample, p]    residuals of YY1
    """
    # Turn minimisation into maximisation if necessary
    E = E_.copy()
    E = op * E
    # Sort from high to low — best first
    index = np.argsort(E)[::-1]
    E = np.sort(E)[::-1]
    P = P[index, :]
    R = R[:, index]


    YY1 = P[:p, :]
    YY2 = op * E[:p]   # turn back to original sign
    YY3 = R[:, :p]

    if p == 1:
        YY1 = YY1.ravel()
        YY3 = YY3.ravel()

    return YY1, YY2, YY3
