import time
import sys
import numpy as np


def evaluation(X, func, y_goal):
    """
    Evaluate the objective function for every solution in the population.

    Parameters
    ----------
    X      : np.ndarray  [x, nParams]  population (each row is one solution)
    func   : callable    objective function with signature:
                         error, houtput = func(P, y_goal)
    y_goal : target output passed through to func

    Returns
    -------
    Fit     : np.ndarray  [x, ]   fitness value (sumsqr of error) per solution
    Error   : np.ndarray  [nData, x]  residuals per solution
    Houtput : list[any]  raw model outputs per solution
    """
    x, _ = X.shape  # x: population size
    Fit = np.zeros((x))
    total_pop = str(x)

    Error   = None  # will be built on first iteration
    Houtput = [None] * x

    for j in range(x):
        # -------- objective function --------
        P = X[j, :]              # parameter vector for this solution

        t_start = time.time()
        error, houtput = func(P, y_goal)   # error and model output
        fit = np.sum(error ** 2)           # sumsqr(error)
        gettime = time.time() - t_start
        # ------------------------------------

        msg = f'simulation time: {gettime:3.5f} --> {j + 1}/{total_pop} '
        sys.stdout.write('\r' + msg)
        sys.stdout.flush()

        Fit[j] = fit

        # Initialise Error array on first iteration once we know its shape
        if Error is None:
            Error = np.zeros((error.size, x))
        Error[:, j] = error

        Houtput[j] = houtput

    print()   # newline after progress output
    return Fit, Error, Houtput
