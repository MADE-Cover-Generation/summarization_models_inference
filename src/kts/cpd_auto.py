import numpy as np

from kts.cpd_nonlin import cpd_nonlin

import time


def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Detect change points automatically selecting their number

    :param K: Kernel between each pair of frames in video
    :param ncp: Maximum number of change points
    :param vmax: Special parameter
    :param desc_rate: Rate of descriptor sampling, vmax always corresponds to 1x
    :param kwargs: Extra parameters for ``cpd_nonlin``
    :return: Tuple (cps, costs)
        - cps - best selected change-points
        - costs - costs for 0,1,2,...,m change-points
    """
    algo_time = time.time()
    start_time = time.time()
    m = ncp
    _, scores = cpd_nonlin(K, m, backtrack=False, **kwargs)
    print(f'Calculated first nonlin time: {time.time() - start_time}')

    N = K.shape[0]
    N2 = N * desc_rate  # length of the video before down-sampling

    penalties = np.zeros(m + 1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m + 1)
    penalties[1:] = (vmax * ncp / (2.0 * N2)) * (np.log(float(N2) / ncp) + 1)

    costs = scores / float(N) + penalties
    m_best = np.argmin(costs)

    start_time = time.time()
    cps, scores2 = cpd_nonlin(K, m_best, **kwargs)
    print(f'Calculated second nonlin time: {time.time() - start_time}')

    print(f'cpd_auto time: {time.time() - algo_time} \n')

    return cps, scores2
