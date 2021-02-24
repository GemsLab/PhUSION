import sys
sys.path.append('src')
import scipy.sparse as sparse
import numpy as np
import scipy as sc
from numpy.linalg import inv, pinv
from utils.utils import log_filter, binary_filter


def laplacian(a):
    n_nodes, _ = a.shape
    posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
    d = sc.sparse.diags(np.array(posinv(a.sum(0))).reshape([-1, ]), 0)
    lap = sc.sparse.eye(n_nodes) - d.dot(a.dot(d))
    return lap


def inv_lap(matrix, args, logger):
    try:
        transform = args["prox_params"]['transform']
        threshold = args["prox_params"]['threshold']
    except KeyError:
        transform = 0
    L = sparse.csr_matrix(pinv(laplacian(matrix).todense()))
    if transform == 1:
        return {0: log_filter(L, threshold)}
    elif transform == 2:
        return {0: binary_filter(L, logger, threshold)}
    else:
        return {0: L}