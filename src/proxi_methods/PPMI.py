import sys
sys.path.append('src')
import scipy.sparse as sparse
from scipy.sparse import csgraph
import scipy.sparse as sp
import numpy as np
import theano
from theano import tensor as T
from utils.errors import MissingParamError
from utils.utils import log_filter, binary_filter
import math


def direct_compute_deepwalk_matrix(A, args, logger):
    res = {}
    try:
        windows = args["prox_params"]['window']
        b = args["prox_params"]['negative']
        transform = args["prox_params"]['transform']
        threshold = args["prox_params"]['threshold']
    except KeyError:
        raise MissingParamError
    for window in windows:
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} A D^{-1/2}
        X = sparse.identity(n) - L
        S = np.zeros_like(X)
        X_power = sparse.identity(n)
        for i in range(window):
            logger.info("Deep Walk %d-th power, %d/%d", i+1, i+1, window)
            X_power = X_power.dot(X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sparse.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T)
        m = T.matrix()
        if transform == 1:
            logger.info("log transform")
            res[window] = log_filter(M, threshold)
            # res[window] = log_filter(M)
        elif transform == 2:
            res[window] = binary_filter(M, logger, threshold)
        else:
            logger.info("no transform")
            res[window] = sparse.csr_matrix(M)
    return res