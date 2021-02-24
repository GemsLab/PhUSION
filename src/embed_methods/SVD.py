import scipy.sparse as sparse
import numpy as np
import theano
from theano import tensor as T


def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    """logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))"""
    return evals


def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    """logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))"""
    return sparse.csr_matrix(Y)


def svd_deepwalk_matrix(X, dim, logger):
    logger.info("SVD on proximity matrix")
    X = X.asfptype()
    u, s, v = sparse.linalg.svds(X.todense(), dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T