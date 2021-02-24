import sys
sys.path.append('src')
import scipy.sparse as sparse
import numpy as np
import scipy as sc
from numpy.linalg import inv, pinv
from utils.utils import log_filter, binary_filter
import pdb


"""def RWTM(matrix, logger, args):
    try:
        transform = args['transform']
        threshold = args['threshold']
    except KeyError:
        transform = 0
    posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
    D_inv = sc.sparse.diags(np.array(posinv(matrix.sum(0))).reshape([-1, ]), 0)
    logger.info(D.shape)
    logger.info(matrix.shape)
    d1 = pinv(D.todense())
    logger.info('d1')
    R = d1.dot(matrix)
    logger.info(R.shape)
    L = sparse.csr_matrix(np.dot(R, R))
    logger.info(L.shape)
    if transform == 1:
        return {0: log_filter(L, threshold)}
    elif transform == 2:
        return {0: binary_filter(L, logger, threshold)}
    else:
        return {0: L}"""


def RWTM(matrix, logger, args):
    try:
        transform = args["prox_params"]['transform']
        threshold = args["prox_params"]['threshold']
        power = args["prox_params"]['power']
    except KeyError:
        transform = 0
        power = 2
    if args["embed_option"]=="proximity":
        posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
        D_inv = sc.sparse.diags(1 / np.array(posinv(matrix.sum(0))).reshape([-1, ]), 0)
        logger.info('d1' + str(power))
        R = D_inv.dot(matrix)
        logger.info(R.shape)
        L = R
        logger.info(L.shape)
        if transform == 1:
            return {0: log_filter(sparse.csr_matrix(L), threshold)}
        elif transform == 2:
            return {0: binary_filter(sparse.csr_matrix(L), logger, threshold)}
        else:
            return {0: sparse.csr_matrix(L)}
    elif args["embed_option"]=="struct":
        matrix = matrix.todense()
        posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
        D = sc.sparse.diags(np.array(posinv(matrix.sum(0))).reshape([-1, ]), 0)
        R = pinv(D.todense()).dot(matrix)
        L = R.copy()
        M = {}
        for i in range(power):
            temp = sparse.csr_matrix(np.linalg.matrix_power(L, i+1))
            if transform == 1:
                M[i] = log_filter(temp, threshold)
            elif transform == 2:
                M[i] = binary_filter(temp, logger, threshold)
            else:
                M[i] = temp
        return M
           