import numpy as np
import scipy.sparse as sparse
import math
from utils.utils import log_filter, binary_filter


def PPR(A, args, logger):
    # beta: higher order coefficient
    # default value of beta = 0.01
    try:
        beta = args['beta']
        transform = args['transform']
        threshold = args['threshold']
    except KeyError:
        beta = 0.01
        transform = 0
    A = np.array(A.todense())
    n_nodes, _ = A.shape
    M_g = np.eye(n_nodes) - beta * A
    M_l = beta * A
    S = np.dot(np.linalg.inv(M_g), M_l)
    if transform == 1:
        return {0: log_filter(sparse.csr_matrix(S), threshold)}
    elif transform == 2:
        return {1: binary_filter(sparse.csr_matrix(S), logger, threshold)}
    else:
        return {0: sparse.csr_matrix(S)}