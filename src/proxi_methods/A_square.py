import numpy as np
import scipy.sparse as sp
import math
from utils.utils import log_filter, binary_filter


def A_square(A, logger, args):
    try:
        transform = args["prox_params"]['transform']
        threshold = args["prox_params"]['threshold']
        power = args["prox_params"]['power']
    except KeyError:
        transform = 0
        power = 2
    if args["embed_option"]=="proximity":
        m = A
        for i in range(power-1):
            m = np.dot(m, A)
        if log_transform == 1:
            return {0: log_filter(sp.csr_matrix(m))}
        else:
            return {0: sp.csr_matrix(m)}
    elif args["embed_option"]=="struct":
        A = A.todense()
        M = {}
        for i in range(power):
            temp = sp.csr_matrix(np.linalg.matrix_power(A, i+1))
            if transform == 1:
                 M[i] = log_filter(temp)
            elif transform == 2:
                M[i] = binary_filter(temp, logger, threshold)
            else:
                M[i] = temp
        return M



def A_A_square(A, logger, args):
    try:
        log_transform = args['transform']
        threshold = args['threshold']
        power = args['power']
    except KeyError:
        log_transform = 0
        power = 2
    M = A + 0.5 * np.dot(A, A)
    if log_transform == 1:
        print('\n\n\n\n1\n\n\n\n')
        return {0: log_filter(sp.csr_matrix(M), threshold)}
    elif log_transform == 2:
        print('\n\n\n\n2\n\n\n\n')
        return {0: binary_filter(sp.csr_matrix(M), logger, threshold)}
    else:
        return {0: sp.csr_matrix(M)}