from scipy.sparse import identity
from scipy.sparse import diags
import scipy.sparse as sparse
import numpy as np
from numpy import square
from numpy import trace
from numpy import amax
from math import sqrt
import math
from utils.utils import log_filter, binary_filter


def InverseMatrix(A, args):
    '''
    use Fast Belief Propagatioin
    CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
    Kenneth Pao, Christos Faloutsos
    Unifying Guilt-by-Association Approaches
    return [I+a*D-c*A]^-1
    '''
    try:
        log_transform = args['log_transform']
    except KeyError:
        log_transform = 0
    I = identity(A.shape[0])  # identity matirx
    D = diags(sum(A).toarray(), [0])  # diagonal degree matrix

    c1 = trace(D.toarray()) + 2
    c2 = trace(square(D).toarray()) - 1
    h_h = sqrt((-c1 + sqrt(c1 * c1 + 4 * c2)) / (8 * c2))

    a = 4 * h_h * h_h / (1 - 4 * h_h * h_h)
    c = 2 * h_h / (1 - 4 * h_h * h_h)

    # M=I-c*A+a*D
    # S=inv(M.toarray())
    '''
    compute the inverse of matrix [I+a*D-c*A]
    use the method propose in Unifying Guilt-by-Association equation 5
    '''
    M = c * A - a * D
    S = I
    mat = M
    power = 1
    while amax(M.toarray()) > 10 ** (-9) and power < 7:
        S = S + mat
        mat = mat * M
        power += 1
    if log_transform == 1:
        return {0: log_filter(sparse.csr_matrix(S))}
    else:
        return {0: S}


def belief_propgation(A, args, logger):
    '''
    use Fast Belief Propagatioin
    CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
    Kenneth Pao, Christos Faloutsos
    Unifying Guilt-by-Association Approaches
    return [I+a*D-c*A]^-1
    '''
    try:
        transform = args["prox_params"]['transform']
        threshold = args["prox_params"]['threshold']
        scale1 = args["prox_params"]['scale1']
        scale2 = args["prox_params"]['scale2']
    except KeyError:
        transform = 0
        scale1 = 1
        scale2 = 1
    I = sparse.identity(A.shape[0])  # identity matirx
    D = sparse.diags(sum(A).toarray(), [0])  # diagonal degree matrix

    c1 = np.trace(D.toarray()) + 2
    c2 = np.trace(np.square(D).toarray()) - 1
    h_h = math.sqrt((-c1 + math.sqrt(c1 * c1 + 4 * c2)) / (8 * c2))

    a = scale1 * 4 * h_h * h_h / (1 - 4 * h_h * h_h)
    c = scale2 * 2 * h_h / (1 - 4 * h_h * h_h)

    # M=I-c*A+a*D
    # S=inv(M.toarray())
    '''
    compute the inverse of matrix [I+a*D-c*A]
    use the method propose in Unifying Guilt-by-Association equation 5
    '''
    M = c * A - a * D
    S = I
    mat = M
    power = 1
    while np.amax(M.toarray()) > 10 ** (-9) and power < 7:
        S = S + mat
        mat = mat * M
        power += 1
    if transform == 1:
        S = log_filter(S, threshold)
    elif transform == 2:
        S = binary_filter(S, logger, threshold)
    return {0: S}
