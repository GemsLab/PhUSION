import sys
sys.path.append('src')
import scipy.sparse as sparse
import numpy as np
import theano
from theano import tensor as T
from utils.errors import MissingParamError
import math
import scipy as sc
import networkx as nx
from numpy.linalg import inv
from utils.utils import log_filter, binary_filter


def laplacian(a):
    n_nodes, _ = a.shape
    posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
    d = sc.sparse.diags(np.array(posinv(a.sum(0))).reshape([-1, ]), 0)
    lap = sc.sparse.eye(n_nodes) - d.dot(a.dot(d))
    return lap


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1 - 2):
        basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def heat_diffusion_ind(graph, args, logger):
    '''
    This method computes the heat diffusion waves for each of the nodes
    INPUT:
    -----------------------
    graph    :    Graph (etworkx)
    taus     :    list of scales for the wavelets. The higher the tau,
                  the better the spread of the heat over the graph
    order    :    order of the polynomial approximation
    proc     :    which procedure to compute the signatures (approximate == that
                  is, with Chebychev approx -- or exact)
    logger   :    The information logger
    OUTPUT:
    -----------------------
    heat     :     tensor of length  len(tau) x n_nodes x n_nodes
                   where heat[tau,:,u] is the wavelet for node u
                   at scale tau
    taus     :     the associated scales
    '''
    # Read parameters
    try:
        taus = args["prox_params"]["taus"]
        proc = args["prox_params"]["proc"]
        order = args["prox_params"]["order"]
        transform = args["prox_params"]["transform"]
        threshold = args["prox_params"]["threshold"]
    except KeyError:
        raise MissingParamError
    ETA_MAX = 0.95
    ETA_MIN = 0.80
    if taus == 'auto':
        lap = laplacian(nx.adjacency_matrix(graph))
        try:
            l1 = np.sort(sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False))[1]
        except:
            l1 = np.sort(sc.sparse.linalg.eigsh(lap, 5,  which='SM',return_eigenvectors=False))[1]
        smax = -np.log(ETA_MIN) * np.sqrt( 0.5 / l1)
        smin = -np.log(ETA_MAX) * np.sqrt( 0.5 / l1)
        taus = [smin,smax/2+smin/2,smax]
        taus = np.array(taus)
        print('here ', taus)
    # Compute Laplacian
    a = nx.adjacency_matrix(graph)
    n_nodes, _ = a.shape
    thres = np.vectorize(lambda x : x if x > 1e-4 * 1.0 / n_nodes else 0)
    lap = laplacian(a)
    n_filters = len(taus)
    if proc == 'exact':
        ### Compute the exact signature
        lamb, U = np.linalg.eigh(lap.todense())
        heat = {}
        for i in range(n_filters):
             heat[i] = U.dot(np.diagflat(np.exp(- taus[i] * lamb).flatten())).dot(U.T)
    else:
        heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) }
        monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
        for k in range(2, order + 1):
            logger.info("Heat diffusion %d-th order: %d/%d", k, k, order)
            monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
        for i in range(n_filters):
            logger.info("Heat diffusion %d-th filter: %d/%d", i+1, i+1, n_filters)
            coeffs = compute_cheb_coeff_basis(taus[i], order)
            heat[i] = sc.sum([ coeffs[k] * monome[k] for k in range(0, order + 1)])
            temp = sc.sparse.csc_matrix(thres(heat[i].A)) # cleans up the small coefficients
            if transform == 1:
                logger.info("log transform")
                """m = T.matrix()
                f = theano.function([m], T.log(T.maximum(m, 0)))
                Y = f(temp.todense().astype(theano.config.floatX))"""
                heat[i] = log_filter(temp, threshold)
            elif transform == 2:
                heat[i] = binary_filter(temp, logger, threshold)
            else:
                heat[i] = temp
    return heat, taus