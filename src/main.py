import scipy.io
import argparse
import logging
import json
import theano
from scipy.sparse import hstack
from embed_methods.SVD import *
from embed_methods.multiscale import *
from proxi_methods.FaBP import *
from proxi_methods.heat_kernel import *
from proxi_methods.PPMI import *
from proxi_methods.PPR import *
from proxi_methods.L_inv import *
from proxi_methods.A_square import *
from proxi_methods.RWTM import *
from utils.errors import *
from utils.utils import *
import pdb

logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'


def prox(mat, args, logger):
    """
    @param mat: The matrix A
    @param args: all the arguments
    @return: a dic containing matrix of different taus
    """
    if args["prox_option"] == "PPMI":
        logger.info("Proximity option: PPMI")
        matrix = direct_compute_deepwalk_matrix(mat, args, logger)
    elif args["prox_option"] == "heat_kernel":
        logger.info("Proximity option: heat_kernel")
        graph = nx.Graph(mat)
        matrix, _ = heat_diffusion_ind(graph, args, logger)
    elif args["prox_option"] == "FaBP":
        logger.info("Proximity option: FaBP")
        matrix = belief_propgation(mat, args, logger)
    elif args["prox_option"] == "PPR":
        logger.info("Proximity option: PPR")
        matrix = PPR(mat, args, logger)
    elif args["prox_option"] == "inv_lap":
        logger.info("Proximity option: inv_lap")
        matrix = inv_lap(mat, args, logger)
    elif args["prox_option"] == "A2":
        logger.info("Proximity option: A2")
        matrix = A_square(mat, logger, args)
    elif args["prox_option"] == "RWTM":
        logger.info("Proximity option: RWTM")
        matrix = RWTM(mat, logger, args)
    else:
        raise WrongProxOptionError
    return matrix


def embed(mat, args, logger):
    """
    @param mat: The torch
    @param args: all the arguments
    @return: embeded matrix
    """
    if args["embed_option"]=="proximity":
        # do singular value decomposition
        logger.info("Embedding option: proximity")
        matrix = svd_deepwalk_matrix(list(mat.values())[0], args["embed_params"]["dim"], logger)
    elif args["embed_option"] == "struct":
        # do structural characteristic function
        logger.info("Embedding option: struct")
        try:
            t = args["embed_params"]["time_pnts"]
        except KeyError:
            raise MissingParamError
        time_pnts = np.linspace(t[0], t[1], t[2])
        chi = charac_function_multiscale(mat, time_pnts, logger)
        matrix = chi
    else:
        raise WrongImbedOptionError
    return matrix


def prox_or_struc_embed(args):
    logger.info("***Start loading adjacency matrix")
    logger.info("Input file: %s", args["input"])
    # load a graph
    A = load_graph(args["input"])

    # compute proximity matrix
    logger.info("***Start computing proximity matrix")
    matrix = prox(A, args, logger)

    # scipy.io.savemat('data/proximity/' + args["prox_file"] + '.mat', {args['prox_option'] + '_' + args['embed_option'] + '_' + str(i): matrix[i] for i in matrix.keys()})

    # compute embeded matrix
    logger.info("***Start embedding proximity matrix")
    Y = embed(matrix, args, logger)
    print(Y.shape)
    # save embedding
    logger.info("***Save embedding to %s", args["output"])
    np.save(args["output"], Y, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str,
            help=".json file containing all the parameters")
    with open(parser.parse_args().param) as f:
        args = json.load(f)
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    prox_or_struc_embed(args)

