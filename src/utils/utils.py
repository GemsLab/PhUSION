import scipy.io
import numpy as np
import utils.struct2vec_graph_utils as struct2vec
import os
import networkx as nx
import scipy.sparse as sp
import math
from utils.errors import *
import pdb


def load_adjacency_matrix(file, logger=None, variable_name="network"):
    data = scipy.io.loadmat(file)
    if logger is not None:
        logger.info("loading mat file %s", file)
    return data[variable_name]


def load_label(file, variable_name="group"):
    data = scipy.io.loadmat(file)
    # logger.info("loading mat file %s", file)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
    print(label.shape, type(label), label.min(), label.max())
    return label


def load_graph(file):
    if file[-4:] == ".mat":
        # load .mat format dataset
        A = load_adjacency_matrix(file, variable_name='network')
        # pdb.set_trace()
    else:
        A=nx.read_edgelist(file)
        head, tail = os.path.split(file)
        label = "labels-"+ tail[0:-9] + ".txt"
        label_dir = os.path.join(head, label)
        labels = struct2vec.load_label(label_dir)
        nodes = list(A)
        nodes.sort()
        A = np.array(nx.to_numpy_matrix(A, nodelist=nodes))
        A = sp.csc_matrix(A)
        # one hot encode labels
        # labels = np.array(labels)
        temp = np.zeros(len(nodes))
        for i in range(len(nodes)):
            temp[i] = int(labels[nodes[i]])
        colors = ((np.arange(4) == temp[...,None]).astype(int))
    return A


def log_filter(matrix, threshold=1): # scale smallest positive element to 1
    m = matrix.todense()
    r, c = m.shape
    if threshold == 1:
        min = np.where(m > 0, m, np.inf).min()
        for i in range(0, r):
            for j in range(0, c):
                if m[i, j] <= 0:
                    m[i, j] = 0
                else:
                    m[i, j] = math.log(m[i, j] / min)
    elif threshold == 0:
        for i in range(0, r):
            for j in range(0, c):
                if m[i, j] > 1:
                    m[i, j] = math.log(m[i, j])
                else:
                    m[i, j] = 0.0
    else:
        print('log transformation threshould error\n')
        exit(1)
    return sp.csr_matrix(m)


def log_filter1(matrix): # use a threshold 1
    m = matrix.todense()
    r, c = m.shape
    for i in range(0, r):
        for j in range(0, c):
            if m[i, j] > 1:
                m[i, j] = math.log(m[i, j])
            else:
                m[i, j] = 0.0
    return sp.csr_matrix(m)


def binary_filter(matrix, logger, threshold=0.95):
    if matrix is None:
        matrix = np.array([[0]])
    flat = np.array(matrix.todense().flatten())[0]
    flat.sort()
    l = int(len(flat) * threshold)
    logger.info(str(len(flat)) + ',' + str(threshold) + ',' + str(l))
    threshold = flat[l]
    if l == 0:
        logger.info("threshold is 0, nothing happen")
    print((matrix > threshold).astype(int))
    return sp.csr_matrix((matrix > threshold).astype(int))


def even_log_scales(dim, ran=(0.01, 100)):
    num_scale = int(256 / dim)
    return np.logspace(ran[0], ran[1], num=num_scale)
