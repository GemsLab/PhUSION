import scipy.io
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys
sys.path.append('data/')


def degree_dist(matrix, name):
    m = matrix.transpose().multiply(matrix)
    sum = np.array(m.sum(axis=0))[0]
    if (np.array(matrix.sum(axis=0))[0] == np.array(matrix.transpose().multiply(matrix).sum(axis=0))[0]).all():
        print(1)
    plt.clf()
    plt.hist(sum)
    plt.xlabel('degree')
    plt.savefig(name + '_S^2.png')


    sum = np.array(matrix.sum(axis=0))[0]
    plt.clf()
    plt.hist(sum)
    plt.xlabel('degree')
    plt.savefig(name + '_S.png')


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
            help=".mat file containing the proximity matrics")
    parser.add_argument("--name", type=str,
                        help="name of the png")
    data = scipy.io.loadmat(parser.parse_args().input)
    name = parser.parse_args().name
    matrix = sp.csr_matrix(data[list(data.keys())[-1]])
    degree_dist(matrix, name)"""
    names = []
    n = ['netmf', 'heat', 'HOPE', 'FaBP', 'invlap']
    for name in n:
        names.append(name)
        names.append(name + '_l')
        names.append(name + '_b1')
        names.append(name + '_b2')
    dire = 'data/proximity/'
    for name in names:
        data = scipy.io.loadmat(dire + name + '.mat')
        matrix = sp.csr_matrix(data[list(data.keys())[-1]])
        degree_dist(matrix, name)