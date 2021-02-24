import scipy.io
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('data/')
from scipy.stats import entropy
import os


def analyze_dist(matrix, head):
    if head != "heat_kernel":
        matrix = np.array(matrix.todense())
    else:
        matrix = np.array(matrix[0].todense())
    rows = matrix.shape[0]
    bins = np.arange(1, rows+1)
    # plt.plot(matrix[0])
    if not os.path.isdir(head):
        os.makedirs(head)
    varrs = []
    entropies = []
    for i in range(rows):
        varrs.append(np.var(matrix[i]))
        entr = entropy(matrix[i], base=2)
        entropies.append(entr)
        plt.plot(bins, matrix[i])
        plt.savefig(os.path.join(head, 'row'+str(i)+'.png'))
        plt.close()
    varrs = np.array(varrs)
    entropies = np.array(entropies)
    np.save(os.path.join(head, 'variance'), varrs, allow_pickle=False)
    np.save(os.path.join(head, 'entropy'), entropies, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
            help=".mat file containing the proximity matrics")
    data = scipy.io.loadmat(parser.parse_args().input)
    matrix = data[list(data.keys())[-1]]
    analyze_dist(matrix, list(data.keys())[-1])