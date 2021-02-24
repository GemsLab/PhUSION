import os
import sys
import pickle as pkl
import numpy as np
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import argparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import pdb
import networkx as nx
sys.path.append('/y/jingzhuu/ProxStruc-pass_torch/src')
import utils.struct2vec_graph_utils as struct2vec
import seaborn as sb
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logger = logging.getLogger(__name__)


def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]


def load_label(file, variable_name="group"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    label = data[variable_name].todense().astype(np.int)
    label = np.array(label)
    print(label.shape, type(label), label.min(), label.max())
    return label


def creat_graph(args):
    if args.input == "barbel":
        # create synthetic barbel graph
        A , colors = barbel_graph(0, 5, 5,plot=False)
        pdb.set_trace()
    elif args.input == "synthetic":
        # create normal synthetic graphs using build_graph
        width_basis = 15
        nbTrials = 20
        basis_type = "cycle" 
        n_shapes = 5  ## numbers of shapes to add 
        list_shapes = [["house"]] * n_shapes
        identifier = 'AA'  ## just a name to distinguish between different trials
        name_graph = 'houses'+ identifier
        sb.set_style('white')
        add_edges = 0
        A, communities, _ , colors = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                        add_random_edges=add_edges, plot=False,
                                        savefig=False)
    elif args.input[-4:] == ".mat":
        # load .mat format dataset
        A = load_adjacency_matrix(args.input,variable_name=args.matfile_variable_name)
        colors = load_label(file=args.input, variable_name=args.matfile_variable_name)
    else:
        A=nx.read_edgelist(args.input)
        head, tail = os.path.split(args.input)
        label = "labels-"+ tail[0:-9] + ".txt"
        label_dir = os.path.join(head, label)
        labels = struct2vec.load_label(label_dir)
        nodes = list(A)
        nodes.sort()
        A = np.array(nx.to_numpy_matrix(A, nodelist=nodes))
        A = sparse.csc_matrix(A)
        # one hot encode labels
        # labels = np.array(labels)
        temp = np.zeros(len(nodes))
        for i in range(len(nodes)):
            temp[i] = int(labels[nodes[i]])
        colors = ((np.arange(4) == temp[...,None]).astype(int))
    return A, colors


def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    y_pred = np.zeros_like(y, dtype=np.int)
    for i in range(y.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1
    return y_pred


def load_w2v_feature(file):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[] for i in range(n)]
                continue
            index = int(content[0])
            for x in content[1:]:
                feature[index].append(float(x))
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro, accuracy = [], [], []
    # pdb.set_trace()
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = construct_indicator(y_score, y_test)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        logger.info("micro f1 %f macro f1 %f accuracy %f", mi, ma, acc)
        micro.append(mi)
        macro.append(ma)
        accuracy.append(acc)
    logger.info("%d fold validation, training ratio %f", len(micro), train_ratio)
    logger.info("Average micro %.2f, Average macro %.2f, Average accuracy %.2f",
            np.mean(micro) * 100,
            np.mean(macro) * 100,
            np.mean(accuracy) * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help="input file path for labels (.mat)")
    parser.add_argument("--embedding", type=str, required=True,
            help="input file path for embedding (.npy)")
    parser.add_argument("--matfile-variable-name", type=str, default='group',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--seed", type=int, default=123,
            help="seed used for random number generator when randomly split data into training/test set.")
    parser.add_argument("--start-train-ratio", type=int, default=10,
            help="the start value of the train ratio (inclusive).")
    parser.add_argument("--stop-train-ratio", type=int, default=90,
            help="the end value of the train ratio (inclusive).")
    parser.add_argument("--num-train-ratio", type=int, default=9,
            help="the number of train ratio choosed from [train-ratio-start, train-ratio-end].")
    parser.add_argument("--C", type=float, default=1.0,
            help="inverse of regularization strength used in logistic regression.")
    parser.add_argument("--num-split", type=int, default=10,
            help="The number of re-shuffling & splitting for each train ratio.")
    parser.add_argument("--output", type=str, default=" ",
            help="Output filename.")
    args = parser.parse_args()
    f_n = args.embedding.split('/')[-1]
    logging.basicConfig(
            # filename=args.output, filemode="w", # uncomment this to log to file
            level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    logger.info("Loading label from %s...", args.input)
    data, label = creat_graph(args)
    logger.info("Label loaded!")
    logger.info("Loading network embedding from %s...", args.embedding)
    ext = os.path.splitext(args.embedding)[1]
    if ext == ".npy":
        embedding = np.load(args.embedding)
    elif ext == ".pkl":
        with open(args.embedding, "rb") as f:
            embedding = pkl.load(f)
    else:
        # Load word2vec format
        embedding = load_w2v_feature(args.embedding)
    logger.info("Network embedding loaded!")

    train_ratios = np.linspace(args.start_train_ratio, args.stop_train_ratio,
            args.num_train_ratio)

    print('e:',embedding.shape)
    print(label.shape,'o')

    for tr in train_ratios:
        predict_cv(embedding, label, train_ratio=tr/100.,
                n_splits=args.num_split, C=args.C, random_state=args.seed)