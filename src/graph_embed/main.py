import os, sys, argparse
from config import *

from read_graph import *
from util import *

from embed import *
from rgm import *
from wl import *

import numpy as np
import pdb
import json
import logging

from sklearn.model_selection import cross_validate
from sklearn import svm

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.path.append('../')
sys.path.append('../proxi_methods')
from embed_methods.SVD import *
from embed_methods.multiscale import *
from proxi_methods.FaBP import *
from proxi_methods.heat_kernel import *
from proxi_methods.PPMI import *
from proxi_methods.PPR import *
from proxi_methods.L_inv import *
from proxi_methods.A_square import *
from proxi_methods.RWTM import *

logger = logging.getLogger(__name__)

# def parse_args():
# 	parser = argparse.ArgumentParser(description="Run graph classification")
# 	parser.add_argument('--dataset', nargs='?', default='mutag', help='dataset (mutag, enzymes, nci, ptc, imdb, letter)')
# 	parser.add_argument('--emb', nargs='?', default='heat_kernel', help='Embedding method (netmf, heat_kernel, FaBP, HOPE, adj)')
# 	parser.add_argument('--dimensionality', nargs='?', type = int, default=128, help='dimensionality of embeddings')
# 	parser.add_argument('--numlevels', nargs='?', type = int, default=4, help="Number of levels for pyramid match kernel")
# 	parser.add_argument('--numfolds', nargs='?', type = int, default=10, help="Number of folds for k-fold cross-validation.  For experiments, 10")
# 	parser.add_argument('--randomseed', nargs='?', type = int, default=0, help="Random seed for splitting data into folds.  For experiments, int 0-9")
# 	parser.add_argument('--normhist', action="store_true", help="whether to normalize histograms for histogram mapping. Default False")
# 	parser.add_argument('--saveembed', action="store_true", help="use this flag to save embeddings to use later")
# 	parser.add_argument('--loadembed', action="store_true", help="use this flag to use previously computed embeddings")
# 	parser.add_argument('--savewl', action="store_true", help="use this flag to store WL label expansions")
# 	parser.add_argument('--loadwl', action="store_true", help="use this flag to use previously computed WL label expansions")
# 	parser.add_argument('--saveoutput', action="store_true", help="use this flag to store output for kernel machine (features/kernel matrix)")
# 	parser.add_argument('--loadoutput', action="store_true", help="use this flag to use previously computed output for kernel machine (features/kernel matrix)")
# 	parser.add_argument('--svmc', nargs='?', type = float, default=1.0, help="SVM tradeoff parameter.  Default is 1.0")
# 	parser.add_argument('--wliter', nargs='?', type = int, default=0, help="Number of iterations of Weisfeiler-Lehman kernel. Default is 0 (no WL kernel)")
# 	parser.add_argument('--singlefold', action="store_true", help="Only perform 1 fold of CV") #TODO could be a new "kernel"
# 	parser.add_argument('--noninductive', action="store_true", help="Noninductive version of xNetMF") #TODO could be a new "kernel"
# 	parser.add_argument('--transductive', action="store_true", help="Perform transductive classification: precompute full kernel matrix")
# 	return parser.parse_args()

def classification(args, train_data, train_labels, test_data, test_labels, kernel = True):
	#Train classifier
	before_train = time.time()
	clf = OneVsRestClassifier(svm.LinearSVC(random_state=1, C = args["svmc"]))
	clf.fit(train_data, train_labels)
	after_train = time.time()
	print("Trained model in time", after_train - before_train)

	before_test = time.time()
	test_predictions = clf.predict(test_data)
	after_test = time.time()
	print("Made test predictions in time", after_test - before_test)

	#Score predictions against ground truth
	acc = accuracy_score(test_labels, test_predictions)
	return acc

def main(args):
	#Load graph data
	graphs = read_combined(dataset_lookup[args["dataset"]])
	graph_labels = np.asarray([g.graph_label for g in graphs])

	embs = []
	time_pnts = np.linspace(0,100,25)
	for i in range(len(graphs)):
		mat = (graphs[i]).adj
		if args["prox_option"] == "PPMI":
			logger.info("Proximity option: PPMI")
			matrix = direct_compute_deepwalk_matrix(mat, args, logger)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "heat_kernel":
			logger.info("Proximity option: heat_kernel")
			graph = nx.Graph(mat)
			matrix, _ = heat_diffusion_ind(graph, args, logger)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "FaBP":
			logger.info("Proximity option: FaBP")
			matrix = belief_propgation(mat, args, logger)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "PPR":
			logger.info("Proximity option: PPR")
			matrix = PPR(mat, args, logger)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "inv_lap":
			logger.info("Proximity option: inv_lap")
			matrix = inv_lap(mat, args, logger)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "A2":
			logger.info("Proximity option: A2")
			matrix = A_square(mat, logger, args)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		elif args["prox_option"] == "RWTM":
			logger.info("Proximity option: RWTM")
			matrix = RWTM(mat, logger, args)
			try:
				t = args["embed_params"]["time_pnts"]
			except KeyError:
				raise MissingParamError
			time_pnts = np.linspace(t[0], t[1], t[2])
			chi = charac_function_multiscale(matrix, time_pnts, logger)
			matrix = chi
			embs.append(np.mean(chi, axis=0))
		else:
			raise WrongProxOptionError

	embs = np.array(embs)
	features = embs

	#Determine folds and classify (will need to perform embedding on a per-fold basis for inductive learning)
	np.random.seed(args["randomseed"])
	fold_order = np.random.permutation(np.arange(len(graphs)))
	fold_size = int(len(graphs) / args["numfolds"])
	remainder = len(graphs) % args["numfolds"]
	fold_start = 0
	fold_acc = list()

	n_folds_to_perform = args["numfolds"]

	for fold in range(n_folds_to_perform):
		'''Split into training (all but current fold) and test data (current fold)'''
		num_test = fold_size
		if fold < remainder: 
			num_test += 1 #so add one more test data point to each of first remainder folds
		num_train = len(graphs) - num_test

		print(fold_start,num_test)
		train_fold_numbers = list(range(fold_start)) + list(range(fold_start + num_test, len(graphs)))
		test_fold_numbers = list(range(fold_start, fold_start + num_test))
		train_indices = fold_order[train_fold_numbers]
		test_indices = fold_order[test_fold_numbers]
		fold_start += num_test

		train_features = features[train_indices]
		train_labels = graph_labels[train_indices]
		test_features = features[test_indices]
		test_labels = graph_labels[test_indices]
		acc = classification(args, train_features, train_labels, test_features, test_labels)

		print("Accuracy score for fold %d: %0.3f" % (fold + 1, acc))
		fold_acc.append(acc)
		avg_acc = sum(fold_acc) / len(fold_acc)
		print("Average accuracy across folds 1-%d: %0.3f" % (fold + 1, avg_acc))

if __name__ == "__main__":
	# if len(sys.argv) < 2:
	# 	print("setting sys.argv", sys.argv)
	# 	sys.argv = "main.py --dataset mutag --emb eigenvector --wliter 2".split()
	# args = parse_args()
	# if args.emb == "eigenvector": args.dimensionality = 6 #to match AAAI 2017
	# print(args)
	# main(args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str,
            help=".json file containing all the parameters")
    with open(parser.parse_args().param) as f:
        args = json.load(f)
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp
    main(args)