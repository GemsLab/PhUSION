import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import scipy.sparse as sparse
import sys
import theano
import pdb
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


sys.path.append('../')
from util.shape import *
import matplotlib.pyplot as plt
import util.build_graph as build_graph
import util.struct2vec_graph_utils as struct2vec
from util.dist import analyze_dist
from embedd import direct_compute_deepwalk_matrix, heat_diffusion_ind, belief_propgation, HOPE, common_neighbor
from proximity import svd_deepwalk_matrix,charac_function_multiscale


def visualize_graphwave(chi, heat_print, taus, colors):
    nb_clust=len(np.unique(colors))
    pca=PCA(n_components=5)
    trans_data=pca.fit_transform(StandardScaler().fit_transform(chi))
    km=sk.cluster.KMeans(n_clusters=nb_clust)
    km.fit(trans_data)
    labels_pred=km.labels_
#     pdb.set_trace()
    ######## Params for plotting
    cmapx=plt.get_cmap('rainbow')
    x=np.linspace(0,1,np.max(labels_pred)+1)
    col=[cmapx(xx) for xx in x ]
    markers = {0:'*',1: '.', 2:',',3: 'o',4: 'v',5: '^',6: '<',7: '>',8: 3 ,9:'d',10: '+',11:'x',12:'D',13: '|',14: '_',15:4,16:0,17:1,18:2,19:6,20:7}
    ########

    for c in np.unique(colors):
        indc=[i for i,x in enumerate(colors) if x==c]
        #print indc
        plt.scatter(trans_data[indc,0], trans_data[indc,1],c=np.array(col)[list(np.array(labels_pred)[indc])] ,marker=markers[c%len(markers)],s=500)
    labels = colors
    for label,c, x, y in zip(labels,labels_pred, trans_data[:, 0], trans_data[:, 1]):
            plt.annotate(label,xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig('test.png')

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # 1- Start by defining our favorite regular structure

    width_basis = 30
    nbTrials = 20
    homs = []
    comps = []
    sils = []
    for trial in range(nbTrials):
        ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
        ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type

        ## 1. Choose the basis (cycle, torus or chain)
        basis_type = 'cycle' 

        ### 2. Add the shapes 
        nb_shapes = 5  ## numbers of shapes to add 
        #shape = ["fan",6] ## shapes and their associated required parameters  (nb of edges for the star, etc)
        #shape = ["star",6]
        # shape = ["fan",6]
        # list_shapes = [["house"]] * nb_shapes
        list_shapes = [["star", 6]] * nb_shapes

        ### 3. Give a name to the graph
        identifier = 'AA'  ## just a name to distinguish between different trials
        name_graph = 'houses' + identifier
        sb.set_style('white')

        ### 4. Pass all these parameters to the Graph Structure
        add_edges = 10 ## nb of edges to add anywhere in the structure
        del_edges  =0

        G, communities, plugins, role_id = build_graph.build_structure(width_basis, basis_type, list_shapes, start=0,
                                    rdm_basis_plugins =False, add_random_edges= add_edges,
                                    plot=False, savefig=True)
        # G , role_id = barbel_graph(0, 5, 5,plot=False)
        plot_networkx(G, role_id, True)
        plt.clf()
        TAUS = [1, 10, 25, 50]
        ORDER = 30
        # TAUS = range(19,21)
        PROC = 'approximate'
        heat_print = np.linspace(0,100,25)
        # matrix, taus = heat_diffusion_ind(G, TAUS, ORDER, PROC)
        taus = TAUS
        mat = np.array(nx.to_numpy_matrix(G))
        mat = sparse.csc_matrix(mat)
        # pdb.set_trace()
        # matrix = direct_compute_deepwalk_matrix(mat,window=10, b=1.0)
        # matrix = belief_propgation(mat)
        # matrix = HOPE(mat)
        matrix = common_neighbor(mat, False)
        matrix = {'0': matrix}
        # pdb.set_trace()
        # matrix = matrix[0]
        # matrix = {'0': matrix}
        chi = charac_function_multiscale(matrix, heat_print)
        mapping_inv={i: taus[i] for i in range(len(taus))}
        mapping={float(v): k for k,v in mapping_inv.items()}
        
        colors = role_id
        nb_clust = len(np.unique(colors))
        pca = PCA(n_components=5)

        trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
        km = sk.cluster.KMeans(n_clusters=nb_clust)
        km.fit(trans_data)
        labels_pred = km.labels_
        ######## Params for plotting
        cmapx = plt.get_cmap('rainbow')
        x = np.linspace(0,1,np.max(labels_pred) + 1)
        col = [cmapx(xx) for xx in x ]
        markers = {0:'*',1: '.', 2:',',3: 'o',4: 'v',5: '^',6: '<',7: '>',8: 3 ,\
                9:'d',10: '+',11:'x',12:'D',13: '|',14: '_',15:4,16:0,17:1,\
                18:2,19:6,20:7}
        ########

        for c in np.unique(colors):
                indc = [i for i, x in enumerate(colors) if x == c]
                #print indc
                plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                            c=np.array(col)[list(np.array(labels_pred)[indc])],
                            marker=markers[c%len(markers)], s=500)
        labels = colors
        for label,c, x, y in zip(labels,labels_pred, trans_data[:, 0], trans_data[:, 1]):
                    plt.annotate(label,xy=(x, y), xytext=(0, 0), textcoords='offset points')
        # plt.savefig('test.png')
                
        ami=sk.metrics.adjusted_mutual_info_score(colors, labels_pred)  
        sil=sk.metrics.silhouette_score(trans_data,labels_pred, metric='euclidean')
        ch=sk.metrics.calinski_harabaz_score(trans_data, labels_pred)
        hom=sk.metrics.homogeneity_score(colors, labels_pred) 
        comp=sk.metrics.completeness_score(colors, labels_pred)
        print ('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
        print (str(hom)+'\t'+str(comp)+'\t'+str(ami)+'\t'+str(nb_clust)+'\t'+str(ch)+'\t'+str(sil))
        homs.append(hom)
        comps.append(comp)
        sils.append(sil)
    homs = np.array(homs)
    comps = np.array(comps)
    sils = np.array(sils)
    print("Average Homogeneity:" + str(np.mean(homs)))
    print("Average Completeness:" + str(np.mean(comps)))
    print("Average Silhouette:" + str(np.mean(sils)))


