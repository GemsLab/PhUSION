"""
This script generate a graph for struct imbedding, not finished
"""
import sys
sys.path.append('src')
import math
from src.utils.shape import *
from src.utils.utils import *
import argparse


def build_structure(width_basis, basis_type, list_shapes, start=0,
                    rdm_basis_plugins =False, add_random_edges=0,
                    plot=False, savefig=False):
    '''This function creates a basis (torus, string, or cycle)
    and attaches elements of the type in the list randomly along the basis.
    Possibility to add random edges afterwards.
    INPUT:
    --------------------------------------------------------------------------------------
    width_basis      :      width (in terms of number of nodes) of the basis
    basis_type       :      (torus, string, or cycle)
    shapes           :      list of shape list (1st arg: type of shape,
                            next args:args for building the shape,
                            except for the start)
    start            :      initial nb for the first node
    rdm_basis_plugins:      boolean. Should the shapes be randomly placed
                            along the basis (True) or regularly (False)?
    add_random_edges :      nb of edges to randomly add on the structure
    plot,savefig     :      plotting and saving parameters
    OUTPUT:
    --------------------------------------------------------------------------------------
    basis            :       a nx graph with the particular shape
    colors           :       labels for each role
    '''
    basis, role_id = eval(basis_type)(start, width_basis)
    n_basis, n_shapes = nx.number_of_nodes(basis), len(list_shapes)
    start += n_basis        # indicator of the id of the next node

    # Sample (with replacement) where to attach the new motives
    if rdm_basis_plugins is True:
        plugins = np.random.choice(n_basis, n_shapes, replace=False)
    else:
        spacing = math.floor(width_basis / n_shapes)
        plugins = [int(k * spacing) for k in range(n_shapes)]
    communities = [0] * n_basis
    seen_shapes = {'basis': [0, n_basis]}
    for p in plugins:
        role_id[p] += 1

    for shape_id, shape in enumerate(list_shapes):
        shape_type = shape[0]
        args = [start]
        if len(shape)>1:
            args += shape[1:]
        args += [0]
        graph_s, roles_graph_s = eval(shape_type)(*args)
        n_s = nx.number_of_nodes(graph_s)
        try:
            col_start = seen_shapes[shape_type][0]
        except:
            col_start = np.max(role_id) + 1
            seen_shapes[shape_type] = [col_start, n_s]
        # Attach the shape to the basis
        basis.add_nodes_from(graph_s.nodes())
        basis.add_edges_from(graph_s.edges())
        basis.add_edges_from([(start, plugins[shape_id])])
        role_id[plugins[shape_id]] += (-2 - 10 * seen_shapes[shape_type][0])
        communities += [shape_id] * n_s
        temp_labels = [r + col_start for r in roles_graph_s]
        temp_labels[0] += 100 * seen_shapes[shape_type][0]
        role_id += temp_labels
        start += n_s

    if add_random_edges > 0:
        # add random edges between nodes:
        for p in range(add_random_edges):
            src, dest = np.random.choice(nx.number_of_nodes(basis),
                                         2, replace=False)
            print (src, dest)
            basis.add_edges_from([(src, dest)])
    if plot is True: plot_networkx(basis, role_id)

    return basis, communities, plugins, role_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
            help="type of the graph")
    parser.add_argument("--output", type=str,
            help="output file location")
    args = parser.parse_args()