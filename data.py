import  numpy as np
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import  sys
import torch
from sklearn.preprocessing import OneHotEncoder


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def load_data(args):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}.npz".format(args.dataset), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(torch.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(args.dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if args.dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    label_v = [np.argmax(label) for label in labels]
    label_v = np.array(label_v)

    idx_train = range(int(len(graph.nodes)*0.1))
    # idx_clean = range(len(idx_train),len(idx_train)+args.clean_label_num)
    # idx_val = range(len(idx_train)+args.clean_label_num,int(len(graph.nodes)*0.8))
    idx_val = range(len(idx_train),int(len(graph.nodes)*0.2))
    idx_test = range(int(len(graph.nodes)*0.2),len(graph.nodes))
    #使用部分标签
    idx_train = idx_train[:int(0.05 * adj.shape[0])]


    data_list_clean = {}
    for j in range(labels.shape[1]):
        data_list_clean[j] = [i + int(len(graph.nodes)*0.1)  for i, label in enumerate(label_v[idx_val]) if label == j]
    # print(labels.shape,'labels')
    # print('data_list_clean',len(data_list_clean))

    # print(data_list_clean)
    # data_list_clean = {}
    list_clean = []
    num = int(args.clean_label_num / labels.shape[1])
    for i, ind in data_list_clean.items():
        np.random.shuffle(ind)
        list_clean.append(ind[0:num])
    idx_clean = np.array(list_clean)  
    idx_clean = idx_clean.flatten()

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    clean_mask = sample_mask(idx_clean, labels.shape[0])

    return adj, features, labels, train_mask, val_mask, test_mask,clean_mask,graph

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return sparse_to_tuple(features) # [coordinates, data, shape], []


def normalize_S(S):
    """Symmetrically normalize similarity matrix."""
    S = sp.coo_matrix(S)
    rowsum = np.array(S.sum(1)) # D
    for i in range(len(rowsum)):
        if rowsum[i] == 0:
            print("error")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return S.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5SD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_S(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)