import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn
import scipy.io as sio
import pdb
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def load_data2(args):
    dataset = args.dataset
    metapaths = args.metapaths_list
    if (metapaths[0] == 'cited') or (metapaths[0]=='citing'):
        target = 'p'
    else:
        target = metapaths[0][0]
    sc = args.sc
    if dataset == 'acm':
        load_prefix = '../arga/ACM/preprocess/'
        metapaths =  [(0,1,0),(0,2,0)]
    #     features[features>1]=1
        labels = np.load(load_prefix + 'truth.npy')
        num_classes = len(set(labels))
        N = labels.shape[0]  # nodesize
        features = sp.load_npz(load_prefix + 'node_features.npz')
        rownetworks = []
        for mp in metapaths:
            adj = sp.load_npz(load_prefix + '-'.join(map(str, mp)) + '_normalize_adj.npz')
            adj[adj>0] = 1
            rownetworks.append(sp.csr_matrix(adj)+0*args.sc * sp.eye(N))
#         features = sp.load_npz(load_prefix + 'all_features.npz')
#         adj = sp.load_npz(load_prefix + 'all_adj.npz')
#         rownetworks = [adj]
        if sp.issparse(features):
            truefeatures = features.toarray()
        else:
            truefeatures = features
        truefeatures = preprocess_features(features)

    elif dataset == 'dblp':
        load_prefix = '../arga/DBLP/preprocess/'
        metapaths =  [(0,1,0),(0,1,2,1,0),(0,1,3,1,0)]
        features = sp.load_npz(load_prefix + 'node_features.npz')
        labels = np.load(load_prefix + 'truth.npy')
        num_classes = len(set(labels))
        N = labels.shape[0]  # nodesize
        rownetworks = []
        for mp in metapaths:
            adj = sp.load_npz(load_prefix + '-'.join(map(str, mp)) + '_normalize_adj.npz')
            adj[adj>0] = 1
            rownetworks.append(sp.csr_matrix(adj)+0*args.sc * sp.eye(N))

        if sp.issparse(features):
            truefeatures = features.toarray()
        else:
            truefeatures = features
        truefeatures = preprocess_features(features)
        
    elif dataset == 'yelp':
         # business 0, user 1, star 2, level 3
        load_prefix = '../arga/Yelp/preprocess/'
        metapaths = [(0,2,0),(0,1,0)]
        features = sp.load_npz(load_prefix + 'node_features.npz')
        labels = np.load(load_prefix + 'truth.npy')
        num_classes = len(set(labels))
        N = labels.shape[0]  # nodesize
        rownetworks = []
        for mp in metapaths:
            adj = sp.load_npz(load_prefix + '-'.join(map(str, mp)) + '_normalize_adj.npz')
            adj[adj>0] = 1
            rownetworks.append(sp.csr_matrix(adj))

        if sp.issparse(features):
            truefeatures = features.toarray()
        else:
            truefeatures = features
        truefeatures = preprocess_features(features)

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(list(range(N)), labels, test_size=0.6, random_state=0)

    truefeatures = sp.lil_matrix(truefeatures)
    nb_classes = len(set(train_y))
    y = indices_to_one_hot(labels, nb_classes)

    return rownetworks, truefeatures, y, train_x, train_x, test_x, labels


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
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

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def row_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return sparse_mx_to_torch_sparse_tensor(mx)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    node_size = sparse_mx.shape[0]
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    edge_weight = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(edge_index, edge_weight, shape)

def process_adj_gat(adj):
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # Tricky implementation of official GAT
    adj = (adj + sp.eye(adj.shape[0])).todense()
    for x in range(0, adj.shape[0]):
        for y in range(0, adj.shape[1]):
            if adj[x, y] == 0:
                adj[x, y] = -9e15
            elif adj[x, y] >= 1:
                adj[x, y] = 0
            else:
                print(adj[x, y], 'error')
    adj = torch.FloatTensor(np.array(adj))
    # adj = sp.coo_matrix(adj)
    return adj