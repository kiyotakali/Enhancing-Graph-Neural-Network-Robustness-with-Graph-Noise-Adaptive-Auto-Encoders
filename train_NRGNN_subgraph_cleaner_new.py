import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.NRGNN_subgraph_cleaner_new import NRGNN_subgraph_cleaner_new
from dataset import Dataset
import os
import pandas as pd
from utils import init_gpuseed

import gc
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 300)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu', help='used gpu id', default='0', type=str, required=False)
parser.add_argument('--debug', action='store_true',
                    default=False, help='debug mode')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
#旧的
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    choices=['cora', 'citeseer','pubmed','dblp'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2,
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=0.03,
                    help='weight of loss of edge predictor')
parser.add_argument('--beta', type=float, default=1,
                    help='weight of the loss on pseudo labels')
parser.add_argument('--t_small',type=float, default=0.1,
                    help='threshold of eliminating the edges')
parser.add_argument('--p_u',type=float, default=0.8,
                    help='threshold of adding pseudo labels')
parser.add_argument("--n_p", type=int, default=50,
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50,
                    help='number of negitive pairs per node')
parser.add_argument("--label_rate", type=float, default=0.05,
                    help='rate of labeled data')
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                    help='type of noises')

#新的
# parser.add_argument('--dataset', type=str, default="cora",
#                     choices=['cora', 'citeseer', 'pubmed', 'dblp', 'ogbn-arxiv', 'polblogs', 'cora_ml'], help='dataset')
# parser.add_argument("--label_rate", type=float, default=0.05,
#                     help='rate of labeled data')
# parser.add_argument('--weight_decay', type=float, default=5e-3,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--edge_hidden', type=int, default=64,
#                     help='Number of hidden units of MLP graph constructor')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
# parser.add_argument('--ptb_rate', type=float, default=0.1,
#                     help="noise ptb_rate")
# parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
#                     help='type of noises')
# parser.add_argument('--p_u', type=float, default=0.8,
#                     help='threshold of adding pseudo labels')
# parser.add_argument("--n_p", type=int, default=50,
#                     help='number of positive pairs per node')
# parser.add_argument("--n_n", type=int, default=50,
#                     help='number of negitive pairs per node')
# parser.add_argument('--epochs', type=int, default=200,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.1,
#                     help='Initial learning rate.')
# parser.add_argument('--alpha', type=float, default=0.03,
#                     help='weight of loss of edge predictor')
# parser.add_argument('--beta', type=float, default=1,
#                     help='weight of the loss on pseudo labels')
# parser.add_argument('--t_small', type=float, default=0.1,
#                     help='threshold of eliminating the edges')




args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
print(device)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

print(args)
# np.random.seed(15) # Here the random seed is to split the train/val/test data

#%%
if args.dataset=='dblp':
    from torch_geometric.datasets import CitationFull
    import torch_geometric.utils as utils
    dataset = CitationFull('./data','dblp')
    adj = utils.to_scipy_sparse_matrix(dataset.data.edge_index)
    features = dataset.data.x.numpy()
    labels = dataset.data.y.numpy()
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    idx_test = idx[:int(0.8 * len(labels))]
    idx_val = idx[int(0.8 * len(labels)):int(0.9 * len(labels))]
    idx_train = idx[int(0.9 * len(labels)):int((0.9+args.label_rate) * len(labels))]
else:
    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
    

#%% add noise to the labels
from utils import noisify_with_P
ptb = args.ptb_rate
nclass = labels.max() + 1
train_labels = labels[idx_train]
noise_y, P = noisify_with_P(train_labels,nclass, ptb, 10, args.noise) 
noise_labels = labels.copy()
noise_labels[idx_train] = noise_y

# %%
import random
init_gpuseed(args.seed, device)
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)



esgnn = NRGNN_subgraph_cleaner_new(args,device)
esgnn.fit(features, adj, noise_labels, idx_train, idx_val)

print("=====test set accuracy=======")
esgnn.test(idx_test)
print("===================================")