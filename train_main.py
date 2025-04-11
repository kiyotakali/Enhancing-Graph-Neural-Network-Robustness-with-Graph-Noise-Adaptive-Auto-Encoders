import os
import time
import argparse
import numpy as np
import scipy
import torch
from scipy.sparse import csr_matrix
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.NRGNN import NRGNN
from models.NRGNN_subgraph import NRGNN_subgraph
from models.NRGNN_subgraph_cleaner import NRGNN_subgraph_cleaner
from models.NRGNN_subgraph_cleaner_new import NRGNN_subgraph_cleaner_new
from models.NRGNN_subgraph_cleaner_new1 import NRGNN_subgraph_cleaner_new1
from dataset import Dataset
import os
import pandas as pd
from utils import init_gpuseed
from models.GCN import GCN
from utils import noisify_with_P
from models.Coteaching import Coteaching
from models.CP import CPGCN
# from models.forward import forward


from models.S_model import S_model


import pandas as pd
import random
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    if args.dataset == 'ogbn-arxiv':
        print('dataset','ogbn-arxiv')
        # adj, features, labels, idx_train, idx_val, idx_test, data, dataset_arxiv = ogbn_dataset('ogbn-arxiv')
        # ptb = args.ptb_rate
        # nclass = labels.max() + 1
        # ganjing = labels.copy()
        # train_labels = labels[idx_train]
        # noise_y, P = noisify_with_P(train_labels, nclass, ptb, 10, args.noise)
        # labels = torch.from_numpy(labels)
        # noise_labels = labels.clone()
        # noise_labels[idx_train] = torch.LongTensor(noise_y)
        # all_train = idx_train
    else:
        # data = Dataset(root='./data', name=args.dataset, train_rate=args.label_rate, seed=0)
        data = Dataset(root='./data', name=args.dataset)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
        # data = Dataset(root='/tmp/', name=args.dataset)
        # adj, features, labels = data.adj, data.features, data.labels
        # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        # all_train = idx_train
        nclass = labels.max() + 1
        ganjing = labels.copy()
        print(3, ganjing.shape)
        print(ganjing[idx_train])
        noise_train, P = noisify_with_P(labels[idx_train], nclass, args.ptb_rate, 0, args.noise)
        noise_val, _ = noisify_with_P(labels[idx_val], nclass, args.ptb_rate, 0, args.noise)
        noise_labels = labels.copy()
        noise_labels[idx_train] = noise_train
        noise_labels[idx_val] = noise_val

    print(args.model_type)
    print(device)
    name = args.dataset
    if args.model_type == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    self_loop=True,
                    dropout=args.dropout, device=device).to(device)
        print('adj',adj.shape)

        model.fit(name, features, adj, noise_labels, idx_train, ganjing, idx_val)
        return model.test(idx_test)
    # elif args.model_type == 'forward':
    #     model = forward(nfeat=features.shape[1],
    #                     nhid=args.hidden,
    #                     nclass=labels.max().item() + 1,
    #                     self_loop=True,
    #                     dropout=args.dropout,
    #                     device=device).to(device)
    #     model.fit(name, args.forward_type, features, adj, noise_labels, idx_train, idx_val)
    #     return model.test(idx_test)
    elif args.model_type == 'Coteaching':
        model = Coteaching(nfeat=features.shape[1],
                           nhid=args.hidden,
                           nclass=labels.max().item() + 1,
                           dropout=args.dropout, device=device).to(device)
        model.fit(name, features, adj, noise_labels, idx_train, idx_val)
        return model.test(idx_test)
    elif args.model_type == 'CPGCN':
        model = CPGCN(nfeat=features.shape[1],
                      nhid=args.hidden,
                      nclass=labels.max().item() + 1,
                      ncluster=labels.max().item() + 1,
                      dropout=args.dropout, device=device).to(device)
        model.fit(name, features, adj, noise_labels, noise_labels, idx_train, idx_val)
        return model.test(idx_test)
    elif args.model_type == 'S_model':
        model = S_model(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout, device=device).to(device)
        model.fit(name, features, adj, noise_labels, idx_train, idx_val)
        return model.test(idx_test)
    elif args.model_type == 'NRGNN':
        esgnn = NRGNN(args, device)
        esgnn.fit(features, adj, noise_labels, idx_train, idx_val)
        return esgnn.test(idx_test)

    elif args.model_type == 'NRGNN_subgraph':
        esgnn = NRGNN_subgraph(args, device)
        esgnn.fit(features, adj, noise_labels, idx_train, idx_val)
        return esgnn.test(idx_test)
    elif args.model_type == 'NRGNN_subgraph_cleaner':
        esgnn = NRGNN_subgraph_cleaner(args, device)
        esgnn.fit(features, adj, noise_labels, idx_train, idx_val)
        return esgnn.test(idx_test)
    elif args.model_type == 'NRGNN_subgraph_cleaner_new':
        esgnn = NRGNN_subgraph_cleaner_new(args, device)
        esgnn.fit(features, adj, noise_labels, idx_train, idx_val)
        return esgnn.test(idx_test)
    elif args.model_type == 'NRGNN_subgraph_cleaner_new1':
        esgnn = NRGNN_subgraph_cleaner_new1(args, device)
        esgnn.fit(features, adj, noise_labels, idx_train, idx_val)
        return esgnn.test(idx_test)

if __name__ == '__main__':
    # Training settings
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', action='store_true',
    #                     default=False, help='debug mode')
    # parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    # parser.add_argument('--gpu', help='used gpu id', default='5', type=str, required=False)
    # parser.add_argument('--dataset', type=str, default="cora",
    #                     choices=['cora', 'citeseer', 'pubmed', 'dblp', 'ogbn-arxiv', 'polblogs', 'cora_ml'],
    #                     help='dataset')
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
    # parser.add_argument('--ptb_rate', type=float, default=0.2,
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
    # parser.add_argument('--lr', type=float, default=0.01,
    #                     help='Initial learning rate.')
    # parser.add_argument('--alpha', type=float, default=0.03,
    #                     help='weight of loss of edge predictor')
    # parser.add_argument('--beta', type=float, default=1,
    #                     help='weight of the loss on pseudo labels')
    # parser.add_argument('--t_small', type=float, default=0.1,
    #                     help='threshold of eliminating the edges')
    # parser.add_argument('--model_type', type=str, default='GCN',
    #                     choices=['GCN', 'PLGCN', 'Coteaching', 'CPGCN', 'forward', 'S_model',
    #                              'NRGNN', 'NRGNNPlus', 'NRGNN_label_fix', 'NRGNN_subgraph',
    #                              'NRGNN_label_fix_pair', 'NRGNN_ogbn'],
    #                     help='type of model')
    # parser.add_argument('--forward_type', type=str, default='type1', choices=['type1', 'type2'],
    #                     help='type of forward_T type1 意味着先训练好模型再确定转移矩阵 type2 意味着过程中确定转移矩阵')
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', help='used gpu id', default='0', type=str, required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--seed', type=int, default=13, help='Random seed.')
    # 旧的
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--edge_hidden', type=int, default=64,
                        help='Number of hidden units of MLP graph constructor')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="cora",
                        choices=['cora', 'citeseer', 'pubmed', 'dblp', 'ogbn-arxiv'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.2,
                        help="noise ptb_rate")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--alpha', type=float, default=0.03,
                        help='weight of loss of edge predictor')
    parser.add_argument('--beta', type=float, default=1,
                        help='weight of the loss on pseudo labels')
    parser.add_argument('--t_small', type=float, default=0.1,
                        help='threshold of eliminating the edges')
    parser.add_argument('--p_u', type=float, default=0.8,
                        help='threshold of adding pseudo labels')
    parser.add_argument("--n_p", type=int, default=50,
                        help='number of positive pairs per node')
    parser.add_argument("--n_n", type=int, default=50,
                        help='number of negitive pairs per node')
    parser.add_argument("--label_rate", type=float, default=0.05,
                        help='rate of labeled data')
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')
    parser.add_argument('--model_type', type=str, default='NRGNN',
                        choices=['GCN', 'PLGCN', 'Coteaching', 'CPGCN', 'forward', 'S_model',
                                 'NRGNN', 'NRGNNPlus', 'NRGNN_label_fix', 'NRGNN_subgraph',
                                 'NRGNN_label_fix_pair', 'NRGNN_ogbn','NRGNN_subgraph_cleaner','NRGNN_subgraph_cleaner_new','NRGNN_subgraph_cleaner_new1'],
                        help='type of model')
    parser.add_argument('--forward_type', type=str, default='type1', choices=['type1', 'type2'],
                        help='type of forward_T type1 意味着先训练好模型再确定转移矩阵 type2 意味着过程中确定转移矩阵')
    args = parser.parse_known_args()[0]

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print(device)

    # args.dataset = 'citeseer'
    # args.noise = 'uniform'

    # if args.dataset == 'citeseer':
    #     if args.noise == 'pair':
    #         args.pre_lr = 1
    #         args.pre_weight_decay = 5e-3
    #         args.nll_lr = 2
    #         args.nll_weight_decay = 1e-2

    # if args.dataset == 'cora':
    #     if args.noise == 'uniform':
            # 0
            # args.mask_rate = 0.8
            # args.mask_times = 20
            # args.pre_lr = 1
            # args.pre_weight_decay = 1e-3
            # args.nll_lr = 2
            # args.nll_weight_decay = 5e-3

            # args.pre_lr = 0.05
            # args.pre_weight_decay = 5e-3
            # args.nll_lr = 2
            # args.nll_weight_decay = 5e-3
            # 4
            # args.pre_lr = 2
            # args.pre_weight_decay = 1e-3
            # args.nll_lr = 2
            # args.nll_weight_decay = 5e-3
            # args.pre_lr = 2
            # args.pre_weight_decay = 5e-3
            # args.nll_lr = 2
            # args.nll_weight_decay = 5e-3
        # if args.noise == 'pair':
        #     args.mask_rate = 0.8
        #     args.mask_times = 20
            # 0
            # args.pre_lr = 1
            # args.pre_weight_decay = 5e-3
            # args.pre_lr = 1
            # args.pre_weight_decay = 5e-2
            # args.pre_lr = 1
            # args.pre_weight_decay = 1e-3
            # args.nll_lr = 1
            # args.nll_weight_decay = 1e-2
            # args.pre_lr = 0.05
            # args.pre_weight_decay = 5e-3
            # args.nll_lr = 2
            # args.nll_weight_decay = 5e-3
    print(args)
    acc = []
    time_list = []
    # for seed in [13, 8, 1536, 3096, 8192]:
    # for seed in [11, 12, 13, 14, 15]:
    # main(args=args)
    for seed in range(11,14):
        t_s = time.time()
        args.seed = seed
        print(f"------------------seed{args.seed}------------------")
        acc.append(main(args=args))
        t_total = time.time()
        time_list.append(t_total - t_s)
    # for pre_lr in [0.01, 0.1, 0.5, 1, 2, 3, 4]:
    #     for pre_weight_decay in [5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
    #         print(pre_lr, "++++", pre_weight_decay)
    #         args.seed = 2
    # # for mask_times in [5, 10, 15, 20, 25]:
    # #         args.seed = 1
    # #         args.mask_time = mask_times
    # #         print(pre_lr, "++++", pre_weight_decay)
    # #         args.pre_lr = pre_lr
    # #         args.pre_weight_decay = pre_weight_decay
    #         args.nll_lr = pre_lr
    #         args.nll_weight_decay = pre_weight_decay
    #         print(f"------------------seed{args.seed}------------------")
    #         acc.append(main(args=args).cpu())
    print(acc)
    print("mean: {:.2f}".format(np.mean(acc) * 100))
    print("mean: {:.2f}".format(np.std(acc) * 100))
    print(time_list)
    print("mean: {:.2f}".format(np.mean(time_list) ))
    print("mean: {:.2f}".format(np.std(time_list) ))