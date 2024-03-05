import torch.autograd as autograd

# 在训练过程中启用异常检测
autograd.set_detect_anomaly(True)

import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.GCN import GCN
from utils import accuracy,sparse_mx_to_torch_sparse_tensor
import networkx as nx
import random
from sklearn.semi_supervised import LabelSpreading

class NRGNN_subgraph_cleaner_new:
    def __init__(self, args, device):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None

    def fit(self, features, adj, labels, idx_train, idx_val):

        args = self.args

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.edge_index = edge_index
        self.features = features
        self.labels = labels
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)

        self.predictor = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        self.model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         self_loop=True,
                         dropout=self.args.dropout, device=self.device).to(self.device)

        self.estimator = EstimateAdj(features.shape[1], args, idx_train ,device=self.device).to(self.device)
        
        # obtain the condidate edges linking unlabeled and labeled nodes
        self.pred_edge_index = self.get_train_edge(edge_index,features,self.args.n_p ,idx_train)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.estimator.parameters())+ list(self.predictor.parameters()),
                               lr=args.lr, weight_decay=args.weight_decay)
        print(features.shape)
        print(labels.shape)

        # 把PyTorch张量复制成一个新的张量
        features_copy = features.clone()
        # 把复制的张量转换成NumPy数组
        features_copy = features_copy.cpu().numpy()

        labels_copy = labels.clone()
        labels_copy = labels_copy.cpu().numpy()

        models = LabelSpreading(kernel='rbf', gamma=0.1, alpha=0.2) # 使用径向基函数核和平滑参数
        models.fit(features_copy, labels_copy) # 使用所有的数据来训练模型

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            print(epoch)
            self.train(epoch, features, edge_index, idx_train, idx_val,models)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weigths)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train(self, epoch, features, edge_index, idx_train, idx_val, model):
        torch.cuda.memory_summary()
        torch.cuda.empty_cache()
        args = self.args

        t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()
        nodes = idx_train
        new_labels = self.labels.clone()


        # # obtain representations and rec loss of the estimator
        # representations, rec_loss = self.estimator(edge_index,features)
        
        # train_length = len(idx_train)
        # idx_mask = idx_train[train_length:]
        # mask_nodes = idx_mask
        # # prediction of accurate pseudo label miner
        # predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index,representations)
        # pred_edge_index = torch.cat([edge_index,self.pred_edge_index],dim=1)
        # predictor_weights = torch.cat([torch.ones([edge_index.shape[1]],device=self.device),predictor_weights],dim=0)

        # log_pred = self.predictor(features,pred_edge_index,predictor_weights)

        # define some parameters
        degree_threshold = 10 # the degree threshold for dropout
        dropout_prob = 0.2 # the dropout probability

        # define a parameter
        similarity_threshold = 0.8 # the similarity threshold for adding edge

        # obtain representations and rec loss of the estimator
        representations, rec_loss = self.estimator(edge_index,features)
        
        # dropout high-degree nodes
        # degree = get_degree(edge_index) # get degree of each node
        # ones = torch.ones(degree.shape[0])
        # ones = ones.to(self.device)
        # features = dropout_by_degree(features, degree, degree_threshold, dropout_prob, ones) # dropout features of high-degree nodes


        train_length = len(idx_train)
        idx_mask = idx_train[train_length:]
        mask_nodes = idx_mask

        # add edge for low-degree nodes based on similarity
        pred_edge_index = add_edge_by_similarity(self.pred_edge_index, representations, similarity_threshold)

        # prediction of accurate pseudo label miner
        predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index,representations)
        pred_edge_index = torch.cat([edge_index,self.pred_edge_index],dim=1)
        predictor_weights = torch.cat([torch.ones([edge_index.shape[1]],device=self.device),predictor_weights],dim=0)

        log_pred = self.predictor(features,pred_edge_index,predictor_weights)
        # obtain accurate pseudo labels and new candidate edges
        if self.best_pred == None:
            
            pred = F.softmax(log_pred,dim=1).detach()
            self.best_pred = pred
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
        else:
            pred = self.best_pred
        
        # 预测标签
            
        # 把PyTorch张量复制成一个新的张量
            
        features_copy = features.clone()
        # 把复制的张量转换成NumPy数组
        features_copy = features_copy.cpu().numpy()

        labels_copy = new_labels.clone()
        labels_copy = labels_copy.cpu().numpy()
        pred_labels = model.predict(features_copy) # 使用当前的模型来预测标签
        # 更新标签
        new_label = torch.tensor(pred_labels) # 将预测标签转换为张量
        new_label = new_label.to(self.device)
        new_label[idx_train] = new_labels[idx_train] # 保持已标记节点的标签不变
        # 拼接标签
        # new_label = torch.cat([self.labels, new_label], dim=0) # 将原始标签和新标签拼接起来
        
        new_idx_train = torch.arange(len(new_label)) # 生成一个与new_label长度相同的索引序列
        diff_idx = torch.nonzero(new_label != self.labels) # 找出new_label中与self.labels不同的位置
        idx_train = torch.from_numpy(idx_train)
        idx_train = idx_train.to(self.device)
        # print(idx_train.shape)
        # print(diff_idx.shape)
        diff_idx = diff_idx.squeeze(1)
        # print(idx_train.shape)
        # print(diff_idx.shape)
        new_idx_train = torch.cat([idx_train, diff_idx],dim=0) # 拼接idx_train和这些位置，作为new_idx_train
        # print(new_idx_train.shape)
        new_idx_train = new_idx_train.long()
        # print(new_label.shape)

        new_label = torch.cat([self.labels, new_label], dim=0) # 将原始标签和新标签拼接起来

        # prediction of the GCN classifier
        estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index,representations)
        estimated_weights = torch.cat([predictor_weights, estimated_weights],dim=0)
        model_edge_index = torch.cat([pred_edge_index,self.unlabel_edge_index],dim=1)
        output = self.model(features, model_edge_index, estimated_weights)
        pred_model = F.softmax(output,dim=1)

        eps = 1e-8
        pred_model = pred_model.clamp(eps, 1-eps)

        # loss from pseudo labels
        loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
        
        # loss of accurate pseudo label miner
        # loss_pred = F.cross_entropy(log_pred[new_idx_train], self.labels[new_idx_train])
        loss_pred = F.cross_entropy(log_pred[new_idx_train], new_label[new_idx_train])
        
        # loss of GCN classifier
        # loss_gcn = F.cross_entropy(output[new_idx_train], self.labels[new_idx_train])
        loss_gcn = F.cross_entropy(output[new_idx_train], new_label[new_idx_train])

        total_loss = loss_gcn + loss_pred + self.args.alpha * rec_loss  + self.args.beta * loss_add

        print(epoch,total_loss)
        print(1,loss_gcn)
        print(2,loss_pred)
        print(3,rec_loss)
        print(4,loss_add)
        # torch.cuda.empty_cache()
        total_loss.backward()



        self.optimizer.step()

        # acc_train = accuracy(output[new_idx_train].detach(), self.labels[new_idx_train])
        acc_train = accuracy(output[new_idx_train].detach(), new_label[new_idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        self.predictor.eval()
        pred = F.softmax(self.predictor(features,pred_edge_index,predictor_weights),dim=1)
        output = self.model(features, model_edge_index, estimated_weights.detach())

        # acc_pred_val = accuracy(pred[new_idx_train], self.labels[new_idx_train])
        # acc_val = accuracy(output[new_idx_train], self.labels[new_idx_train])
        acc_pred_val = accuracy(pred[new_idx_train], new_label[new_idx_train])
        acc_val = accuracy(output[new_idx_train], new_label[new_idx_train])
        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
            self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = estimated_weights.detach()
            self.best_model_index = model_edge_index
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: {:.4f}'.format(self.best_val_acc.item()))

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_pred: {:.4f}'.format(loss_pred.item()),
                      'loss_add: {:.4f}'.format(loss_add.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'acc_pred_val: {:.4f}'.format(acc_pred_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))
                print('Size of add idx is {}'.format(len(self.idx_add)))


    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        labels = self.labels

        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = torch.cat([self.edge_index,self.pred_edge_index],dim=1)
        output = self.predictor(features, pred_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tPredictor results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        self.model.eval()
        estimated_weights = self.best_graph
        model_edge_index = self.best_model_index
        output = self.model(features, model_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("\tGCN classifier results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)
    

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None

        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
                _,rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,num_nodes = len(features)).to(self.device)

        return poten_edges

    def get_model_edge(self, pred):

        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u)]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
        mask = (row!=col)
        unlabel_edge_index = torch.stack([row[mask],col[mask]], dim=0)

        return unlabel_edge_index, idx_add
                        
#%%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, args, idx_train ,device='cuda'):
        super(EstimateAdj, self).__init__()
        
        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.representations = 0

    def forward(self, edge_index, features):

        representations = self.estimator(features,edge_index,\
                                        torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        return representations,rec_loss
    
    def get_estimated_weigths(self, edge_index, representations):

        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)
        # 牛逼
        # estimated_weights = F.relu(output.clone())
        estimated_weights = F.relu(output).detach()

        estimated_weights[estimated_weights < self.args.t_small] = 0.0

        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 

        return rec_loss

def add_edge_by_similarity(edge_index, representations, threshold):
    # edge_index is a 2D tensor of shape [2, E], where E is the number of edges
    # representations is a 2D tensor of shape [N, D], where D is the dimension of representation
    # threshold is a scalar, indicating the similarity threshold for adding edge
    # return a 2D tensor of shape [2, E + M], where M is the number of added edges
    similarity = torch.matmul(representations, representations.t()) # compute similarity matrix of shape [N, N]
    candidates = torch.nonzero(similarity > threshold) # find candidate pairs of shape [M, 2]
    new_edge_index = torch.cat([edge_index, candidates.t()], dim=1) # concatenate original and new edge index
    return new_edge_index

    # edge_index is a 2D tensor of shape [2, E], where E is the number of edges
    # representations is a 2D tensor of shape [N, D], where D is the dimension of representation
    # threshold is a scalar, indicating the similarity threshold for adding edge
    # return a 2D tensor of shape [2, E + M], where M is the number of added edges
    # cos = torch.nn.CosineSimilarity(dim=0) # create a cosine similarity object
    # similarity = cos(representations.unsqueeze(0), representations.unsqueeze(1)) # compute similarity matrix of shape [N, N]
    # row, col = torch.where(similarity > threshold) # find candidate pairs of shape [M]
    # candidates = torch.stack([row, col], dim=0) # reshape candidates to shape [2, M]
    # new_edge_index = torch.cat([edge_index, candidates], dim=1) # concatenate original and new edge index
    # new_edge_index = torch.unique(new_edge_index, dim=1) # remove duplicate edges
    # return new_edge_index

    # edge_index is a 2D tensor of shape [2, E], where E is the number of edges
    # features is a 2D tensor of shape [N, F], where F is the number of features
    # labels is a 1D tensor of shape [N], where N is the number of nodes
    # representations is a 2D tensor of shape [N, D], where D is the dimension of representation
    # threshold is a scalar, indicating the similarity threshold for adding edge
    # return a 2D tensor of shape [2, E + M], where M is the number of added edges
    # cos = torch.nn.CosineSimilarity(dim=0) # create a cosine similarity object
    # similarity = cos(representations.unsqueeze(0), representations.unsqueeze(1)) # compute similarity matrix of shape [N, N]
    # row, col = torch.where(similarity > threshold) # find candidate pairs of shape [M]
    # candidates = [] # initialize the list of candidates
    # # create a Data object from edge index, features and labels
    # data = Data(edge_index=edge_index, x=features, y=labels)
    # # convert the Data object to a networkx object
    # G = utils.to_networkx(data)
    # for i in range(len(row)): # loop through the candidate pairs
    #     node1 = row[i].item() # get the first node index
    #     node2 = col[i].item() # get the second node index
    #     if node1 != node2: # check if they are different
    #         unconnected = find_unconnected_neighbors(G, node1, node2) # find their unconnected neighbors
    #         if unconnected is not None: # if there are any unconnected neighbors
    #             candidates.append(unconnected) # add them to the list
    # candidates = torch.tensor(candidates).t() # convert the list to a tensor and transpose it
    # new_edge_index = torch.cat([edge_index, candidates], dim=1) # concatenate original and new edge index
    # new_edge_index = torch.unique(new_edge_index, dim=1) # remove duplicate edges
    # return new_edge_index

 

def get_degree(edge_index):
    # edge_index is a 2D tensor of shape [2, E], where E is the number of edges
    # return a 1D tensor of shape [N], where N is the number of nodes
    return torch.sum(edge_index, dim=0)



def dropout_by_degree(features, degree, threshold, p, ones):
    # features is a 2D tensor of shape [N, F], where F is the number of features
    # degree is a 1D tensor of shape [N], where N is the number of nodes
    # threshold is a scalar, indicating the degree threshold for dropout
    # p is a scalar, indicating the dropout probability
    # return a 2D tensor of shape [N, F], where some features are zeroed out
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # choose the device
    features = features.to(device) # move features to the device
    degree = degree.to(device) # move degree to the device
    prob = torch.rand(degree.shape[0], device=device) # generate random probability for each node
    mask = torch.where(degree > threshold, prob < p, ones) # generate mask for dropout
    mask = mask.unsqueeze(1) # reshape mask to [N, 1]
    mask = mask.expand(features.shape) # expand mask to match the features shape
    features = torch.mul(features, mask) # element-wise multiplication
    features[degree > threshold] = features[degree > threshold] / (1 - p) # rescale the features of high-degree nodes
    return features

def find_neighbors(G, node):
    # G is a networkx object
    # node is an integer, representing the node index
    # return a list of integers, representing the neighbor node indices
    return list(nx.neighbors(G, node))

def find_unconnected_neighbors(G, node1, node2):
    # G is a networkx object
    # node1 and node2 are integers, representing the node indices
    # return a tuple of integers, representing the unconnected neighbor node indices
    neighbors1 = find_neighbors(G, node1) # find the neighbors of node1
    neighbors2 = find_neighbors(G, node2) # find the neighbors of node2
    unconnected = [] # initialize the list of unconnected neighbor pairs
    for n1 in neighbors1: # loop through the neighbors of node1
        for n2 in neighbors2: # loop through the neighbors of node2
            if n1 != n2 and not G.has_edge(n1, n2): # check if they are different and not connected
                unconnected.append((n1, n2)) # add them to the list
    if len(unconnected) > 0: # if there are any unconnected neighbor pairs
        return random.choice(unconnected) # return a random pair
    else: # otherwise
        return None # return None