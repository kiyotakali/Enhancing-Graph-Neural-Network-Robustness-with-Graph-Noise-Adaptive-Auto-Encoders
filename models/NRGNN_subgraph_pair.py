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

class NRGNN:
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

        # Train model
        t_total = time.time()

        model = Net(1433, 7).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        graph1 = (features, edge_index, labels)
        for epoch in range(200):
            # for step, graph in enumerate(graph1):
            for step, (x, e, y) in enumerate(zip(features, edge_index, labels)):
                # x, edge_index, y = getInput(graph)
                # x = torch.from_numpy(x).float().to(self.device)
                # edge_index = torch.from_numpy(edge_index).long().to(self.device)
                # y[y < 0] = 0
                # y = torch.from_numpy(y).long().to(self.device)
                
                model.train()
                optimizer.zero_grad()

                # Get output
                out = model(x, edge_index)   # (N, 2)
                out = out.to(self.device)
                # Get loss
                loss = F.cross_entropy(out, y)

                # Backward
                loss.backward()
                optimizer.step()
                
                # Get predictions and calculate training accuracy
                _, pred = out.cpu().detach().max(dim=-1)  # (N)
                self.labels = self.labels.cpu().detach()
                correct = float(pred.eq(y).sum().item())
                acc = correct / pred.shape[0]
                print('[Epoch {}/200, step {}/400] Loss {:.4f}, train acc {:.4f}'.format(epoch, step, loss.cpu().detach().data.item(), acc))

            # Evaluation on test data every 10 epochs
            if (epoch+1) % 10 == 0:
                model.eval()
                print('Accuracy: {:.4f}'.format(evalModel(model, self.dataset[400:])))
                t_total = time.time()

        for epoch in range(args.epochs):
            print(epoch)
            self.train(epoch, features, e, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)
        self.predictor.load_state_dict(self.predictor_model_weigths)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train(self, epoch, features, edge_index, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.predictor.train()
        self.optimizer.zero_grad()

        # obtain representations and rec loss of the estimator
        representations, rec_loss = self.estimator(edge_index,features)
        
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
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        
        # loss of GCN classifier
        loss_gcn = F.cross_entropy(output[idx_train], self.labels[idx_train])

        total_loss = loss_gcn + loss_pred + self.args.alpha * rec_loss  + self.args.beta * loss_add

        print(epoch,total_loss)
        print(1,loss_gcn)
        print(2,loss_pred)
        print(3,rec_loss)
        print(4,loss_add)

        total_loss.backward()



        self.optimizer.step()

        acc_train = accuracy(output[idx_train].detach(), self.labels[idx_train])

        # Evaluate validation set performance separately,
        self.model.eval()
        self.predictor.eval()
        pred = F.softmax(self.predictor(features,pred_edge_index,predictor_weights),dim=1)
        output = self.model(features, model_edge_index, estimated_weights.detach())

        acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
        acc_val = accuracy(output[idx_val], self.labels[idx_val])

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


class AggrSum(nn.Module):
    def __init__(self):
        super(AggrSum, self).__init__()
    
    def forward(self, H, X_node, node_num):
        # H : (N, s) -> (V, s)
        # X_node : (N, )
        # print(X_node.shape, X_node.dtype)
        # print(torch.isnan(X_node).any())
        # print(torch.isinf(X_node).any())

        # mask = torch.stack([X_node] * node_num, 0)
        mask = torch.cat([X_node] * node_num, 0)

        # mask = mask.float() - torch.unsqueeze(torch.range(0,node_num-1).float(), 1)
        # mask = mask.float() - torch.unsqueeze(torch.arange(0,node_num).float())
        mask = mask.float() - torch.unsqueeze(torch.arange(0,node_num, device='cuda:0').float(), dim=1)
        mask = (mask == 0).float()
        # (V, N) * (N, s) -> (V, s)
        return torch.mm(mask, H)

class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.aggregation = AggrSum()
        
    def forward(self, x, edge_index):
        # Add self-connect edges
        edge_index = self.addSelfConnect(edge_index, x.shape[0])
        node_num = x.shape[0]
        
        # Apply linear transform
        x = self.linear(x)
        
        # Normalize message
        row, col = edge_index
        deg = self.calDegree(row, x.shape[0]).float()
        deg_sqrt = deg.pow(-0.5)  # (N, )
        norm = deg_sqrt[row] * deg_sqrt[col]
        
        # Node feature matrix
        tar_matrix = torch.index_select(x, dim=0, index=col)
        tar_matrix = norm.view(-1, 1) * tar_matrix  # (E, out_channel)
        # Aggregate information
        aggr =  self.aggregation(tar_matrix, row, node_num)  # (N, out_channel)
        return aggr
        
    def calDegree(self, edges, num_nodes):
        # ind, deg = np.unique(edges.cpu().numpy(), return_counts=True)
        # deg_tensor = torch.zeros((num_nodes, ), dtype=torch.long)
        # deg_tensor[ind] = torch.from_numpy(deg)
        # return deg_tensor.to(edges.device)
            # Transfer edges and num_nodes to device
        device = torch.device('cuda:0') 
        edges = edges.to(device)
        ind, deg = np.unique(edges.cpu().numpy(), return_counts=True)
        # Specify the device for deg_tensor
        deg_tensor = torch.zeros((num_nodes, ), dtype=torch.long, device=device)
        # deg_tensor[ind] = torch.from_numpy(deg)
        deg_tensor[ind] = torch.from_numpy(deg).to(device)
        # Transfer deg_tensor to edges.device
        return deg_tensor.to(device)
    
    def addSelfConnect(self, edge_index, num_nodes):
        selfconn = torch.stack([torch.range(0, num_nodes-1, dtype=torch.long)]*2,
                               dim=0).to(edge_index.device)
        return torch.cat(tensors=[edge_index, selfconn],
                         dim=1)
    
class Net(nn.Module):
    def __init__(self, feat_dim, num_class):
        super(Net, self).__init__()
        self.conv1 = GCNConv(feat_dim, 16)
        self.conv2 = GCNConv(16, num_class)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.softmax(x, dim=-1)
    
def getInput(graph):
    embedding = np.diag(np.ones((1433)))
    graph = graph.cpu().numpy()
    print(graph[0])
    print(graph[1])
    print(graph[2])
    x = embedding[[i-1 for i in graph[0]]]
    edge_index = np.array([np.array([i,j]) for i,j in graph[1]]).T  # (2, E)
    edge_index = np.concatenate([edge_index] * 2, axis=1)   # (2, 2*E)
    y = np.array(graph[2])
    return x, edge_index, y
    # print(graph[0])
    # print(graph[1])
    # print(graph[2])
    # embedding = np.diag(np.ones((1433)))
    # print(type(graph[0]))
    # x = embedding[[i-1 for i in graph[0]]]
    # x = graph.feature
    # edge_index = graph.edge_index
    # y=graph.labels
    # x = embedding[[max(0, min(hash(i)-1, 1432)) for i in graph[0].split()]]
    # edge_index = np.array([np.array([i,j]) for i,j in graph[1]]).T  # (2, E)
    # 使用 hash 函数
    # edge_index = np.array([np.array([hash(i), hash(j)]) for i,j in eval(graph[1])]).T  # (2, E)
    # edge_index = np.concatenate([edge_index] * 2, axis=1)   # (2, 2*E)
    # y = np.array(graph[2])
    return x, edge_index, y

def evalModel(model, dataset):
    for graph in dataset:
        x, edge_index, y = getInput(graph)
        x = torch.from_numpy(x).float()
        edge_index = torch.from_numpy(edge_index).long()
        y[y < 0] = 0
        y = torch.from_numpy(y).long()
        
        acc_list = []
        _, pred = model(x, edge_index).max(dim=1)
        acc_list.append(float(pred.eq(y).sum().item())/y.shape[0])
    return sum(acc_list)/ len(acc_list)

