# 导入NetworkX库
import networkx as nx
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.NRGNN import NRGNN
from dataset import Dataset
import os
import pandas as pd
from utils import init_gpuseed

# 定义子图匹配的函数，输入为两个图，输出为一个布尔值，表示是否匹配
def subgraph_match(G1, G2):
    # 将图的特征矩阵和边矩阵转换为NetworkX的图对象
    G1 = nx.from_numpy_matrix(G1[1], create_using=nx.Graph)
    G2 = nx.from_numpy_matrix(G2[1], create_using=nx.Graph)
    # 为图的节点添加特征属性
    for i in range(G1.number_of_nodes()):
        G1.nodes[i]['feature'] = G1[0][i]
    for i in range(G2.number_of_nodes()):
        G2.nodes[i]['feature'] = G2[0][i]
    # 定义一个节点匹配的函数，用于判断两个节点的特征是否相等
    def node_match(n1, n2):
        return n1['feature'] == n2['feature']
    # 调用NetworkX的子图同构函数，返回一个生成器，包含所有的映射
    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match=node_match)
    # 如果生成器不为空，说明存在至少一个匹配，返回True，否则返回False
    return not GM.is_empty()

# 定义子图的数据，这里使用一个三角形作为子图
sub_graph = ([[1,0,0],[0,1,0],[0,0,1]], [[0,1,1],[1,0,1],[1,1,0]])

# 在train函数中，对每个输入的图进行子图匹配，修改标签
def train(epoch, features, edge_index, idx_train, idx_val):
    # 遍历所有的图
    for i in range(len(features)):
        # 获取当前图的特征矩阵和边矩阵
        feature = features[i]
        edge = edge_index[i]
        # 获取当前图的标签矩阵
        label = labels[i]
        # 调用子图匹配的函数，判断当前图是否包含子图
        if subgraph_match((feature, edge), sub_graph):
            # 如果包含子图，将所有包含子图的节点的标签设置为1，其余的节点的标签设置为-1
            # 这里我们假设子图匹配的函数返回的第一个映射是正确的，你也可以根据你的需求进行修改
            mapping = next(subgraph_match((feature, edge), sub_graph))
            for j in range(len(feature)):
                if j in mapping.keys():
                    label[j] = 1
                else:
                    label[j] = -1
        # 将修改后的标签重新赋值给labels
        labels[i] = label
    # 继续进行模型的训练和验证，沿用原来的代码
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
