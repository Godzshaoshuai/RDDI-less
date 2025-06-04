from datetime import datetime
import time 
import argparse
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import LEConv

from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings

from rdkit import Chem
import torch
from torch_geometric.data import Data


from torch_geometric.utils import (remove_self_loops, degree,
                                   batched_negative_sampling)
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=128, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')# 200 used
parser.add_argument('--batch_size', type=int, default=256, help='batch size')                   #1024 used
parser.add_argument('--channels', type=int, default=55, help='num of channels')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='inductive.pkl')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
#device = 'cuda:1' if torch.cuda.is_available() and args.use_cuda else 'cpu'
# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

# 根据 CUDA 的可用性和用户的选择来设置设备
if cuda_available:
    device = torch.device("cuda")  # 如果有 GPU 可用，使用 GPU
else:
    device = torch.device("cpu")  # 否则使用 CPU


# 打印设备信息
print(f"Using device: {device}")
#device = torch.device("cuda")
print(args)
############################################################

###### Dataset
df_ddi_train = pd.read_csv('inductive_data/fold1/train.csv')
df_ddi_s1 = pd.read_csv('inductive_data/fold1/s1.csv')
df_ddi_s2 = pd.read_csv('inductive_data/fold1/s2.csv')


train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=2)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=2)

def smiles_to_graph(smiles):
    """
    将 SMILES 字符串转换为图结构（节点特征，边索引，边特征）
    """
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    # 获取节点特征（例如，原子类型的 one-hot 编码）
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())

    # 获取边的信息（包括边的索引和特征）
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # 图是无向的，所以要添加反向边
        edge_attr.append([bond.GetBondTypeAsDouble()])
        edge_attr.append([bond.GetBondTypeAsDouble()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)  # 节点特征

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None

# class RCGRL_IVGenerator(nn.Module):
#     def __init__(self, q_net, process_net):
#         super().__init__()
#         self.q_net = q_net
#         self.process_net = process_net
#
#     def forward(self, data):
#         edge_scores = self.q_net(data)
#         (robust_x, robust_edge_index, _, _, robust_batch), \
#         (_, _, _, _, _), _ = self.process_net(data, edge_scores)
#         return {"robust": (robust_x, robust_edge_index, robust_batch)}

class RCGRL_IVGenerator(nn.Module):
    def __init__(self, q_net, process_net):
        super().__init__()
        self.q_net = q_net
        self.process_net = process_net

    def forward(self, data):
        # 获取边得分（IVs）
        edge_scores = self.q_net(data)

        # 使用 ProcessNet 获取鲁棒图和全图数据
        (robust_x, robust_edge_index, _, _, robust_batch), \
        (_, _, _, _, _), _ = self.process_net(data, edge_scores)

        # 返回鲁棒图数据
        return robust_x, robust_edge_index, robust_batch

class QNet(nn.Module):  # The q(.) for calculating IVs

    def __init__(self):
        super(QNet, self).__init__()
        self.convq1 = LEConv(in_channels=55, out_channels=args.channels)
        self.convq2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )

    def forward(self, data):
        print(data)
        print(data.x)
        print(data.edge_index)
        print(data.edge_attr)
        print("attr  above")
        print("weidu ",data.x.dim(),data.edge_index.dim(),data.edge_attr.dim())
        q = self.convq1(data.x, data.edge_index,data.edge_attr.view(-1))
        print("q1",q)
        print("q1.shape",q.shape)
        q = self.convq2(q, data.edge_index,data.edge_attr.view(-1))
        print("q2",q)
        print("q2.shape",q.shape)

        row, col = data.edge_index
        print("row",row)
        print("row.shape",row.shape)
        print("col",col)
        print("col.shape",col.shape)
        edge_rep = torch.cat([q[row], q[col]], dim=-1)
        print("edge_rep",edge_rep)
        print("edge_rep.shape",edge_rep.shape)
        edge_weight = self.mlp(edge_rep).view(-1)  # The edge wights as IVs
        print("edge_weight",edge_weight)
        return edge_weight

class ProcessNet(nn.Module):

    def __init__(self, drop):
        super(ProcessNet, self).__init__()
        self.conv1 = LEConv(in_channels=55, out_channels=args.channels)
        self.conv2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )
        self.d = drop

    def forward(self, data, edge_score):
        # batch_norm
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        # 标记边信息，分别稳健边和健全边
        (robust_edge_index, robust_edge_attr, robust_edge_weight), \
            (full_edge_index, full_edge_attr, full_edge_weight) = drop_info_return_full(data, edge_score, self.d)  # r

        # 更新节点信息
        robust_x, robust_edge_index, robust_batch, _ = relabel(x, robust_edge_index, data.batch)
        full_x, full_edge_index, full_batch, _ = relabel(x, full_edge_index, data.batch)

        return (robust_x, robust_edge_index, robust_edge_attr, robust_edge_weight, robust_batch), \
            (full_x, full_edge_index, full_edge_attr, full_edge_weight, full_batch), \
            edge_score

def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos

def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

def drop_info_return_full(data, edge_score, d, require_edge_reserve_index=False):
    robust_edge_index = torch.LongTensor([[], []]).to(data.x.device)
    robust_edge_weight = torch.tensor([]).to(data.x.device)
    edge_reserve_index = torch.LongTensor([]).to(data.x.device)
    robust_edge_attr = torch.tensor([]).to(data.x.device)

    full_edge_index = torch.LongTensor([[], []]).to(data.x.device)
    full_edge_weight = torch.tensor([]).to(data.x.device)
    full_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    # counter = 0
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int((1 - d) * N)
        edge_attr = data.edge_attr[C:C + N]
        single_mask = edge_score[C:C + N]
        # single_mask = F.sigmoid(edge_score[C:C + N] * 100)

        # single_mask = single_mask.pow(1)
        single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve = rank[:n_reserve]
        # idx_reserve = rank

        robust_edge_index = torch.cat([robust_edge_index, edge_index[:, idx_reserve]], dim=1)
        # robust_edge_index = torch.cat([robust_edge_index, edge_index[:, :]], dim=1)
        full_edge_index = torch.cat([full_edge_index, edge_index[:, :]], dim=1)

        # robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        robust_edge_weight = torch.cat([robust_edge_weight, single_mask[idx_reserve]])
        idx_reserve_tn = torch.from_numpy(idx_reserve).to(device)
        # print("idx_reserve is ", idx_reserve)
        # print("edge_reserve_index is ", edge_reserve_index)
        # counter = counter + 1
        # print("counter is", counter)
        # if counter == 64:
        #     print(" ")
        edge_reserve_index = torch.cat([edge_reserve_index, idx_reserve_tn + C])
        # print("edge_reserve_index is ", edge_reserve_index)
        full_edge_weight = torch.cat([full_edge_weight, single_mask])

        # robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        robust_edge_attr = torch.cat([robust_edge_attr, edge_attr[idx_reserve]])
        full_edge_attr = torch.cat([full_edge_attr, edge_attr])

        # print("edge_reserve_index is ", edge_reserve_index)
        # print("C is ", C)

    if require_edge_reserve_index:
        return (robust_edge_index, robust_edge_attr, robust_edge_weight, edge_reserve_index), \
            (full_edge_index, full_edge_attr, full_edge_weight)
    else:
        return (robust_edge_index, robust_edge_attr, robust_edge_weight), \
            (full_edge_index, full_edge_attr, full_edge_weight)

def do_compute(batch, device, model):
        # print("do_computer")
        # print(datetime.today())

        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''

        pos_tri, neg_tri = batch

        pos_tri = [tensor.to(device=device) for tensor in pos_tri]

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        p_score = model(pos_tri)
        n_score = model(neg_tri)
        probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu(),torch.sigmoid(n_score.detach()).cpu()])
        ground_truth = np.concatenate([np.ones(len(p_score)),np.zeros(len(n_score))])
        return p_score, n_score, probas_pred, ground_truth

def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap
# def train_rcgrl_dsn(model, iv_generator, train_loader, loss_fn, optimizer, args, device):
#     model.train()
#     iv_generator.train()
#     print("train start")
#     for epoch in range(args.epoch):
#         # === 第一阶段：训练DSN-DDI部分 ===
#         iv_generator.eval()
#         model.train()
#         total_loss = 0
#         print("第%d次model train " % epoch)
#         for batch in train_loader:
#             batch = [t.to(device) for t in batch]
#             scores = model(batch, iv_generator=iv_generator, use_rcgrl=True)
#             labels = torch.ones_like(scores)
#
#             loss = loss_fn(scores, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print("第%d次IV train " % epoch)
#         # === 第二阶段：训练RCGRL部分（图结构裁剪器） ===
#         model.eval()
#         iv_generator.train()
#         rcgrl_loss_total = 0
#
#         for batch in train_loader:
#             h_data = batch[0].to(device)  # 假设每个 batch 是一个三元组中的一个图
#             edge_scores = iv_generator.q_net(h_data)
#
#             (robust_x, robust_edge_index, edge_attr, _, batch), _, _ = \
#                 iv_generator.process_net(h_data, edge_scores)
#
#             # 生成鲁棒图的表示和预测
#             set_masks(edge_attr, model)
#             robust_x = model.get_graph_rep(robust_x, robust_edge_index, edge_attr, batch)
#             robust_out = model.get_robust_pred(robust_x)
#             clear_masks(model)
#
#             y = h_data.y.to(device)  # 标签
#             rcgrl_loss = F.cross_entropy(robust_out, y)
#             optimizer.zero_grad()
#             rcgrl_loss.backward()
#             optimizer.step()
#             rcgrl_loss_total += rcgrl_loss.item()
#
#         print(f"[Epoch {epoch}] DDI Loss: {total_loss:.4f} | RCGRL Loss: {rcgrl_loss_total:.4f}")
def train_rcgrl_dsn(model, iv_generator, train_loader, loss_fn, optimizer, args, device):
    print("x")
    model.train()
    print("1")
    iv_generator.train()
    print("1")
    for epoch in range(args.n_epochs):
        # === 第一阶段：训练 DSN-DDI 模块 ===
        iv_generator.eval()
        model.train()
        total_loss = 0
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []
        n=0
        # for batch in train_loader:
        #     print("%d model",n)
        #     n=n+1
        #     model.train()
        #     p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
        #     train_probas_pred.append(probas_pred)
        #     train_ground_truth.append(ground_truth)
        #     loss, loss_p, loss_n = loss_fn(p_score, n_score)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_loss += loss.item() * len(p_score)
        # train_loss /= len(train_data)#原始代码插入
        # model.eval()
        # iv_generator.train()
        # rcgrl_loss_total = 0
        m = 0
        for batch in train_loader:
            print("%d model", m)
            m = m + 1
            #假设 batch 中包含三元组 (drug1, drug2, relation, split)
            pos_tri,neg_tri = batch
            #drug1_smiles, drug2_smiles, relation, split = batch
            pos_h_data, pos_t_data, pos_rels, pos_b_graph = pos_tri
            print("pos",pos_tri)
            print("pos_data",pos_h_data)
            print("pos_t_data",pos_t_data)
            print("pos_rels",pos_rels)
            print("pos_b_graph",pos_b_graph)
            print("end")
            neg_h_data, neg_t_data, neg_rels, neg_b_graph = neg_tri
            # 将 SMILES 转换为图结构
            h_data = pos_h_data # 转换为图   错 直接用dataload中的pos_tri和neg_tri中的data进行访问即可
            t_data = pos_t_data  # 转换为图
            relation = pos_rels
            # 将数据转换到设备
            h_data = h_data.to(device)
            t_data = t_data.to(device)
            relation = relation.to(device)

            # r使用 RCGRL 生成鲁棒图
            h_robust_x, h_robust_edge_index, h_robust_batch = iv_generator(h_data)
            h_data_new = Data(h_robust_x, h_robust_edge_index)
            t_robust_x, t_robust_edge_index, t_robust_batch = iv_generator(t_data)
            t_data_new = Data(t_robust_x, t_robust_edge_index)
            pos_tri_new = Data(h_data_new, t_data_new,relation,pos_b_graph)

            h_data = neg_h_data  # 转换为图   错 直接用dataload中的pos_tri和neg_tri中的data进行访问即可
            t_data = neg_t_data  # 转换为图
            relation = neg_rels
            # 将数据转换到设备
            h_data = h_data.to(device)
            t_data = t_data.to(device)
            relation = relation.to(device)

            # r使用 RCGRL 生成鲁棒图
            h_robust_x, h_robust_edge_index, h_robust_batch = iv_generator(h_data)
            h_data_new = Data(h_robust_x, h_robust_edge_index)
            t_robust_x, t_robust_edge_index, t_robust_batch = iv_generator(t_data)
            t_data_new = Data(t_robust_x, t_robust_edge_index)
            neg_tri_new = Data(h_data_new, t_data_new, relation, pos_b_graph)

            batch_new = Data(pos_tri_new, neg_tri_new)

            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            # scores = model((h_data, t_data, relation, split), iv_generator=iv_generator, use_rcgrl=True)#改到这了
            #
            # labels = torch.ones_like(scores)  # 假设是正样本，可扩展负采样
            # loss = loss_fn(scores, labels)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # total_loss += loss.item()

        print(f"[Epoch {epoch}] DDI Loss: {total_loss:.4f}")

        # === 第二阶段：训练 RCGRL 的 QNet + ProcessNet ===
        # model.eval()
        # iv_generator.train()
        # rcgrl_loss_total = 0
        # for batch in train_loader:
        #     h_data = batch[0].to(device)  # 假设每个 batch 是一个三元组中的一个图
        #     edge_scores = iv_generator.q_net(h_data)
        #
        #     # 获取鲁棒图和全图的数据
        #     (robust_x, robust_edge_index, _, _, robust_batch), \
        #     (_, _, _, _, _), _ = iv_generator.process_net(h_data, edge_scores)
        #
        #     # 将鲁棒图数据传入 DSN-DDI 模型
        #     set_masks(_, model)  # 使用 RCGRL 生成的掩码
        #     robust_graph_x = model.get_graph_rep(robust_x, robust_edge_index, _, robust_batch)
        #     robust_out = model.get_robust_pred(robust_graph_x)
        #     clear_masks(model)
        #
        #     y = h_data.y.to(device)  # 标签
        #     rcgrl_loss = F.cross_entropy(robust_out, y)
        #     optimizer.zero_grad()
        #     rcgrl_loss.backward()
        #     optimizer.step()
        #     rcgrl_loss_total += rcgrl_loss.item()
        #
        # print(f"[Epoch {epoch}] RCGRL Loss: {rcgrl_loss_total:.4f}")

def test(s1_data_loader, s2_data_loader, model):
    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model=model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)
        model.train()
        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        s1_acc, s1_auc_roc, s1_f1,s1_precision,s1_recall,s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        

        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device,model=model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)
        model.train()
        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        s2_acc, s2_auc_roc, s2_f1,s2_precision,s2_recall,s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

    print('\n')
    print('============================== Best Result ==============================')
    print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, s1_precision: {s1_precision:.4f},s1_recall: {s1_recall:.4f},s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
    print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, s2_precision: {s2_precision:.4f},s2_recall: {s2_recall:.4f},s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')
print("3")
model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
print("4")
q_net = QNet().to(device)#Qnet接受图数据
print("5")
process_net = ProcessNet(drop=0.75).to(device)
print("6")
iv_generator = RCGRL_IVGenerator(q_net, process_net)
print("7")

# print(model)
model.to(device=device)


if __name__ == '__main__':
    # print("Train will begin")
    print("first of  all")
    train_rcgrl_dsn(model,iv_generator,train_data_loader,loss, optimizer, args, device)  #s1_data_loader, s2_data_loader,
    # print("!!Train has finished.Test_model will begin")
    # print("!!!Train has finished.Test_model will begin")
    # print("!!!!Train has finished.Test_model will begin")
    test_model = torch.load(pkl_name)
    print("!!!!Test_model has finished,Test will begin")
    print("!!!Test_model has finished,Test will begin")
    print("！！Test_model has finished,Test will begin")
    test(s1_data_loader, s2_data_loader, test_model)
    print("Fnished")