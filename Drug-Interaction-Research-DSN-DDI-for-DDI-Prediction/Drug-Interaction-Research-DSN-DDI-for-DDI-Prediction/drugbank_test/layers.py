import math
import datetime
import  random
import torch
from torch import nn
import torch.nn.functional as F
import argparse

from torch_geometric.nn import GCNConv, SAGPooling, global_add_pool, GATConv, GINEConv, LEConv
from torch_geometric.data import Data


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
      
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        
        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
       
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,32,2)
    
    def forward(self,data):
        input_feature,edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature,edge_index)
        return intra_rep

# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim,input_dim),32,2)
    
    def forward(self,h_data,t_data,b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input,t_input),edge_index)
        h_rep = self.inter((t_input,h_input),edge_index[[1,0]])
        return h_rep,t_rep



class QNetWrapper(nn.Module):
    """适配药物分子图的工具变量生成器"""

    def __init__(self, node_dim, out_dim):
        super().__init__()
        # 分子图专用特征提取
        self.conv1 = GINEConv(

           nn.Sequential(
                nn.Linear(node_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim)
            ))

        # 边权重计算
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        row, col = data.edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=-1)
        return torch.sigmoid(self.edge_mlp(edge_feat)).squeeze()


class QNetWrapper2(nn.Module):  # The q(.) for calculating IVs

    def __init__(self,*args ):
        super(QNetWrapper2, self).__init__()
        self.convq1 = LEConv(in_channels=4, out_channels=args.channels)
        self.convq2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels * 2, args.channels * 4),
            nn.ReLU(),
            nn.Linear(args.channels * 4, 1)
        )

    def forward(self, data):
        q = self.convq1(data.x, data.edge_index, data.edge_attr.view(-1))
        q = self.convq2(q, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([q[row], q[col]], dim=-1)
        edge_weight = self.mlp(edge_rep).view(-1)  # The edge wights as IVs

        return edge_weight




class ProcessNetWrapper(nn.Module):
    """支持双药协同处理的改进型"""

    def __init__(self, in_dim, out_dim, n_heads, drop_rate):
        super().__init__()
        # 共享基础特征提取
        self.conv_shared = GATConv(in_dim, out_dim // n_heads, n_heads)

        # 双重处理路径
        self.robust_conv = GATConv(out_dim, out_dim // 2, 2)
        self.full_conv = GATConv(out_dim, out_dim // 2, 2)

        # 基于RCGRL的边剪枝
        self.drop_rate = drop_rate
        #self.edge_drop = self.SparseEdgeDrop(edge_index,rate= drop_rate)##随机剪枝？这里疑惑，盲目的剪枝确实可以慢慢的趋近正解，机器学习可能就是如此愚笨，给一个随机的可能性和一个限制结局的选择性，无数次演变总用正确的可能

    def forward(self, h_data, h_weights, t_data, t_weights):
        # 特征共享提取
        h_data.x = self.conv_shared(h_data.x, h_data.edge_index)
        t_data.x = self.conv_shared(t_data.x, t_data.edge_index)

        # 鲁棒路径处理
        h_robust = self._process_single(h_data, h_weights, is_robust=True)
        t_robust = self._process_single(t_data, t_weights, is_robust=True)

        # 完整路径处理
        h_full = self._process_single(h_data, h_weights, is_robust=False)
        t_full = self._process_single(t_data, t_weights, is_robust=False)

        return (h_robust, h_full), (t_robust, t_full)

    import random
    import torch

    def SparseEdgeDrop(edge_index, edge_weights, rate):
        """Randomly drops edges based on the given rate.

        Args:
            edge_index (torch.Tensor): Original edge indices, shape (2, E).
            edge_weights (torch.Tensor): Original edge weights, shape (E, *).
            rate (float): Drop rate. The proportion of edges to be dropped.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - new_edge_index (torch.Tensor): Processed edge indices, shape (2, k).
                - new_edge_weights (torch.Tensor): Processed edge weights, shape (k, *).
        """
        u, v = edge_index
        E = u.size(0)

        if E == 0 or rate <= 0:
            return edge_index, edge_weights

        # Calculate number of edges to keep
        keep_num = int(E * (1 - rate))
        if keep_num <= 0:
            return torch.zeros(2, 0, dtype=u.dtype), torch.zeros(0, dtype=edge_weights.dtype)

        # Randomly select indices to keep
        indices = list(range(E))
        random.shuffle(indices)
        keep_indices = indices[:keep_num]

        # Select kept edges
        new_u = u[keep_indices]
        new_v = v[keep_indices]
        new_weights = edge_weights[keep_indices]

        return torch.stack([new_u, new_v]), new_weights

    def _process_single(self, data, edge_weights, is_robust):
        # 边剪枝操作
        if is_robust:
            edge_index, edge_attr = self.SparseEdgeDrop(
                data.edge_index, edge_weights,
                rate=self.drop_rate
            )
        else:
            edge_index, edge_attr = data.edge_index, data.edge_attr

        # 路径特征处理
        x = self.robust_conv(data.x, edge_index) if is_robust \
            else self.full_conv(data.x, edge_index)

        return Data(x=x, edge_index=edge_index, batch=data.batch)




class AlternateTrainer:
    def __init__(self, model, lr=1e-3, ema_decay=0.999):
        self.model = model
        self.optim = torch.optim.AdamW([
            {'params': model.q_params(), 'lr': lr / 10},  # Q网络更慢的学习
            {'params': model.p_params(), 'lr': lr}
        ])
        self.ema = EMA(ema_decay)

    def train_epoch(self, loader, epoch):
        # 阶段切换逻辑
        if epoch % 2 == 0:
            self.model.set_mode('q_train')  # 仅更新Q网络
        else:
            self.model.set_mode('p_train')  # 仅更新处理网络

        for batch in loader:
            self.optim.zero_grad()

            # 前向传播
            h_data, t_data = batch
            scores = self.model((h_data, t_data))

            # 损失计算（含正则项）
            # loss = self.criterion(scores) + \
            #        self.model.topology_loss() * 0.1 + \
            #        self.model.sparsity_loss() * 0.01
            #
            # loss.backward()
            self.optim.step()
            self.ema.update(self.model)  # 参数平滑

    # def criterion(self, scores):
    #     # 自定义多任务损失函数
    #     return F.binary_cross_entropy_with_logits(scores, labels) + \
    #         F.mse_loss(scores, aux_labels)


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)