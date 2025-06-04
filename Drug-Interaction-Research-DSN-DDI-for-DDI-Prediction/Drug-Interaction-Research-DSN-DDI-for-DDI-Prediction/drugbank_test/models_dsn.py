import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )

from layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )
import time


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


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        # print(kge_heads.size(), kge_tails.size(), rels.size())
        attentions = self.co_attention(kge_heads, kge_tails)
        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores

    # intra+inter


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x,
                                                                                                   h_data.edge_index,
                                                                                                   batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x,
                                                                                                   t_data.edge_index,
                                                                                                   batch=t_data.batch)

        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb



class CSS_DDI(nn.Module):
    def __init__(self, n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super(CSS_DDI, self).__init__()
        self.n_atom_feats = n_atom_feats
        self.n_atom_hid = n_atom_hid
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.heads_out_feat_params = heads_out_feat_params
        self.blocks_params = blocks_params

        # Initial normalization layer
        self.initial_norm = LayerNorm(self.n_atom_feats)

        # List to hold the blocks of the model
        self.blocks = nn.ModuleList()
        for block_params in self.blocks_params:
            block = CSSBlock(self.n_atom_feats, self.n_atom_hid, block_params)
            self.blocks.append(block)
            self.n_atom_feats = block_params['out_features']  # Update input features for the next block

        # Co-attention layer for interaction between drugs
        self.co_attention = CoAttentionLayer(self.kge_dim)

        # Knowledge Graph Embedding module for relation prediction
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, h_data, t_data, r_data, edge_index):
        # Apply initial normalization
        h_data.x = self.initial_norm(h_data.x)
        t_data.x = self.initial_norm(t_data.x)

        # Pass through each block
        for block in self.blocks:
            h_data, t_data = block(h_data, t_data, edge_index)

        # Co-attention to combine representations
        combined_repr = self.co_attention(h_data.x, t_data.x)

        # Knowledge Graph Embedding for relation prediction
        scores = self.KGE(combined_repr, r_data)

        return scores

class CSSBlock(nn.Module):
    def __init__(self, in_features, out_features, block_params):
        super(CSSBlock, self).__init__()
        self.conv1 = GATConv(in_features, block_params['mid_features'], block_params['num_heads'])
        self.conv2 = GATConv(block_params['mid_features'] * block_params['num_heads'], out_features, 1)

    def forward(self, h_data, t_data, edge_index):
        # First GAT layer
        h_data.x = self.conv1(h_data.x, edge_index)
        t_data.x = self.conv1(t_data.x, edge_index)

        # Second GAT layer
        h_data.x = self.conv2(h_data.x, edge_index)
        t_data.x = self.conv2(t_data.x, edge_index)

        return h_data, t_data