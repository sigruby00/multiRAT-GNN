import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import BatchNorm
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv
from torch_geometric.utils import softmax


def scatter_mean(src, index, dim, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1  # Auto set if not specified

    assert index.max().item() < dim_size, f"[scatter_mean] index {index.max().item()} >= dim_size {dim_size}"

    sum_tensor = torch.zeros((dim_size,) + src.shape[1:], device=src.device)
    count_tensor = torch.zeros((dim_size,), device=src.device)

    sum_tensor.index_add_(dim, index, src)
    count_tensor.index_add_(0, index, torch.ones_like(index, dtype=torch.float))

    count_tensor = count_tensor.clamp(min=1)
    mean_tensor = sum_tensor / count_tensor.view(-1, *([1] * (sum_tensor.dim() - 1)))

    return mean_tensor


# ATARI: GCNConv
class GCNNet(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim=64):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(n_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x

# GATConv
class GATNet(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim=64, heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(n_node_features, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.conv3 = GATConv(hidden_dim * heads, 1, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x



class LinkModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_node_features + n_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, src_node, dst_node, link_attr, u=None, batch=None):
        link_input = torch.cat([src_node, dst_node, link_attr], dim=1)
        return self.mlp(link_input)


class NodeModel(nn.Module):
    def __init__(self, in_node_dim, edge_feat_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp_agg = nn.Sequential(
            nn.Linear(edge_feat_dim + in_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(hidden_dim + in_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, node_feat, edge_index, edge_attr, u=None, batch=None):
        row, col = edge_index
        agg_input = torch.cat([node_feat[col], edge_attr], dim=1)
        agg_msg = self.mlp_agg(agg_input)
        agg_msg = scatter_mean(agg_msg, row, dim=0, dim_size=node_feat.size(0))
        update_feat = torch.cat([node_feat, agg_msg], dim=1)
        return self.mlp_update(update_feat)


# MPNN
class MPNN(nn.Module):
    def __init__(self, n_node_features, n_edge_features, hidden_dim):
        super().__init__()
        self.edge1 = LinkModel(n_node_features, n_edge_features, hidden_dim, hidden_dim)
        self.node1 = NodeModel(n_node_features, hidden_dim, hidden_dim, hidden_dim)

        self.edge2 = LinkModel(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.node2 = NodeModel(hidden_dim, hidden_dim, hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        node_feat, edge_index, link_attr = data.x, data.edge_index, data.edge_attr
        row, col = edge_index

        # First message passing step
        src_node, dst_node = node_feat[row], node_feat[col]
        link_attr = self.edge1(src_node, dst_node, link_attr)
        node_feat = self.node1(node_feat, edge_index, link_attr)
        node_feat = F.relu(node_feat)

        # Second message passing step
        src_node, dst_node = node_feat[row], node_feat[col]
        link_attr = self.edge2(src_node, dst_node, link_attr)
        node_feat = self.node2(node_feat, edge_index, link_attr)

        return node_feat



# HTNet: Heterogeneous GAT with link (edge) attention
class Perceptron(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0, norm=False, act=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = BatchNorm(out_dim) if norm else None
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        if self.act:
            x = F.relu(x)
        if self.norm:
            x = self.norm(x)
        return x


class EGCNConvPyG(MessagePassing):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats):
        super().__init__(aggr='mean')
        self.fc_node = nn.Linear(in_node_feats, out_node_feats)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats)
        self.out_edge_feats = out_edge_feats

    def forward(self, x, edge_index, edge_attr):
        self.edge_attr = edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        edge_feat = self.fc_ni(x_i) + self.fc_nj(x_j) + self.fc_fij(edge_attr)
        return self.fc_node(x_j) * 1.0  # Apply dummy attention (set to 1)


class EGATConvPyG(MessagePassing):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats):
        super().__init__(aggr='add')
        self.fc_node = nn.Linear(in_node_feats, out_node_feats)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats)
        self.attn = nn.Parameter(torch.Tensor(1, out_edge_feats))
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_attr):
        self.edge_attr = edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        e = self.fc_ni(x_i) + self.fc_nj(x_j) + self.fc_fij(edge_attr)
        e = F.leaky_relu(e)
        alpha = (e * self.attn).sum(dim=-1, keepdim=True)
        alpha = softmax(alpha, index)
        return self.fc_node(x_j) * alpha


class HTNetPyG(nn.Module):
    def __init__(self, num_layer, dim, is_hetero, edge_types, n_node_features, n_edge_features):
        super().__init__()
        self.num_layer = num_layer
        self.is_hetero = is_hetero
        self.edge_types = edge_types

        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.convs = nn.ModuleList()

        # Input MLPs (per node type)
        self.input_mlps = nn.ModuleDict({
            'sta': nn.Linear(n_node_features, dim),
            'ap': nn.Linear(n_node_features, dim),
        })

        # Linear layers per edge type
        self.edge_input_mlps = nn.ModuleDict({
            str(rel): nn.Linear(n_edge_features, dim) for rel in edge_types
        })

        for _ in range(num_layer):
            self.node_norms.append(BatchNorm(dim, track_running_stats=False))
            self.edge_norms.append(BatchNorm(dim, track_running_stats=False))
            convs = {}

            for rel in edge_types:
                if is_hetero:
                    convs[rel] = EGATConvPyG(dim, dim, dim, dim)
                else:
                    convs[rel] = EGCNConvPyG(dim, dim, dim, dim)
            self.convs.append(HeteroConv(convs, aggr='mean'))

        # DGL-style 2-layer perceptron
        self.predict = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        # Optional Softplus activation (currently disabled)
        # self.softplus = nn.Softplus()

    def forward(self, data: HeteroData):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        # Node 입력 임베딩
        x_dict = {k: self.input_mlps[k](v) for k, v in x_dict.items()}

        # Edge feature 임베딩
        edge_attr_dict = {
            k: self.edge_input_mlps[str(k)](v) for k, v in edge_attr_dict.items() if str(k) in self.edge_input_mlps
        }

        for i in range(self.num_layer):
            x_dict = {k: self.node_norms[i](v) for k, v in x_dict.items()}
            edge_attr_dict = {k: self.edge_norms[i](v) for k, v in edge_attr_dict.items()}

            conv_out = self.convs[i](x_dict, edge_index_dict, edge_attr_dict)

            # Preserve and update original x_dict
            updated_x_dict = x_dict.copy()
            updated_x_dict.update(conv_out)  # 덮어쓰기
            x_dict = updated_x_dict

        # Ensure 'sta' key exists in x_dict
        if 'sta' not in x_dict:
            raise ValueError("'sta' not found in x_dict. Check if edge_types include dst_type='sta'.")

        h = x_dict['sta']
        out = self.predict(h)
        # return self.softplus(out)
        return out
