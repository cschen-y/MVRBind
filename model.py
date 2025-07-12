import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data

def construct_graph(sequence_length, window_size):
    start_points, end_points = [], []
    for i in range(sequence_length):
        for j in range(max(0, i - window_size), min(sequence_length, i + window_size + 1)):
            if i != j:
                start_points.append(i)
                end_points.append(j)
    return torch.tensor([start_points, end_points], dtype=torch.long)

def merge_graph(data_list):
    combined_x = torch.cat([data.feature for data in data_list], dim=0)
    offsets = torch.cumsum(torch.tensor([data.feature.size(0) for data in data_list]), dim=0)
    offsets = torch.cat([torch.tensor([0]), offsets[:-1]])
    combined_edge_index = torch.cat([data.edge + offset for data, offset in zip(data_list, offsets)], dim=1)
    return Data(x=combined_x, edge_index=combined_edge_index)

class SelfAttentionFourLevels(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)
        self.scale_factor = torch.sqrt(torch.tensor(output_dim, dtype=torch.float32))
        self.output_proj = nn.Linear(output_dim * 4, 256)

    def forward(self, individual_info, local_info, regional_info, global_info):
        x = torch.stack([individual_info, local_info, regional_info, global_info], dim=1)
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        return self.output_proj(attention_output.view(attention_output.size(0), -1))

class MVRBind(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim * 3, embed_dim)
        self.conv3 = GCNConv(embed_dim * 3, embed_dim)

        self.lin1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.lin2 = nn.Linear(128, 1)

        self.lin4 = nn.Linear(256, 256)

        self.act = nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 3, nhead=2, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.self_attention = SelfAttentionFourLevels(embed_dim, 128)

        self.lin_c1 = nn.Linear(embed_dim * 3, embed_dim)
        self.lin_c2 = nn.Linear(embed_dim * 3, embed_dim)
        self.lin_c3 = nn.Linear(embed_dim * 3, embed_dim)

        self.conv1_se = GCNConv(embed_dim, embed_dim)
        self.conv2_se = GCNConv(embed_dim, embed_dim)
        self.conv3_se = GCNConv(embed_dim, embed_dim)

        self.conv1_f = GCNConv(embed_dim, embed_dim)
        self.conv2_f = GCNConv(embed_dim, embed_dim)
        self.conv3_f = GCNConv(embed_dim, embed_dim)

        self.dropout_rate = 0.5
        self.features = []

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index_f = construct_graph(x.shape[0], 13)
        two_part_graph = data.sub_graph
        sed_graph = merge_graph(two_part_graph)
        x_s, edge_index_s = sed_graph.x, sed_graph.edge_index


        x = x + 0.05 * torch.randn_like(x)

        # Spatial & Frequency Graph Paths
        x1_s = F.dropout(F.relu(self.conv1_se(x, edge_index_s)), p=self.dropout_rate, training=self.training)
        x2_s = F.dropout(F.relu(self.conv2_se(x1_s, edge_index_s)), p=self.dropout_rate, training=self.training)
        x3_s = F.dropout(F.relu(self.conv3_se(x2_s, edge_index_s)), p=self.dropout_rate, training=self.training)

        x1_f = F.dropout(F.relu(self.conv1_f(x, edge_index_f)), p=self.dropout_rate, training=self.training)
        x2_f = F.dropout(F.relu(self.conv2_f(x1_f, edge_index_f)), p=self.dropout_rate, training=self.training)
        x3_f = F.dropout(F.relu(self.conv3_f(x2_f, edge_index_f)), p=self.dropout_rate, training=self.training)

        # Main Path
        x1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout_rate, training=self.training)
        x1 = torch.cat([x1, x1_s, x1_f], dim=1)
        x1 = self.transformer(x1.unsqueeze(0)).squeeze(0)

        x2 = F.dropout(F.relu(self.conv2(x1, edge_index)), p=self.dropout_rate, training=self.training)
        x2 = torch.cat([x2, x2_s, x2_f], dim=1)
        x2 = self.transformer(x2.unsqueeze(0)).squeeze(0)

        x3 = F.dropout(F.relu(self.conv3(x2, edge_index)), p=self.dropout_rate, training=self.training)
        x3 = torch.cat([x3, x3_s, x3_f], dim=1)
        x3 = self.transformer(x3.unsqueeze(0)).squeeze(0)

        x1 = F.dropout(F.relu(self.lin_c1(x1)), p=self.dropout_rate, training=self.training)
        x2 = F.dropout(F.relu(self.lin_c2(x2)), p=self.dropout_rate, training=self.training)
        x3 = F.dropout(F.relu(self.lin_c3(x3)), p=self.dropout_rate, training=self.training)

        x5 = self.self_attention(x, x1, x2, x3)
        self.features.append(x5.detach())

        x = F.dropout(F.relu(self.lin4(x5)), p=self.dropout_rate, training=self.training)
        x = F.dropout(self.act(self.bn1(self.lin1(x))), p=self.dropout_rate, training=self.training)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        return x
