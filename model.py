import numpy as np
import torch
from torch import nn
from torch_geometric.nn import TopKPooling, SAGEConv,GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

def construct_graph(sequence_length, window_size):
    start_points = []
    end_points = []
    for i in range(sequence_length):
        for j in range(max(0, i - window_size), min(sequence_length, i + window_size + 1)):
            if i != j:
                start_points.append(i)
                end_points.append(j)
    edge_f = torch.tensor([start_points, end_points], dtype=torch.long)
    return edge_f

def merge_graph(data_list):
    combined_x = torch.cat([data.feature for data in data_list], dim=0)
    offsets = torch.cumsum(torch.tensor([data.feature.size(0) for data in data_list]), dim=0)
    offsets = torch.cat([torch.tensor([0]), offsets[:-1]])
    combined_edge_index = torch.cat([data.edge + offset for data, offset in zip(data_list, offsets)], dim=1)
    combined_data = Data(x=combined_x, edge_index=combined_edge_index)

    return combined_data

class SelfAttentionFourLevels(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionFourLevels, self).__init__()
        self.query_layer = nn.Linear(input_dim, output_dim)
        self.key_layer = nn.Linear(input_dim, output_dim)
        self.value_layer = nn.Linear(input_dim, output_dim)
        self.scale_factor = torch.sqrt(torch.tensor(output_dim, dtype=torch.float32))

    def forward(self, individual_info, local_info, regional_info, global_info):
        x = torch.stack([individual_info, local_info, regional_info, global_info], dim=1)

        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 加权求和
        attention_output = torch.matmul(attention_weights, V)


        return attention_output.view(attention_output.size(0), -1)

class MVRBind(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MVRBind, self).__init__()
        self.conv1 = GCNConv(136, 136)
        self.conv2 = GCNConv(408, 136)
        self.conv3 = GCNConv(408, 136)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128, 1)
        self.lin4 = torch.nn.Linear(512, 256)
        self.act1 = torch.nn.ReLU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=408,
            nhead=4,
            dim_feedforward=256
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.self_attention = SelfAttentionFourLevels(136, 128)

        self.lin_c1 = torch.nn.Linear(408, 136)
        self.lin_c2 = torch.nn.Linear(408, 136)
        self.lin_c3 = torch.nn.Linear(408, 136)

        self.conv1_se = GCNConv(embed_dim, embed_dim)
        self.conv2_se = GCNConv(embed_dim, embed_dim)
        self.conv3_se = GCNConv(embed_dim, embed_dim)

        self.conv1_f = GCNConv(embed_dim, embed_dim)
        self.conv2_f = GCNConv(embed_dim, embed_dim)
        self.conv3_f = GCNConv(embed_dim, embed_dim)

        self.features = []

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index_f = construct_graph(x.shape[0],13)
        two_part_graph = data.sub_graph
        sed_graph = merge_graph(two_part_graph)
        x_s, edge_index_s = sed_graph.x, sed_graph.edge_index
        x1_s =  F.relu(self.conv1_se(x, edge_index_s))
        x2_s =  F.relu(self.conv2_se(x1_s, edge_index_s))
        x3_s =  F.relu(self.conv3_se(x2_s, edge_index_s))

        x1_f =  F.relu(self.conv1_f(x, edge_index_f))
        x2_f =  F.relu(self.conv2_f(x1_f, edge_index_f))
        x3_f =  F.relu(self.conv3_f(x2_f, edge_index_f))

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([x1, x1_s,x1_f], dim=1)

        x1 = x1.unsqueeze(0)
        x1 = self.transformer(x1)
        x1 = x1.squeeze(0)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = torch.cat([x2, x2_s,x2_f], dim=1)

        x2 = x2.unsqueeze(0)
        x2 = self.transformer(x2)
        x2 = x2.squeeze(0)

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = torch.cat([x3, x3_s, x3_f], dim=1)

        x3 = x3.unsqueeze(0)
        x3 = self.transformer(x3)
        x3 = x3.squeeze(0)

        x1 = F.relu(self.lin_c1(x1))
        x2 = F.relu(self.lin_c2(x2))
        x3 = F.relu(self.lin_c3(x3))
        x5 = self.self_attention(x, x1, x2, x3)
        self.features.append(x5.detach())
        x = self.lin4(x5)
        x = self.act1(x)
        x = self.lin1(x)
        x = self.act1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)
        return x
