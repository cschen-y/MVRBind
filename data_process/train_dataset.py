import torch
from torch_geometric.data import Data, InMemoryDataset


class TrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(TrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_train.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        features_s,edges_s = test.get_edge_and_feature("")
        features = data_process.get_add_dl_data(self.i_parameter)
        edges, coordinate = set_graph.get_edge(self.i_parameter, data_file='')
        labels = data_process.get_cal_label('')
        for feature, edge, label, feature_s, edge_s in zip(features, edges, labels, features_s, edges_s):
            edge = torch.tensor(edge, dtype=torch.long)
            edge_s = torch.tensor(edge_s, dtype=torch.long)
            feature_s = torch.tensor(feature_s, dtype=torch.float)
            feature = torch.tensor(feature, dtype=torch.float)
            feature = torch.cat([feature_s, feature], dim=1)
            label = torch.tensor(label, dtype=torch.float)
            data_s = Data(edge=edge_s, feature=feature_s, label=label)
            data = Data(x=feature, edge_index=edge, y=label, sub_graph=data_s)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

