import os
import re

from math import sqrt
import numpy as np
from Bio.PDB import PDBParser
import networkx as nx
import pickle
from v2.reproduction.get_pdb_feature import get_pdb_features
import copy
from torch_geometric.data import Data, InMemoryDataset
import torch

def compute_eccentricity_with_zero(G):
    eccentricity = {}
    for node in G.nodes():
        try:
            distances = nx.single_source_shortest_path_length(G, node)
            eccentricity[node] = max(distances.values())
        except nx.NetworkXNoPath:
            eccentricity[node] = 0
    return eccentricity

def get_node_topology(adj_matrix):
    G = nx.from_numpy_matrix(adj_matrix)
    DG = dict(G.degree())
    NC = {node: np.mean([DG[neighbor] for neighbor in G.neighbors(node)]) if DG[node] > 0 else 0
          for node in G.nodes()}
    BC = nx.betweenness_centrality(G)
    CL = nx.closeness_centrality(G)
    EC = compute_eccentricity_with_zero(G)
    all_properties = []
    for node in G.nodes():
        node_properties = [DG[node], NC[node], BC[node], CL[node], EC[node]]
        all_properties.append(node_properties)

    return all_properties

def calculate_contact_matrix(input_file, chain_id, cutoff, mode, output_dir):
    pdb_id = re.match(r'(.+)\.pdb', input_file).group(1)
    pdb_id = pdb_id.split('/')[-1]
    contact_file_path = os.path.join(output_dir, pdb_id + '.dat')
    if os.path.exists(contact_file_path):
        with open(contact_file_path, 'rb') as f:
            all_contact_matrix = pickle.load(f)
    else:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB_structure', input_file)
        residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                      'A23', 'U37', 'IU']
        model_index = 0
        all_contact_matrix = []
        for model in structure:
            print(model_index)
            for chain in model:
                atoms = []
                count  = 0
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.get_resname().replace(" ", "") in residue_id:
                            count = count + 1
                            for atom in residue:
                                atoms.append((atom.get_serial_number(),
                                              residue.get_id()[1],  # 氨基酸编号
                                              atom.get_coord(),  # 原子坐标 (x, y, z)
                                              chain.get_id()))  # 链ID

            number_of_atoms = len(atoms)
            atom_serial_numbers = np.array([atom[0] for atom in atoms])
            amino_acid_numbers = np.array([atom[1] for atom in atoms])
            atom_positions = np.array([atom[2] for atom in atoms])
            chain_ids = [atom[3] for atom in atoms]
            number_of_amino_acids = 1
            revised_amino_acid_numbers = np.ones(number_of_atoms, dtype=int)
            for i in range(1, number_of_atoms):
                if abs(amino_acid_numbers[i] - amino_acid_numbers[i - 1]) > 0:
                    number_of_amino_acids += 1
                    revised_amino_acid_numbers[i:] = number_of_amino_acids
            contact_matrix = np.zeros((count, count), dtype=int)

            for i in range(number_of_atoms):
                for j in range(number_of_atoms):
                    if abs(revised_amino_acid_numbers[i] - revised_amino_acid_numbers[j]) <= 1:
                        continue
                    distance = np.linalg.norm(atom_positions[i] - atom_positions[j])
                    if distance <= cutoff:
                        contact_matrix[revised_amino_acid_numbers[i] - 1, revised_amino_acid_numbers[j] - 1] = 1
            model_index += 1
            all_contact_matrix.append(contact_matrix)

        with open(contact_file_path, 'wb') as f:
            pickle.dump(all_contact_matrix, f)
    return all_contact_matrix


class TestDataset(InMemoryDataset):
    def __init__(self, root, c_features,all_edges,all_labels,all_edge_s,transform=None, pre_transform=None, pre_filter=None):
        self.c_features = c_features
        self.all_edges = all_edges
        self.all_labels = all_labels
        self.all_edge_s = all_edge_s
        super(TestDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
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
        for feature, edge, label,edge_s in zip(self.c_features, self.all_edges, self.all_labels,self.all_edge_s):
            edge = torch.tensor(edge, dtype=torch.long)
            edge_s = torch.tensor(edge_s, dtype=torch.long)
            feature = torch.tensor(feature, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.float)
            sub_graph = Data(edge=edge_s, feature=feature, label=label)
            data = Data(x=feature, edge_index=edge, y=label, sub_graph=sub_graph)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_single_rna_coordinate(chain_id, pdb_file_path):
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    rna_coordinate = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MY", pdb_file_path)
    all_model_coordinate = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                rna_coordinate = []
                for residue in chain:
                    count = 0
                    x, y, z = 0, 0, 0
                    if residue.get_resname().replace(" ", "") in residue_id:
                        for atom in residue:
                            x_atom, y_atom, z_atom = atom.get_coord()
                            # print(x_atom)
                            x = x + x_atom
                            y = y + y_atom
                            z = z + z_atom
                            count += 1
                        rna_coordinate.append([x / count, y / count, z / count])
        all_model_coordinate.append(rna_coordinate)
    return all_model_coordinate

def find_min_indices(lst, num_indices):
    index_value_pairs = list(enumerate(lst))
    sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1])
    min_indices = [index for index, value in sorted_pairs[:min(num_indices, len(lst))]]

    return min_indices

def get_rna_edge(top_k, rna_coordinate):
    e_1 = []
    e_2 = []
    dis = []
    for index_1 in range(len(rna_coordinate)):
        for index_2 in range(len(rna_coordinate)):
            if index_2 == index_1:
                dis.append(100000)
            else:
                dis.append(
                    sqrt(sum([(abs(a - b)) ** 2 for a, b in zip(rna_coordinate[index_1], rna_coordinate[index_2])])))
        find_index = find_min_indices(dis, top_k)
        for item in range(len(find_index)):
            e_1.append(index_1)
            e_2.append(find_index[item])
        dis = []
    return [e_1, e_2]

def get_conformation_data():
    top_k = 8
    all_data = []
    pdb_file_path = "./data/apo_pdb"
    pdb_list = ["1PJYA", "1SCLA", "2L5ZA"]
    conformation_count = [20, 6, 21]
    all_topology = []
    all_edges_conformation = []
    for pdb in pdb_list:
        rna_topology = []
        rna_edges =[]
        rna_adj_matrix = calculate_contact_matrix(f'{pdb_file_path}/{pdb[:4]}.pdb', chain_id=pdb[4], cutoff=8.0,
                                                  mode="rep", output_dir="data/conformation/contact_matrix")
        for adj_matrix in rna_adj_matrix:
            topology = get_node_topology(adj_matrix)
            rna_topology.append(np.array(topology))
        all_topology.append(rna_topology)
        coordinate = get_single_rna_coordinate(pdb[4],f'{pdb_file_path}/{pdb[:4]}.pdb')
        for rna_coordinate in coordinate:
            edges = get_rna_edge(top_k, rna_coordinate)
            rna_edges.append(edges)
        all_edges_conformation.append(rna_edges)
    all_features, all_edges, all_labels, all_edge_s = get_pdb_features("./data/apo_conformation_pdb",
                                                                       "./data/apo_conformation_fastas",
                                                                       "./data/apo_msa_result",
                                                                       "./data/apo_conformation_em",
                                                                       "./data/apo_conformation_asa", 8)
    with open("./label/label_Tapo.pkl", 'rb') as f:
        temp = []
        all_labels = pickle.load(f)
        for i in range(len(all_labels)):
            if i in [1,2,3]:
                temp.append(all_labels[i])
        all_labels = temp
    for i in range(3):
        c_features = []
        for c in range(conformation_count[i]):
            temp = copy.deepcopy(all_features[i])
            all_features[i][:, -10:-5] = copy.deepcopy(all_topology[i][c])
            c_features.append(copy.deepcopy(all_features[i]))
            all_features[i] = temp
        c_edges = [all_edges[i]] * conformation_count[i]
        c_labels = [all_labels[i]] * conformation_count[i]
        all_data.append((c_features, c_labels, all_edges_conformation[i]))
        TestDataset(f"./pt/{i}",c_features,c_edges,c_labels,all_edges_conformation[i])
    with open('rnasite/rnasite_conformation_net.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    return all_data





