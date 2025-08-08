import os
import pickle
from xml.sax.handler import all_features

import numpy as np
from Bio.PDB import PDBParser
from collections import defaultdict
from math import sqrt
import torch
from torch_geometric.data import Data, InMemoryDataset

from data_process.get_topology import get_node_topology
from data_process.get_connection_ad import calculate_contact_matrix
from data_process.msa import get_msa_score
from data_process.process_secondary_structure import get_edge_and_feature
from collections import defaultdict


def find_min_indices(lst, num_indices):
    index_value_pairs = list(enumerate(lst))
    sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1])
    min_indices = [index for index, value in sorted_pairs[:min(num_indices, len(lst))]]

    return min_indices


def get_label(pdb_file,chain_id):
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_file)
    rna_residues = []
    non_nucleotide_atoms = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if chain.id == chain_id:
                if residue.get_resname().replace(" ", "") in residue_id:
                    rna_residues.append(residue)
                elif residue.get_resname().replace(" ", "").upper() not in ['HOH']:
                    non_nucleotide_atoms.extend(residue.get_atoms())
    if len(non_nucleotide_atoms) == 0:
        return np.zeros(len(rna_residues))
    residue_labels = np.zeros(len(rna_residues))
    distance_threshold = 4
    for non_nucleotide_atom in non_nucleotide_atoms:
        for i, residue in enumerate(rna_residues):
            for atom in residue.get_atoms():
                if atom.get_name().startswith("H"):
                    continue
                distance = atom - non_nucleotide_atom
                if distance < distance_threshold:
                    residue_labels[i] = 1
                    break

    return residue_labels

def get_single_rna_coordinate(chain_id, pdb_file_path):
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    rna_coordinate = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("MY", pdb_file_path)
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:
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
    return rna_coordinate

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


def data_normaliztion(data):
    all_data = []
    nor_data = []
    count = 0
    for rna in data:
        for residue in rna:
            all_data.append(residue)
    for item in all_data:
        x = float(item - np.mean(all_data)) / np.std(all_data)
        nor_data.append(x)
    for rna_index in range(len(data)):
        for residue_index in range(len(data[rna_index])):
            data[rna_index][residue_index] = nor_data[count]
            count += 1
    return data

def get_asa(file_path='./out_asa'):
    files = os.listdir(file_path)
    files = sorted(files)
    all_asa = []
    for item in files:
        with open(file_path + '/' + item, 'r', encoding='utf-8') as f:
            rna_asa = []
            for line in f:
                if not line.startswith("#") and '\t' in line:
                    line = line.replace('\n', "")
                    temp = line.split('\t\t')
                    rna_asa.append(int(temp[-1]))
            all_asa.append(rna_asa)
    all_asa = data_normaliztion(all_asa)
    return all_asa

def get_residue_embedding(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        embedding_list = data.split("]")
        residue_list = []
        result = []
        for one in embedding_list:
            if len(one) < 10:
                continue
            one = one.replace('[', "")
            one = one.replace(']', "")
            one = one.replace('\n', "")
            one = one.replace(' ', "")
            one = one.replace("'", "")
            one = one.replace("'", "")
            residue_list.append(one)
        # print(residue_list)
        for item in residue_list:
            if item[0] == ',':
                item = item[1:]
            re_list = item.split(',')
            number_list = []
            for number_str in re_list:
                number_str = number_str.strip()
                number_list.append(float(number_str))
            result.append(number_list)
        return result


def get_embedding(rna_file_path):
    result = []
    files = os.listdir(rna_file_path)
    files = sorted(files)
    for file_path in files:
        data = get_residue_embedding(rna_file_path + '/' + file_path)
        result.append(data)
    return result

def get_pdb_features(pdb_file_path='./data/data_apo_holo_pdb_adjust',fasta_file_path="./data/apo_holo_fasta",msa_file_path='./data/apo_holo_msa/aglin',em_file_path='./data/apo_holo_feature/apo_holo_em', asa_file_path = "./data/apo_holo_feature/apo_holo_asa",top_k=8):
    residue_one_hot = {'A': [1, 0, 0, 0],
                       'U': [0, 1, 0, 0],
                       'C': [0, 0, 1, 0],
                       'G': [0, 0, 0, 1]}
    all_em = get_embedding(f'{em_file_path}')
    fasta_list = sorted(os.listdir(fasta_file_path))
    all_asa = get_asa(asa_file_path)
    all_feature_s,edge_s = get_edge_and_feature(fasta_file_path,'./data_process/data/secondary_structure')
    all_msa = []
    all_topology = []
    all_rna_type = []
    all_feature = []
    all_label = []
    all_edge = []
    for pdb in fasta_list:
        rna_type = []
        adj_matrix = calculate_contact_matrix(f'{pdb_file_path}/{pdb[:5]}.pdb', chain_id=pdb[4], cutoff=8.0, mode="rep", output_dir="data/apo_holo_feature/contact_matrix")
        topology = get_node_topology(adj_matrix)
        msa = get_msa_score(f'{msa_file_path}/{pdb[:5]}_msa_aligned.aln')
        rna_label = get_label(f'{pdb_file_path}/{pdb[:5]}.pdb',pdb[4])
        rna_coordinate = get_single_rna_coordinate(pdb[4], f'{pdb_file_path}/{pdb[:5]}.pdb')
        rna_edge = get_rna_edge(top_k, rna_coordinate)
        with open(f"{fasta_file_path}/{pdb}", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for residue_type in lines[1].replace("\n", "").replace(" ", ""):
                rna_type.append(residue_one_hot[residue_type])
        all_topology.append(topology)
        all_msa.append(msa)
        all_rna_type.append(rna_type)
        all_label.append(rna_label)
        all_edge.append(rna_edge)
    for node_em, node_topology, node_msa, node_rna_type, node_feature_s, node_asa in zip(all_em, all_topology, all_msa, all_rna_type,all_feature_s,all_asa):
        all_em = np.array(node_em)
        all_topology = np.array(node_topology)
        all_msa = np.array(node_msa)
        all_rna_type = np.array(node_rna_type)
        all_feature_s = np.array(node_feature_s)
        all_asa = np.array(node_asa)
        print(all_rna_type.shape, all_em.shape, all_msa.shape, all_asa.shape, all_topology.shape, all_feature_s.shape)
        rna_feature = np.column_stack((all_rna_type, all_em, all_msa, all_asa, all_topology, all_feature_s))
        all_feature.append(rna_feature)
    return  all_feature, all_edge, all_label,edge_s

def get_cal_label(label_file_path):
    with open(label_file_path, 'rb') as f:
        label = pickle.load(f)
    return label


class TestDataset(InMemoryDataset):
    def __init__(self, root, pdb_file_path=None, fasta_file_path=None, msa_file_path=None, em_file_path=None, asa_file_path=None, label_file_path=None, top_k=None, mode="train", transform=None, pre_transform=None, pre_filter=None):
        self.pdb_file_path = pdb_file_path
        self.fasta_file_path = fasta_file_path
        self.msa_file_path = msa_file_path
        self.em_file_path = em_file_path
        self.asa_file_path = asa_file_path
        self.label_file_path = label_file_path
        self.top_k = top_k
        self.mode = mode
        super(TestDataset, self).__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_train.pt']

    def download(self):
        pass

    def process(self):
        if not all([self.pdb_file_path, self.fasta_file_path, self.msa_file_path, self.em_file_path, self.asa_file_path, self.top_k]):
            raise ValueError("pdb_file_path, fasta_file_path, msa_file_path, em_file_path, asa_file_path, top_k 必须提供")

        data_list = []
        all_features, all_edges, all_labels, all_edge_s = get_pdb_features(
            self.pdb_file_path, self.fasta_file_path, self.msa_file_path,
            self.em_file_path, self.asa_file_path, self.top_k
        )
        for feature, edge, label, edge_s in zip(all_features, all_edges, all_labels, all_edge_s):
            edge = torch.tensor(edge, dtype=torch.long)
            edge_s = torch.tensor(edge_s, dtype=torch.long)
            feature = torch.tensor(feature, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.float)
            sub_graph = Data(edge=edge_s, feature=feature, label=label)
            data = Data(x=feature, edge_index=edge, y=label, sub_graph=sub_graph)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



