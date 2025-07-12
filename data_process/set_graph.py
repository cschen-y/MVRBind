import os
from math import sqrt
import numpy as np
import torch
from Bio import PDB
from Bio.PDB import PDBParser


def aggregation(feature, edges, top_k=9):
    feature = feature.tolist()
    aggregation_feature = []
    one_feature_length = len(feature[0])
    temp = []
    count = 0
    for index_s, index_e in zip(edges[0], edges[1]):
        # print(index_s)
        index_s = int(index_s)
        index_e = int(index_e)
        if index_s != count:
            if len(temp) < top_k * one_feature_length:
                temp.extend(np.random.uniform(0, 0.001, [1, top_k * one_feature_length - len(temp)]).tolist()[0])
            aggregation_feature.append(temp)
            temp = []
            count = count + 1
        temp.extend(feature[index_e])
        # temp.extend(em[index_e])
    aggregation_feature.append(temp)
    aggregation_feature = torch.tensor(aggregation_feature, dtype=torch.float)
    return aggregation_feature


def find_min_indices(lst, num_indices):
    index_value_pairs = list(enumerate(lst))
    sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1])
    min_indices = [index for index, value in sorted_pairs[:min(num_indices, len(lst))]]
    return min_indices


def get_single_rna_coordinate(chain_id, pdb_file_path, residue_id):
    count_que =0
    rna_coordinate = []
    coordinate_c_rna = []
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("MY", pdb_file_path)
    model = structure[0]
    temp = 0
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                count = 0
                x, y, z = 0, 0, 0
                if residue.get_resname().replace(" ", "") in residue_id:
                    for atom in residue:
                        x_atom, y_atom, z_atom = atom.get_coord()
                        x = x + x_atom
                        y = y + y_atom
                        z = z + z_atom
                        count += 1
                    rna_coordinate.append([x / count, y / count, z / count])
                    atoms = residue.get_atoms()
                    coordinate_c = []
                    for atom in atoms:
                        atom_type = atom.get_name()
                        if 'C' in atom_type and "'" not in atom_type:
                            coordinate_c.extend(atom.get_coord())
                            temp = atom.get_coord()
                    if len(coordinate_c) < 15:
                        if len(coordinate_c) == 0:
                            count_que = count_que + 1
                        add_count = int(5 - len(coordinate_c) / 3)
                        coordinate_c.extend(list(temp) * add_count)
                    if len(coordinate_c) > 15:
                        coordinate_c = coordinate_c[:15]
                    coordinate_c_rna.append(coordinate_c)
    return rna_coordinate, coordinate_c_rna


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

def sliding_window_edges(data, window_size):
    start_points = []
    end_points = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data) - 1, i + window_size // 2)
        for j in range(start, end + 1):
            if j != i:
                start_points.append(data[i])
                end_points.append(data[j])

    return [start_points, end_points]

def calculate_contact_matrix(input_file, chain_id, cutoff, mode):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB_structure', input_file)
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    atoms = []
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                if residue.get_resname().replace(" ", "") in residue_id:
                    for atom in residue:
                        atoms.append((atom.get_serial_number(),
                                      residue.get_id()[1],
                                      atom.get_coord(),
                                      chain.get_id()))

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
    contact_matrix = np.zeros((number_of_amino_acids, number_of_amino_acids), dtype=int)

    for i in range(number_of_atoms):
        for j in range(number_of_atoms):
            if abs(revised_amino_acid_numbers[i] - revised_amino_acid_numbers[j]) <= 1:
                continue
            distance = np.linalg.norm(atom_positions[i] - atom_positions[j])
            if distance <= cutoff:
                contact_matrix[revised_amino_acid_numbers[i] - 1, revised_amino_acid_numbers[j] - 1] = 1


    return contact_matrix


def get_edge(top_k=8, file_path='./data/pdbFiles', data_file='./data/fastas'):
    files = os.listdir(data_file)
    files = sorted(files)
    all_coordinate = []
    all_edges = []
    all_coordinate_c_rna = []
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    for item in files:
        rna_coordinate, coordinate_c_rna = get_single_rna_coordinate(item[4], file_path + '/' + item[:4] + '.pdb',
                                                                     residue_id)
        all_coordinate.append(rna_coordinate)
        all_coordinate_c_rna.append(coordinate_c_rna)
    all_edges = []
    for rna_c in all_coordinate:
        rna_edge = get_rna_edge(top_k, rna_c)
        all_edges.append(rna_edge)
    return all_edges, all_coordinate


def get_truth_edges(top_k=30, file_path='data/pdbFiles', data_file='data/fastas',
                    length_file='data/Train60.txt'):
    edges, c = get_edge(top_k=top_k, file_path=file_path, data_file=data_file)
    count = 0
    length_data = []
    all_start_edge = []
    all_end_edge = []
    with open(length_file, "r") as f:
        for line in f:
            line = line.strip("\n")
            if count % 3 == 1:
                line.replace(" ", "")
                length_data.append(len(line))
            count += 1
    count = 0
    rna_count = 0
    for graph_edge in edges:
        start_edge = graph_edge[0]
        end_edge = graph_edge[1]
        length = len(start_edge)
        for item in range(length):
            start_edge[item] = start_edge[item] + count
            end_edge[item] = end_edge[item] + count
        all_start_edge.extend(start_edge)
        all_end_edge.extend(end_edge)
        count = count + length_data[rna_count]
        rna_count = rna_count + 1
    big_edge = [all_start_edge, all_end_edge]
    return big_edge, c
