import os
import pickle

import numpy as np
import pandas as pd
import set_graph

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


def get_msa(file_path):
    with open(file_path,'rb') as f:
        feature_list = pickle.load(f)
    return feature_list

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


def get_data(file_em='./out_em', only_sequence_train_T60='./only_sequence_train_T60.txt', asa='./out_asa', file_msa='../MSA/train60MSA.pkl'):
    residue_one_hot = {'A': [1, 0, 0, 0],
                       'U': [0, 1, 0, 0],
                       'C': [0, 0, 1, 0],
                       'G': [0, 0, 0, 1]}
    mass = {'A': 0.46352,
            'U': -0.91164,
            'C': -0.97043,
            'G': 1.41855
            }
    PKa = {'A': -1.09202,
           'U': 0.72535,
           'C': -0.86883,
           'G': 1.23550
           }

    all_em = get_embedding(file_em)
    all_asa = get_asa(asa)
    all_mas = get_msa(file_msa)

    all_feature = []
    sequences = []
    with open(only_sequence_train_T60, 'r', encoding='utf-8') as f:
        for line in f:
            sequences.append(line.replace("\n", "").replace(" ", ""))
    for rna_index in range(len(all_em)):
        rna_feature = []
        for residue_index in range(len(all_em[rna_index])):
            residue_feature = []
            residue_type = sequences[rna_index][residue_index]
            residue_feature.extend(residue_one_hot[residue_type])
            residue_feature.extend(all_em[rna_index][residue_index])
            residue_feature.append(all_mas[rna_index][residue_index])
            residue_feature.append(all_asa[rna_index][residue_index])
            rna_feature.append(residue_feature)
        all_feature.append(rna_feature)
    return all_feature

def get_test18_data():
    residue_one_hot = {'A': [1, 0, 0, 0],
                       'U': [0, 1, 0, 0],
                       'C': [0, 0, 1, 0],
                       'G': [0, 0, 0, 1]}
    mass = {'A': 0.46352,
            'U': -0.91164,
            'C': -0.97043,
            'G': 1.41855
            }
    PKa = {'A': -1.09202,
           'U': 0.72535,
           'C': -0.86883,
           'G': 1.23550
           }

    all_em = get_embedding('../data/test18_em')
    all_asa = get_asa('../data/test18_asa')
    all_mas = get_msa('../../MSA/test18MSA.pkl')

    all_feature = []
    sequences = []
    with open('../data/only_sequence_test18.txt', 'r', encoding='utf-8') as f:
        for line in f:
            sequences.append(line.replace("\n", "").replace(" ", ""))
    for rna_index in range(len(all_em)):
        rna_feature = []
        for residue_index in range(len(all_em[rna_index])):
            residue_feature = []
            residue_type = sequences[rna_index][residue_index]
            residue_feature.extend(residue_one_hot[residue_type])
            residue_feature.extend(all_em[rna_index][residue_index])
            residue_feature.append(all_mas[rna_index][residue_index])
            residue_feature.append(all_asa[rna_index][residue_index])
            rna_feature.append(residue_feature)
        all_feature.append(rna_feature)
    return all_feature

def get_cl_many_graph(file_path='data/dl/train_dataset_w0', lic=6):
    result = []
    train_files = os.listdir(file_path)
    for item in sorted(train_files):
        # print(item)
        train_feature = []
        data = pd.read_csv(f"{file_path}/{item}", header=None)
        for i in range(data.shape[0]):
            train_feature.append(list(data.iloc[i, 1:lic]))
        result.append(train_feature)
    return result

def get_add_dl_data(top_k):
    result = []
    data_o = get_data(file_em='../data/out_em', only_sequence_train_T60='data/only_sequence_train_T60.txt',
                      asa='data/out_asa')
    dl_data = get_cl_many_graph(file_path='../data/dl/train_dataset_w0')
    e, coordinate = set_graph.get_edge(top_k, data_file='../../v3/data/fastas')
    for i, j, c in zip(data_o, dl_data, coordinate):
        result_one = []
        for item1, item2, item3 in zip(i, j, c):
            item1.extend(item2)
            result_one.append(item1)
        result.append(result_one)
    return result

def get_add_dl_data_test18(top_k):
    result = []
    data_o = get_test18_data()
    dl_data = get_cl_many_graph("../data/dl/test_dataset_w0")
    edges, coordinate = set_graph.get_truth_edges(top_k=top_k, data_file='../data/test18_fastas',
                                                  length_file="../data/Test18.txt")
    for i, j, z in zip(data_o, dl_data, coordinate):
        result_one = []
        for item1, item2, item3 in zip(i, j, z):
            item1.extend(item2)
            result_one.append(item1)
        result.append(result_one)
    return result

def get_label(label_file_path='./data/Train60.txt'):
    all_label = []
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for ann in f:
            ann = ann.strip('\n')
            rna_label = []
            if ann.startswith('0') or ann.startswith('1'):
                ann.replace(" ", "")
                ann_list = ann.split(',')
                for item in ann_list:
                    rna_label.append(int(item))
                all_label.append(rna_label)
    return all_label

def get_cal_label(label_file_path):
    with open(label_file_path, 'rb') as f:
        label = pickle.load(f)
    return label