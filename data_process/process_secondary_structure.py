import os
import pandas as pd
import re



def get_fasta_rna_length(fasta_file_path):
    files = os.listdir(fasta_file_path)
    files = sorted(files)
    length_dist = {}
    for file_name in files:
        with open(fasta_file_path + f"/{file_name}") as file:
            lines = file.readlines()
            length_dist[file_name[:4].upper()] = len(lines[1])
    return length_dist


def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    numbers_string = ''.join(numbers)
    return numbers_string


def select_line(file_name, add_file_name, chain_id, t_length):
    forward_residue_count = 0
    with open(file_name + add_file_name) as file:
        lines = file.readlines()
        for line_count in range(len(lines)):
            if line_count % 3 == 1:
                forward_residue_count = forward_residue_count + len(lines[line_count]) - 1
            if line_count % 3 == 0:
                if lines[line_count][-2] == chain_id:
                    # if t_length == len(lines[line_count + 1]) - 1:
                    #     # print("true")
                    # else:
                    #     # print("false")
                    return file_name, len(lines[line_count + 1]) - 1, forward_residue_count
    return file_name, False, False


def choose_chain(pdb_name, chain_id, fasta_file_path,s_file_path):
    align_length = get_fasta_rna_length(fasta_file_path)
    t_length = align_length[pdb_name]
    pdb_name = pdb_name.upper()
    if pdb_name == "2NOK" and chain_id == "B":
        t_length = 20
    file_name = f"{s_file_path}/{pdb_name}/dssr-nc_visualization-hybrid-pseudoknots_as_paired_residues-varna/"
    if os.path.exists(file_name + f"{pdb_name}-2D-dotbracket.txt"):
        # with open(file_name + f"{pdb_name}-2D-dotbracket.txt") as file:
        file_name, length, residue_count = select_line(file_name, f"{pdb_name}-2D-dotbracket.txt", chain_id,
                                                       t_length)
        return file_name, length, residue_count

    count = 1
    forward_residue_count = 0
    file_name = file_name + "1/"
    while os.path.exists(file_name):
        # with open(chain_id_file_path) as file:
        file_name, length, residue_count = select_line(file_name, f"/{pdb_name}-2D-dotbracket.txt", chain_id, t_length)
        if not length:
            count = count + 1
            forward_residue_count = forward_residue_count + residue_count
            file_name = file_name[:-2] + str(count) + "/"
        else:
            return file_name, length, forward_residue_count
    return False, False


def get_edge_and_feature(fasta_file_path,s_file_path='./data_process/data/secondary_structure_data'):
    files = os.listdir(fasta_file_path)
    files = sorted(files)
    all_feature = []
    all_edge = []
    for fasta_name in files:
        # print(fasta_name)
        pdb_name = fasta_name[:4].upper()
        chain_id = fasta_name[4].upper()
        print(pdb_name)
        secondary_data_path, length, forward_residue_count = choose_chain(pdb_name, chain_id, fasta_file_path,s_file_path)
        if pdb_name == "5E54" and chain_id == "A":
            length = 65
        rna_feature = [[0 for _ in range(5)] for _ in range(length)]
        # print(length)
        index_dist_0 = {}
        index_dist_s = {}
        e1 = []
        e2 = []
        with open(secondary_data_path + f"{pdb_name}-2D-ct.txt") as file:
            # 读取所有行
            lines = file.readlines()
            count = 0
            # 跳过第一行，从第二行开始处理
            for line in lines[forward_residue_count + 1:]:
                parts = line.split()
                index_dist_s[parts[-1]] = count
                index_dist_0[parts[0]] = count
                count = count + 1
                if count == length:
                    break
        with open(secondary_data_path + f"{pdb_name}-2D-bpseq.txt") as file:
            lines = file.readlines()
            count = 0
            for line in lines[forward_residue_count:]:
                parts = line.split()
                if parts[-1] == "0":
                    rna_feature[index_dist_0[parts[0]]][0] = 1
                else:
                    start = index_dist_0[parts[0]]
                    if parts[-1] in index_dist_0:
                        end = index_dist_0[parts[-1]]
                        e1.append(start)
                        e2.append(end)
                count = count + 1
                if count == length:
                    break
        if os.path.exists(secondary_data_path + f"{pdb_name}-base-phoshpate.csv"):
            base_phoshpate = pd.read_csv(secondary_data_path + f"{pdb_name}-base-phoshpate.csv", sep=';')
            for interaction in base_phoshpate["Base-pair"]:
                interaction = interaction.replace(" ", "").replace("\n", "")
                residue_name = interaction.split('-')
                e1_number = extract_numbers(residue_name[0])
                e2_number = extract_numbers(residue_name[1])
                if residue_name[0][0] == chain_id and residue_name[1][
                    0] == chain_id and e1_number in index_dist_s and e2_number in index_dist_s:
                    e1.append(index_dist_s[e1_number])
                    e2.append(index_dist_s[e2_number])
                    rna_feature[index_dist_s[e1_number]][1] = 1
                    rna_feature[index_dist_s[e2_number]][1] = 1
                else:
                    continue
        if os.path.exists(secondary_data_path + f"{pdb_name}-base-ribose.csv"):
            base_phoshpate = pd.read_csv(secondary_data_path + f"{pdb_name}-base-ribose.csv", sep=';')
            for interaction in base_phoshpate["Base-pair"]:
                interaction = interaction.replace(" ", "").replace("\n", "")
                residue_name = interaction.split('-')
                e1_number = extract_numbers(residue_name[0])
                e2_number = extract_numbers(residue_name[1])
                if residue_name[0][0] == chain_id and residue_name[1][
                    0] == chain_id and e1_number in index_dist_s and e2_number in index_dist_s:
                    e1.append(index_dist_s[e1_number])
                    e2.append(index_dist_s[e2_number])
                    rna_feature[index_dist_s[e1_number]][2] = 1
                    rna_feature[index_dist_s[e2_number]][2] = 1
                else:
                    continue
        if os.path.exists(secondary_data_path + f"{pdb_name}-non-canonical.csv"):
            base_phoshpate = pd.read_csv(secondary_data_path + f"{pdb_name}-non-canonical.csv", sep=';')
            for interaction in base_phoshpate["Base-pair"]:
                interaction = interaction.replace(" ", "").replace("\n", "")
                residue_name = interaction.split('-')
                e1_number = extract_numbers(residue_name[0])
                e2_number = extract_numbers(residue_name[1])
                if residue_name[0][0] == chain_id and residue_name[1][
                    0] == chain_id and e1_number in index_dist_s and e2_number in index_dist_s:
                    e1.append(index_dist_s[e1_number])
                    e2.append(index_dist_s[e2_number])
                    rna_feature[index_dist_s[e1_number]][3] = 1
                    rna_feature[index_dist_s[e2_number]][3] = 1
                else:
                    continue
        start_node_index = range(length)[:-1]
        end_node_index = range(length)[1:]
        e1.extend(start_node_index)
        e1.extend(end_node_index)
        e2.extend(end_node_index)
        e2.extend(start_node_index)
        for i in range(length):
            rna_feature[i][4] = e1.count(i)
        all_edge.append([e1, e2])
        all_feature.append(rna_feature)
    return all_feature, all_edge
