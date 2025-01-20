import os
import pickle

from Bio.Align.Applications import ClustalwCommandline
from Bio import AlignIO
from math import log2
from collections import Counter
import os

from scipy.stats import pearsonr

def calculate_entropy(column):
    total = len(column)
    frequencies = Counter(column)
    entropy = 0
    for base, count in frequencies.items():
        if base != '-':  # 排除gap
            freq = count / total
            entropy -= freq * log2(freq)
    return entropy

def conservation_score(alignment):
    num_positions = alignment.get_alignment_length()
    scores = []
    max_entropy = log2(4)
    sequence_ids = [record.id for record in alignment]
    first_seq_index = None
    for index, seq_id in enumerate(sequence_ids):
        if  not seq_id.startswith("seq"):
            first_seq_index = index
            break

    for i in range(num_positions):
        column = alignment[:, i]
        if column[first_seq_index] == '-':
            continue
        entropy = calculate_entropy(column)
        conservation = 1 - (entropy / max_entropy)
        scores.append(conservation)

    return scores

def get_msa(file_path_aln):
    conservation_scores = 0
    try:
        alignment = AlignIO.read(file_path_aln, "clustal")
        conservation_scores = conservation_score(alignment)
    except Exception as e:
        print(f"读取对齐文件时出错: {e}")
        print(file_path_aln)

    return conservation_scores