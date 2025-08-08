from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO
import os
import random
import time
import pickle
from Bio.Align.Applications import ClustalwCommandline
from Bio import AlignIO
from math import log2
from collections import Counter

def calculate_entropy(column):
    total = len(column)
    frequencies = Counter(column)
    entropy = 0
    for base, count in frequencies.items():
        if base != '-':
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

def get_msa_score(file_path_aln):
    conservation_scores = 0
    try:
        alignment = AlignIO.read(file_path_aln, "clustal")
        conservation_scores = conservation_score(alignment)
    except Exception as e:
        print(f"读取对齐文件时出错: {e}")
        print(file_path_aln)

    return conservation_scores

def get_msa(fasta_directory,output_directory):
    os.makedirs(output_directory, exist_ok=True)

    fasta_files = [f for f in os.listdir(fasta_directory) if f.endswith('.fasta')]
    for fasta_file in fasta_files:
        file_path = os.path.join(fasta_directory, fasta_file)
        msa_file_name = fasta_file.replace('.fasta', '_msa.fasta')
        msa_file_path = os.path.join(output_directory, msa_file_name)
        if os.path.exists(msa_file_path):
            print(f"{msa_file_path}")
            continue
        sequence = SeqIO.read(file_path, format="fasta")
        result_handle = NCBIWWW.qblast("blastn", "nt", sequence.seq)
        count = 1
        with open(msa_file_path, "w") as msa_file:
            msa_file.write(f">{sequence.id}\n{sequence.seq}\n")
            written_sequences = {sequence.seq}
            blast_record = NCBIXML.read(result_handle)
            for idx, alignment in enumerate(blast_record.alignments):
                for hsp in alignment.hsps:
                    subject_sequence = hsp.sbjct
                    if subject_sequence not in written_sequences:
                        msa_file.write(f">seq{count}_{fasta_file.replace('.fasta', '')}\n{subject_sequence}\n")
                        written_sequences.add(subject_sequence)
                        count += 1

        result_handle.close()
        print(f"MSA FASTA文件已生成：{msa_file_path}")

