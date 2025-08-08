import pickle
from Bio.Align.Applications import ClustalwCommandline
from Bio import AlignIO
from math import log2
from collections import Counter
import os
from msa import get_msa
import argparse
from scipy.stats import pearsonr

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
        if  not seq_id.startswith("seq") and len(seq_id) == 5:
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

def get_score(clustalw_exe, input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    all_conservation_scores = {}

    for input_fasta in sorted(os.listdir(input_directory)):
        if input_fasta.endswith(".fasta"):
            input_fasta_path = os.path.join(input_directory, input_fasta)
            output_aln = os.path.join(output_directory, f"{os.path.splitext(input_fasta)[0]}_aligned.aln")

            sequence_count = 0
            with open(input_fasta_path, 'r') as file:
                for line in file:
                    if line.startswith('>'):  # Each sequence starts with '>'
                        sequence_count += 1
            if sequence_count > 1:
                clustalw_cline = ClustalwCommandline(clustalw_exe, infile=input_fasta_path, outfile=output_aln)
                try:
                    stdout, stderr = clustalw_cline()
                except Exception as e:
                    print(f"Error calling ClustalW: {e}")
                    print(input_fasta)
                    continue
            else:
                # Re-write the output in the desired Clustal format with two blank lines
                with open(output_aln, 'w') as file:
                    file.write("CLUSTAL W (2.1) multiple sequence alignment\n\n\n")  # 2 blank lines after header
                    # Write the single sequence in Clustal format
                    with open(input_fasta_path, 'r') as fasta_file:
                        for line in fasta_file:
                            if line.startswith('>'):
                                # Write the header
                                file.write(f"{line.strip()[1:]: <15}")
                            else:
                                # Write the sequence
                                file.write(line.strip())
                    file.write("\n")

            try:
                alignment = AlignIO.read(output_aln, "clustal")
                print(f"Alignment file loaded successfully: {output_aln}")
            except Exception as e:
                print(f"Failed to read alignment file: {e}")
                print(input_fasta)
                continue

            conservation_scores = conservation_score(alignment)
            all_conservation_scores[input_fasta] = conservation_scores

    score_all = []
    for fasta_file, scores in all_conservation_scores.items():
        score_s = []
        for i, score in enumerate(scores):
            score_s.append(score)
        score_all.append(score_s)


def main():
    parser = argparse.ArgumentParser(description="Run MSA and conservation score pipeline")

    parser.add_argument("--clustalw", type=str, required=True,
                        help="Path to ClustalW executable (e.g. /path/to/clustalw2)")
    parser.add_argument("--fasta", type=str, required=True,
                        help="Path to input FASTA files")
    parser.add_argument("--msa_out", type=str, required=True,
                        help="Path to save intermediate MSA results")
    parser.add_argument("--aligned_out", type=str, required=True,
                        help="Path to save final alignment and conservation scores")

    args = parser.parse_args()

    os.makedirs(args.msa_out, exist_ok=True)
    os.makedirs(args.aligned_out, exist_ok=True)

    get_msa(args.fasta, args.msa_out)
    get_score(args.clustalw, args.msa_out, args.aligned_out)


if __name__ == "__main__":
    main()



