import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import os
import torch
from data_process.set_seed import set_seed
from sklearn.cluster import KMeans
from torch_geometric.loader import DataLoader
from model import MVRBind
from data_process.get_pdb_feature import TestDataset
from Bio import SeqIO

def print_sequences(fasta_dir):
    for file_name in sorted(os.listdir(fasta_dir)):
        if file_name.endswith(".fasta") or file_name.endswith(".fa"):
            fasta_path = os.path.join(fasta_dir, file_name)
            for record in SeqIO.parse(fasta_path, "fasta"):
                print(f">{record.id}")
                print(record.seq)


def predict(args):

    set_seed(1)

    apo_dataset = TestDataset(
        root=args.root,
        pdb_file_path=args.pdb,
        fasta_file_path=args.fasta,
        msa_file_path=args.msa,
        em_file_path=args.em,
        asa_file_path=args.asa,
        top_k=8,
        label_file_path="",
        mode="predict"
    )

    apo_loader = DataLoader(apo_dataset, batch_size=1)
    model = MVRBind(136)
    model.load_state_dict(torch.load("model_parameters/model.pt"))
    model.eval()


    print_sequences(args.fasta)

    with torch.no_grad():
        for idx, data in enumerate(apo_loader):
            output = model(data)
            output_np = output.detach().numpy().reshape(-1, 1)

            if len(output_np) < 2:
                continue

            kmeans = KMeans(n_clusters=2, random_state=0).fit(output_np)
            centers = kmeans.cluster_centers_.flatten()
            threshold = np.mean(centers)

            binary_predictions = (output_np > threshold).astype(int).flatten().tolist()
            print(f"predict binding sites: {binary_predictions}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict RNA binding sites using MVRBind")

    parser.add_argument('--root', type=str, required=True, help='Path to root directory for saving processed data (e.g., ./pt/pdb_id_pt)')
    parser.add_argument('--pdb', type=str, required=True, help='Path to PDB file directory')
    parser.add_argument('--fasta', type=str, required=True, help='Path to FASTA file directory')
    parser.add_argument('--msa', type=str, required=True, help='Path to MSA file directory')
    parser.add_argument('--em', type=str, required=True, help='Path to EM feature file directory')
    parser.add_argument('--asa', type=str, required=True, help='Path to ASA file directory')

    args = parser.parse_args()
    predict(args)
