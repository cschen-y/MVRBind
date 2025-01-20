from Bio.PDB import PDBList
from Bio import SeqIO
import requests


def download_pdb_and_fasta(pdb_id_list, output_dir):
    for pdb_id in pdb_id_list:
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
        fasta_path = f"{output_dir}/{pdb_id}.fasta"
        response = requests.get(fasta_url)
        if response.status_code == 200:
            with open(fasta_path, "w") as fasta_file:
                fasta_file.write(response.text)


output_directory = ""
pdb_id_to_download = []
download_pdb_and_fasta(pdb_id_to_download, output_directory)
