import os.path
from itertools import chain

from Bio.PDB import PDBParser, PDBIO, Chain, Model, Structure, Residue, Atom
import requests
import pandas as pd
import ast
from Bio.PDB import PDBParser, Polypeptide
import os
import argparse
from downpdb import download_and_extract_chain


ions_and_unwanted = {
    'H2O', 'HOH', "NA", "K", "MG", "CA", "AL", "FE", "ZN", "CU", "MN", "CO", "NI", "AG", "PB", "AU",
    "CR", "TI", "MO", "SR", "CL", 'IR', 'IRI', 'BA', 'CBR', 'O2Z', 'PO4', 'HG'
}


def get_parent_residue(resname):
    url = f'https://data.rcsb.org/rest/v1/core/chemcomp/{resname}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        parent = data.get('chem_comp', {}).get('mon_nstd_parent_comp_id')
        if parent and parent[0] in ['A', 'U', 'G', 'C']:
            return parent
    return None


def create_new_chain_with_renumbered_residues(pdb_path, output_path, source_chain_id, new_chain_id, ligand=set()):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_path)
    model = structure[0]

    source_chain = model[source_chain_id]
    new_chain = Chain.Chain(new_chain_id)

    new_resseq = 1
    ligands = []

    for residue in source_chain:
        hetfield, _, icode = residue.id
        resname = residue.get_resname().strip()

        if resname in ions_and_unwanted:
            continue

        if resname in ligand:
            continue

        if resname not in ['A', 'U', 'G', 'C']:
            parent = get_parent_residue(resname)
            if parent and parent[0] in ['A', 'U', 'G', 'C']:
                resname = parent[0]
            else:
                # print(pdb_path, resname)
                continue

        new_residue = Residue.Residue((' ', new_resseq, ' '), resname, residue.segid)
        for atom in residue:
            new_atom = Atom.Atom(
                atom.get_name(),
                atom.get_coord(),
                atom.get_bfactor(),
                atom.get_occupancy(),
                atom.get_altloc(),
                atom.get_fullname(),
                atom.get_serial_number(),
                atom.element
            )
            new_residue.add(new_atom)

        new_chain.add(new_residue)
        new_resseq += 1

    new_model = Model.Model(0)
    new_model.add(new_chain)
    new_structure = Structure.Structure("modified")
    new_structure.add(new_model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)

    if ligands:
        print(f"Ligands skipped (but saved info):")
        for name, resid, atoms in ligands:
            print(f"  - {name} at {resid} ({len(atoms)} atoms)")
    return new_resseq


def get_pdb(pdb_file_path):
    pdb_id_list = os.listdir(pdb_file_path)

    for pdb_id in pdb_id_list:
        pdb_path = os.path.join(pdb_file_path, pdb_id)
        chain_id = pdb_id[4]
        new_chain_id = pdb_id[4]
        new_pdb_path = pdb_path
        ligand_set = {}
        create_new_chain_with_renumbered_residues(
            pdb_path, new_pdb_path, source_chain_id=chain_id, new_chain_id=new_chain_id, ligand=ligand_set
        )

def pdb_to_fasta(pdb_dir, fasta_dir):
    parser = PDBParser(QUIET=True)

    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    base_map = {
        'A': 'A',
        'G': 'G',
        'C': 'C',
        'U': 'U',
    }

    for pdb_file in os.listdir(pdb_dir):
        if not pdb_file.endswith(".pdb"):
            continue
        pdb_path = os.path.join(pdb_dir, pdb_file)
        structure = parser.get_structure('structure', pdb_path)
        model = structure[0]

        for chain in model:
            seq = []
            for residue in chain:
                resname = residue.get_resname().strip()
                one_letter = base_map.get(resname, 'N')
                seq.append(one_letter)
            fasta_seq = ''.join(seq)

            fasta_filename = pdb_file.replace('.pdb', '.fasta')
            fasta_path = os.path.join(fasta_dir, fasta_filename)
            with open(fasta_path, 'w') as f:
                f.write(f">{pdb_file.replace('.pdb', '')}\n")
                for i in range(0, len(fasta_seq), 80):
                    f.write(fasta_seq[i:i+80] + "\n")

            print(f"Saved FASTA for {pdb_file} chain {chain.id} to {fasta_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess PDB and convert to FASTA")
    parser.add_argument('--pdb_id_chain', type=str, default='1AJUA',
                        help="PDB ID with chain (e.g., 1AJUA), default='1AJUA'")
    parser.add_argument('--pdb_dir', type=str, default='./out_test/pdb',
                        help="Directory to save PDB files (default='./out_test/pdb')")
    args = parser.parse_args()

    pdb_id_chain = args.pdb_id_chain.upper()
    pdb_file_path = args.pdb_dir
    fasta_dir = pdb_file_path.replace("pdb", "fasta")

    pdb_id = pdb_id_chain[:-1]
    pdb_chain = pdb_id_chain[-1]

    os.makedirs(pdb_file_path, exist_ok=True)
    os.makedirs(fasta_dir, exist_ok=True)

    download_and_extract_chain(pdb_id, pdb_chain, pdb_file_path)

    get_pdb(pdb_file_path)

    pdb_to_fasta(pdb_file_path, fasta_dir)

    print("✅ PDB downloaded, cleaned, and converted to FASTA.")


if __name__ == '__main__':
    main()

