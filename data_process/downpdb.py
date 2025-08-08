import os
import requests
from Bio import PDB
from Bio import SeqIO
from io import StringIO

def download_and_extract_chain(pdb_id, chain_id, out_dir):
    pdb_id = pdb_id.upper()
    chain_id = chain_id.upper()

    pdb_dir = out_dir
    os.makedirs(pdb_dir, exist_ok=True)

    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_response = requests.get(pdb_url)
    pdb_response.raise_for_status()
    pdb_content = pdb_response.text

    # 解析PDB并提取指定链
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_content))

    io = PDB.PDBIO()
    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

    pdb_output = os.path.join(pdb_dir, f"{pdb_id}{chain_id}.pdb")
    io.set_structure(structure)
    io.save(pdb_output, select=ChainSelect())
    print(f"Saved chain {chain_id} of {pdb_id} to {pdb_output}")


