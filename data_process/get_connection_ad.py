import os
import re
import numpy as np
from Bio.PDB import PDBParser


def calculate_contact_matrix(input_file, chain_id, cutoff, mode, output_dir):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB_structure', input_file)
    residue_id = ['A', 'U', 'C', 'G', 'DA', 'DU', 'DC', 'DG', 'PSU', 'CBV', '5BU', 'UMS', 'CSL', 'CCC', 'GTP', 'GDP',
                  'A23', 'U37', 'IU']
    atoms = []
    model = structure[0]
    count = 0
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                if residue.get_resname().replace(" ", "") in residue_id:
                    count = count + 1
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
    contact_matrix = np.zeros((count, count), dtype=int)
    for i in range(number_of_atoms):
        for j in range(number_of_atoms):
            if abs(revised_amino_acid_numbers[i] - revised_amino_acid_numbers[j]) <= 1:
                continue
            distance = np.linalg.norm(atom_positions[i] - atom_positions[j])
            if distance <= cutoff:
                contact_matrix[revised_amino_acid_numbers[i] - 1, revised_amino_acid_numbers[j] - 1] = 1
    return contact_matrix
