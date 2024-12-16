from biotite.structure.io.pdb import PDBFile
import biotite.structure as struc
import numpy as np
import warnings; warnings.filterwarnings('ignore')
import torch



def place_fourth_atom(a, b, c, bond_angle, torsion, bond_length):
    # Place atom D with respect to atom C at origin.
    d = np.array(
        [
            bond_length * np.cos(np.pi - bond_angle),
            bond_length * np.cos(torsion) * np.sin(bond_angle),
            bond_length * np.sin(torsion) * np.sin(bond_angle),
        ]
    ).T

    # Transform atom D to the correct frame.
    bc = c - b
    bc /= np.linalg.norm(bc)  # Unit vector from B to C.

    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)  # Normal vector of the plane defined by a, b, c.

    M = np.array([bc, np.cross(n, bc), n]).T
    return M @ d + c



def angles2coord(angles, n_ca=1.46, ca_c=1.54, c_n=1.33):
    """Given L x 6 angle matrix,
    reconstruct the Cartesian coordinates of atoms.
    Returns L x 3 coordinate matrix.

    Implements NeRF (Natural Extension Reference Frame) algorithm.
    """
    if isinstance(angles, torch.Tensor):
        phi, psi, omega, theta1, theta2, theta3 = angles.T.numpy()
    else:
        phi, psi, omega, theta1, theta2, theta3 = angles.T


    torsions = np.stack([psi[:-1], omega[:-1], phi[1:]], axis=-1).flatten()
    bond_angles = np.stack([theta2[:-1], theta3[:-1], theta1[1:]], axis=-1).flatten()

    #
    # Place the first three atoms.
    #
    # The first atom (N) is placed at origin.
    a = np.zeros(3)
    # The second atom (Ca) is placed on the x-axis.
    b = np.array([1, 0, 0]) * n_ca
    # The third atom (C) is placed on the xy-plane with bond angle theta1[0]
    c = np.array([np.cos(np.pi - theta1[0]), np.sin(np.pi - theta1[0]), 0]) * ca_c + b

    # Iteratively place the fourth atom based on the last three atoms.

    coords = [a, b, c]
    # cycle through [n, ca, c, n, ca, c, ...]
    for i, bond_length in enumerate([c_n, n_ca, ca_c] * (len(angles) - 1)):
        torsion, bond_angle = torsions[i], bond_angles[i]
        d = place_fourth_atom(a, b, c, bond_angle, torsion, bond_length)
        coords.append(d)

        a, b, c = b, c, d

    return np.array(coords)




def coor_to_pdb(coords,file_name):
    res_name = [
        "THR", "THR", "THR", "LEU", "LEU", "LEU", "TYR", "TYR", "TYR", "ALA", "ALA", "ALA", "PRO", "PRO", "PRO", "GLY", "GLY", "GLY", "GLY", "GLY", "GLY", "TYR", "TYR", "TYR", "ASP", "ASP", "ASP", "ILE", "ILE", "ILE", "MET", "MET", "MET", "GLY", "GLY", "GLY", "TYR", "TYR", "TYR", "LEU", "LEU", "LEU", "ILE", "ILE", "ILE", "GLN", "GLN", "GLN", "ILE", "ILE", "ILE", "MET", "MET", "MET", "ASN", "ASN", "ASN", "ARG", "ARG", "ARG", "PRO", "PRO", "PRO", "ASN", "ASN", "ASN", "PRO", "PRO", "PRO", "GLN", "GLN", "GLN", "VAL", "VAL", "VAL", "GLU", "GLU", "GLU", "LEU", "LEU", "LEU", "GLY", "GLY", "GLY", "PRO", "PRO", "PRO", "VAL", "VAL", "VAL", "ASP", "ASP", "ASP", "THR", "THR", "THR", "SER", "SER", "SER", "CYS", "CYS", "CYS", "ALA", "ALA", "ALA", "LEU", "LEU", "LEU", "ILE", "ILE", "ILE", "LEU", "LEU", "LEU", "CYS", "CYS", "CYS", "ASP", "ASP", "ASP", "LEU", "LEU", "LEU", "LYS", "LYS", "LYS", "GLN", "GLN", "GLN", "LYS", "LYS", "LYS", "ASP", "ASP", "ASP", "THR", "THR", "THR", "PRO", "PRO", "PRO", "ILE", "ILE", "ILE", "VAL", "VAL", "VAL", "TYR", "TYR", "TYR", "ALA", "ALA", "ALA", "SER", "SER", "SER", "GLU", "GLU", "GLU", "ALA", "ALA", "ALA", "PHE", "PHE", "PHE", "LEU", "LEU", "LEU", "TYR", "TYR", "TYR", "MET", "MET", "MET", "THR", "THR", "THR", "GLY", "GLY", "GLY", "TYR", "TYR", "TYR", "SER", "SER", "SER", "ASN", "ASN", "ASN", "ALA", "ALA", "ALA", "GLU", "GLU", "GLU", "VAL", "VAL", "VAL", "LEU", "LEU", "LEU", "GLY", "GLY", "GLY", "ARG", "ARG", "ARG", "ASN", "ASN", "ASN", "CYS", "CYS", "CYS", "ARG", "ARG", "ARG", "PHE", "PHE", "PHE", "LEU", "LEU", "LEU", "GLN", "GLN", "GLN", "SER", "SER", "SER", "PRO", "PRO", "PRO", "ASP", "ASP", "ASP", "GLY", "GLY", "GLY", "MET", "MET", "MET", "VAL", "VAL", "VAL", "LYS", "LYS", "LYS", "PRO", "PRO", "PRO", "LYS", "LYS", "LYS", "SER", "SER", "SER", "THR", "THR", "THR", "ARG", "ARG", "ARG", "LYS", "LYS", "LYS", "TYR", "TYR", "TYR", "VAL", "VAL", "VAL", "ASP", "ASP", "ASP", "SER", "SER", "SER", "ASN", "ASN", "ASN", "THR", "THR", "THR", "ILE", "ILE", "ILE", "ASN", "ASN", "ASN", "THR", "THR", "THR", "MET", "MET", "MET", "ARG", "ARG", "ARG", "LYS", "LYS", "LYS", "ALA", "ALA", "ALA", "ILE", "ILE", "ILE", "ASP", "ASP", "ASP", "ARG", "ARG", "ARG", "ASN", "ASN", "ASN", "ALA", "ALA", "ALA", "GLU", "GLU", "GLU", "VAL", "VAL", "VAL", "GLN", "GLN", "GLN", "VAL", "VAL", "VAL", "GLU", "GLU", "GLU", "VAL", "VAL", "VAL", "VAL", "VAL", "VAL", "ASN", "ASN", "ASN", "PHE", "PHE", "PHE", "LYS", "LYS", "LYS", "LYS", "LYS", "LYS", "ASN", "ASN", "ASN", "GLY", "GLY", "GLY", "GLN", "GLN", "GLN", "ARG", "ARG", "ARG", "PHE", "PHE", "PHE", "VAL", "VAL", "VAL", "ASN", "ASN", "ASN", "PHE", "PHE", "PHE", "LEU", "LEU", "LEU", "THR", "THR", "THR", "MET", "MET", "MET", "ILE", "ILE", "ILE", "PRO", "PRO", "PRO", "VAL", "VAL", "VAL", "ARG", "ARG", "ARG", "ASP", "ASP", "ASP", "GLU", "GLU", "GLU", "THR", "THR", "THR", "GLY", "GLY", "GLY", "GLU", "GLU", "GLU", "TYR", "TYR", "TYR", "ARG", "ARG", "ARG", "TYR", "TYR", "TYR", "SER", "SER", "SER", "MET", "MET", "MET", "GLY", "GLY", "GLY", "PHE", "PHE", "PHE", "GLN", "GLN", "GLN", "CYS", "CYS", "CYS", "GLU", "GLU", "GLU"
    ]
    coords = coords
    num_residues = len(coords) // 3
    structure = struc.AtomArray(len(coords))
    structure.coord = coords
    structure.atom_name = ["N", "CA", "C"] * num_residues
    structure.res_name = res_name[:len(coords)]
    structure.res_id = np.repeat(range(1, num_residues + 1), 3)
    pdb = PDBFile()
    pdb.set_structure(structure)
    pdb.write(f"{file_name}")

