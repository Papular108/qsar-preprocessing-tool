from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Avalon import pyAvalonTools
import numpy as np
from rdkit import Chem


def compute_descriptors(mol):
    """
    Compute key physicochemical descriptors for a molecule.

    Parameters:
        mol: RDKit Mol object

    Returns:
        dict: descriptor name -> value
    """
    descriptors = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
    }

    return descriptors


def compute_fingerprint(mol, fp_type="morgan", radius=2, n_bits=2048):
    """
    Compute a molecular fingerprint for a given molecule.

    Parameters:
        mol: RDKit Mol object
        fp_type (str): one of "morgan", "maccs", "topological", "atom_pair", "torsion", "avalon"
        radius (int): radius for morgan fingerprint (ignored for other types)
        n_bits (int): length of the bit vector (ignored for maccs, which is fixed length)

    Returns:
        tuple: (fingerprint_array, error_message)
            fingerprint_array (numpy array) if successful, otherwise None
            error_message (str) if failed, otherwise None
    """
    try:
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif fp_type == "topological":
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        elif fp_type == "atom_pair":
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
        elif fp_type == "torsion":
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        elif fp_type == "avalon":
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        else:
            return None, f"Unknown fingerprint type: '{fp_type}'"

        fp_array = np.array(fp)
        return fp_array, None

    except Exception as e:
        return None, f"Fingerprint computation failed: {str(e)}"
