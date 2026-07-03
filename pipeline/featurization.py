from rdkit.Chem import Descriptors
import pandas as pd
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
        fp_type (str): one of "morgan", "fcfp", "maccs", "topological", "atom_pair", "torsion", "avalon", "pattern", "layered"
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
        elif fp_type == "fcfp":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True)
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
        elif fp_type == "pattern":
            fp = Chem.PatternFingerprint(mol, fpSize=n_bits)
        elif fp_type == "layered":
            fp = Chem.LayeredFingerprint(mol, fpSize=n_bits)
        else:
            return None, f"Unknown fingerprint type: '{fp_type}'"

        fp_array = np.array(fp)
        return fp_array, None

    except Exception as e:
        return None, f"Fingerprint computation failed: {str(e)}"




def featurize_dataset(mol_list, fp_type="morgan", radius=2, n_bits=2048):
    """
    Compute descriptors and fingerprints for a list of molecules,
    returning one combined DataFrame.

    Parameters:
        mol_list (list): list of RDKit Mol objects
        fp_type (str): fingerprint type, passed to compute_fingerprint
        radius (int): radius for morgan fingerprint
        n_bits (int): fingerprint length

    Returns:
        tuple: (dataframe, errors)
            dataframe (pandas DataFrame): one row per molecule, descriptor columns + fingerprint bit columns
            errors (list): list of dicts for molecules that failed, with index and reason
    """
    rows = []
    errors = []

    for index, mol in enumerate(mol_list):
        descriptors = compute_descriptors(mol)
        fp_array, fp_error = compute_fingerprint(mol, fp_type=fp_type, radius=radius, n_bits=n_bits)

        if fp_error:
            errors.append({"index": index, "reason": fp_error})
            continue

        row = {"SMILES": Chem.MolToSmiles(mol)}
        row.update(descriptors)
        for bit_index, bit_value in enumerate(fp_array):
            row[f"{fp_type}_bit_{bit_index}"] = bit_value

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    return dataframe, errors
