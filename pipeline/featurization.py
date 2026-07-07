from rdkit.Chem import Descriptors
import pandas as pd
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Avalon import pyAvalonTools
import numpy as np
from rdkit import Chem, DataStructs


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




def compute_esol(mol):
    """
    Predict aqueous solubility using the ESOL model (Delaney, 2004).

    LogS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RotBonds - 0.74*AromaticProportion

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (log_s, solubility_mg_ml, solubility_mol_l, solubility_class, error)
    """
    try:
        clogp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy == 0:
            return None, None, None, None, "No heavy atoms"
        n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_proportion = n_aromatic / n_heavy

        log_s = 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * aromatic_proportion

        solubility_mol_l = 10 ** log_s
        solubility_mg_ml = solubility_mol_l * mw

        if log_s > 0:
            sol_class = "Highly soluble"
        elif log_s > -1:
            sol_class = "Soluble"
        elif log_s > -2:
            sol_class = "Moderately soluble"
        elif log_s > -4:
            sol_class = "Slightly soluble"
        elif log_s > -6:
            sol_class = "Insoluble"
        else:
            sol_class = "Poorly soluble"

        return log_s, solubility_mg_ml, solubility_mol_l, sol_class, None
    except Exception as e:
        return None, None, None, None, f"ESOL computation failed: {str(e)}"


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


def _mol_to_fp_obj(mol, fp_type="morgan", radius=2, n_bits=2048):
    """Return an RDKit fingerprint object (not numpy array) for Tanimoto calculation."""
    if fp_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == "fcfp":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True)
    elif fp_type == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "topological":
        return Chem.RDKFingerprint(mol, fpSize=n_bits)
    elif fp_type == "atom_pair":
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    elif fp_type == "torsion":
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    elif fp_type == "avalon":
        return pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    elif fp_type == "pattern":
        return Chem.PatternFingerprint(mol, fpSize=n_bits)
    elif fp_type == "layered":
        return Chem.LayeredFingerprint(mol, fpSize=n_bits)
    return None


def find_similar_molecules(query_mol, target_mols, top_n=10, fp_type="morgan", radius=2, n_bits=2048):
    """
    Find the most similar molecules to a query using Tanimoto similarity.

    Parameters:
        query_mol: RDKit Mol object for the query
        target_mols: list of RDKit Mol objects to search
        top_n: number of top results to return
        fp_type: fingerprint type
        radius: Morgan radius (only for morgan/fcfp)
        n_bits: fingerprint bit length

    Returns:
        list of (index, mol, similarity_score) sorted by similarity descending
    """
    if not target_mols:
        return []

    query_fp = _mol_to_fp_obj(query_mol, fp_type, radius, n_bits)
    if query_fp is None:
        return []

    results = []
    for i, mol in enumerate(target_mols):
        if mol is None:
            continue
        target_fp = _mol_to_fp_obj(mol, fp_type, radius, n_bits)
        if target_fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(query_fp, target_fp)
        results.append((i, mol, sim))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def batch_similarity_search(query_mols, target_mols, top_n=5, threshold=0.4,
                            fp_type="morgan", radius=2, n_bits=2048):
    """
    Search multiple reference molecules against a target set.

    Computes target fingerprints once for efficiency.

    Parameters:
        query_mols: list of (name, RDKit Mol) tuples
        target_mols: list of RDKit Mol objects
        top_n: max hits per query
        threshold: minimum Tanimoto to include
        fp_type, radius, n_bits: fingerprint settings

    Returns:
        dict mapping query_name -> list of (target_index, similarity) sorted descending
    """
    if not target_mols or not query_mols:
        return {}

    # Pre-compute target fingerprints once
    target_fps = []
    for mol in target_mols:
        if mol is not None:
            target_fps.append(_mol_to_fp_obj(mol, fp_type, radius, n_bits))
        else:
            target_fps.append(None)

    results = {}
    for name, q_mol in query_mols:
        if q_mol is None:
            results[name] = []
            continue
        q_fp = _mol_to_fp_obj(q_mol, fp_type, radius, n_bits)
        if q_fp is None:
            results[name] = []
            continue
        hits = []
        for i, t_fp in enumerate(target_fps):
            if t_fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(q_fp, t_fp)
            if sim >= threshold:
                hits.append((i, sim))
        hits.sort(key=lambda x: x[1], reverse=True)
        results[name] = hits[:top_n]

    return results
