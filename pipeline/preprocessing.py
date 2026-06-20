from rdkit import Chem
from rdkit.Chem import Descriptors
from molvs import Standardizer
standardizer = Standardizer()
from rdkit.Chem.SaltRemover import SaltRemover
salt_remover = SaltRemover()


def parse_smiles(smiles):
    """
    Convert a SMILES string into an RDKit Mol object.

    Parameters:
        smiles (str): SMILES string representing a molecule

    Returns:
        tuple: (mol, error_message)
            mol is an RDKit Mol object if successful, otherwise None
            error_message is None if successful, otherwise a string explaining what went wrong
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, f"Could not parse SMILES: '{smiles}'"

    return mol, None


def standardize_molecule(mol):
    """
    Standardize an RDKit Mol object using MolVS.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (standardized_mol, error_message)
            standardized_mol is the cleaned Mol object if successful, otherwise None
            error_message is None if successful, otherwise a string explaining what went wrong
    """
    try:
        standardized_mol = standardizer.standardize(mol)
        return standardized_mol, None
    except Exception as e:
        return None, f"Standardization failed: {str(e)}"


def strip_salts(mol):
    """
    Remove salt fragments from an RDKit Mol object.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (stripped_mol, error_message)
            stripped_mol is the molecule with salts removed if successful, otherwise None
            error_message is None if successful, otherwise a string explaining what went wrong
    """
    try:
        stripped_mol = salt_remover.StripMol(mol)
        return stripped_mol, None
    except Exception as e:
        return None, f"Salt stripping failed: {str(e)}"


def check_lipinski(mol, max_violations=1):
    """
    Check a molecule against Lipinski's Rule of Five.

    Parameters:
        mol: RDKit Mol object
        max_violations (int): maximum allowed rule violations (default 1, standard threshold)

    Returns:
        tuple: (passes, descriptors, reason)
            passes (bool): True if violations <= max_violations
            descriptors (dict): MW, LogP, HBD, HBA, and violation count
            reason (str or None): explanation if it fails, otherwise None
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    descriptors = {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "violations": violations,
    }

    passes = violations <= max_violations

    if not passes:
        reason = f"Lipinski violations = {violations} (MW={mw:.1f}, LogP={logp:.2f}, HBD={hbd}, HBA={hba})"
    else:
        reason = None

    return passes, descriptors, reason
