from rdkit import Chem
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
