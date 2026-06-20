from rdkit import Chem
from rdkit.Chem import Descriptors
from molvs import Standardizer
standardizer = Standardizer()
from rdkit.Chem.SaltRemover import SaltRemover
salt_remover = SaltRemover()
from rdkit.Chem import FilterCatalog
pains_params = FilterCatalog.FilterCatalogParams()
pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(pains_params)


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


def check_pains(mol):
    """
    Check a molecule against the PAINS filter catalog.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (is_pains, matched_name)
            is_pains (bool): True if the molecule matches a PAINS pattern
            matched_name (str or None): name of the matched PAINS pattern, or None if no match
    """
    match = pains_catalog.GetFirstMatch(mol)

    if match is not None:
        return True, match.GetDescription()

    return False, None


def deduplicate_molecules(mol_list):
    """
    Remove duplicate molecules from a list, based on canonical SMILES comparison.

    Parameters:
        mol_list (list): list of RDKit Mol objects

    Returns:
        tuple: (unique_mols, duplicate_info)
            unique_mols (list): molecules with duplicates removed, keeping the first occurrence
            duplicate_info (list): list of dicts describing each removed duplicate,
                                    e.g. {"index": 5, "duplicate_of_index": 2, "smiles": "CCO"}
    """
    seen_smiles = {}
    unique_mols = []
    duplicate_info = []

    for index, mol in enumerate(mol_list):
        canonical = Chem.MolToSmiles(mol)

        if canonical in seen_smiles:
            duplicate_info.append({
                "index": index,
                "duplicate_of_index": seen_smiles[canonical],
                "smiles": canonical,
            })
        else:
            seen_smiles[canonical] = index
            unique_mols.append(mol)

    return unique_mols, duplicate_info
