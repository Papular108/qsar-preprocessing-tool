from rdkit import Chem
import sys
from rdkit.Chem import RDConfig
sys.path.append(RDConfig.RDContribDir + "/SA_Score")
import sascorer
from rdkit.Chem import Descriptors
from molvs import Standardizer
standardizer = Standardizer()
from rdkit.Chem.SaltRemover import SaltRemover
salt_remover = SaltRemover()
from rdkit.Chem import FilterCatalog
pains_params = FilterCatalog.FilterCatalogParams()
pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(pains_params)
brenk_params = FilterCatalog.FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)


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


def check_veber(mol):
    """
    Check a molecule against Veber's rules (oral bioavailability).

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (passes, descriptors, reason)
    """
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    passes = rotatable_bonds <= 10 and tpsa <= 140

    descriptors = {"RotatableBonds": rotatable_bonds, "TPSA": tpsa}

    reason = None
    if not passes:
        reason = f"Veber violation (RotatableBonds={rotatable_bonds}, TPSA={tpsa:.1f})"

    return passes, descriptors, reason


def check_ghose(mol):
    """
    Check a molecule against Ghose filter (drug-like property ranges).

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (passes, descriptors, reason)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    atom_count = mol.GetNumAtoms()

    passes = (
        160 <= mw <= 480
        and -0.4 <= logp <= 5.6
        and 40 <= mr <= 130
        and 20 <= atom_count <= 70
    )

    descriptors = {"MW": mw, "LogP": logp, "MolarRefractivity": mr, "AtomCount": atom_count}

    reason = None
    if not passes:
        reason = f"Ghose violation (MW={mw:.1f}, LogP={logp:.2f}, MR={mr:.1f}, Atoms={atom_count})"

    return passes, descriptors, reason


def check_egan(mol):
    """
    Check a molecule against Egan's egg boundary (LogP vs TPSA).

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (passes, descriptors, reason)
    """
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    passes = -1 <= logp <= 5.88 and tpsa <= 131.6

    descriptors = {"LogP": logp, "TPSA": tpsa}

    reason = None
    if not passes:
        reason = f"Egan violation (LogP={logp:.2f}, TPSA={tpsa:.1f})"

    return passes, descriptors, reason


def check_muegge(mol):
    """
    Check a molecule against Muegge's pharmacophore-like rules.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (passes, descriptors, reason)
    """
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rings = Descriptors.RingCount(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    passes = (
        200 <= mw <= 600
        and -2 <= logp <= 5
        and tpsa <= 150
        and rings <= 7
        and hbd <= 5
        and hba <= 10
        and rotatable_bonds <= 15
    )

    descriptors = {
        "MW": mw, "LogP": logp, "TPSA": tpsa, "Rings": rings,
        "HBD": hbd, "HBA": hba, "RotatableBonds": rotatable_bonds,
    }

    reason = None
    if not passes:
        reason = f"Muegge violation (MW={mw:.1f}, LogP={logp:.2f}, TPSA={tpsa:.1f}, Rings={rings})"

    return passes, descriptors, reason


def check_brenk(mol):
    """
    Check a molecule against the Brenk filter catalog (structural alerts for unwanted groups).

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (is_brenk_hit, matched_name)
    """
    match = brenk_catalog.GetFirstMatch(mol)

    if match is not None:
        return True, match.GetDescription()

    return False, None


def compute_synthetic_accessibility(mol):
    """
    Compute the Synthetic Accessibility (SA) score for a molecule.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (sa_score, error_message)
            sa_score (float): 1 = easy to synthesize, 10 = very difficult
            error_message (str or None)
    """
    try:
        score = sascorer.calculateScore(mol)
        return score, None
    except Exception as e:
        return None, f"SA score computation failed: {str(e)}"


def run_preprocessing_pipeline(
    smiles_list,
    lipinski_max_violations=1,
    enable_veber=False,
    enable_ghose=False,
    enable_egan=False,
    enable_muegge=False,
    enable_brenk=False,
    enable_sa_score=False,
):
    """
    Run the full preprocessing pipeline on a list of SMILES strings.

    Parameters:
        smiles_list (list): list of raw SMILES strings
        lipinski_max_violations (int): threshold passed to check_lipinski
        enable_veber (bool): apply Veber's rule filter
        enable_ghose (bool): apply Ghose filter
        enable_egan (bool): apply Egan filter
        enable_muegge (bool): apply Muegge filter
        enable_brenk (bool): apply Brenk structural alert filter
        enable_sa_score (bool): compute SA score for kept molecules (informational, not a filter)

    Returns:
        dict: {
            "kept_mols": list of surviving RDKit Mol objects,
            "kept_smiles": list of their canonical SMILES,
            "audit_trail": list of dicts, one per step, with counts,
            "removed_log": list of dicts, one per removed molecule, with reason,
            "sa_scores": list of SA scores per kept molecule (only if enable_sa_score=True), else None
        }
    """
    audit_trail = []
    removed_log = []

    # Step 1: parse
    current = []
    for i, smiles in enumerate(smiles_list):
        mol, error = parse_smiles(smiles)
        if mol is None:
            removed_log.append({"original_index": i, "smiles": smiles, "step": "parse_smiles", "reason": error})
        else:
            current.append((i, mol))
    audit_trail.append({"step": "parse_smiles", "input_count": len(smiles_list), "output_count": len(current)})

    # Step 2: standardize
    next_step = []
    for i, mol in current:
        std_mol, error = standardize_molecule(mol)
        if std_mol is None:
            removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "standardize_molecule", "reason": error})
        else:
            next_step.append((i, std_mol))
    audit_trail.append({"step": "standardize_molecule", "input_count": len(current), "output_count": len(next_step)})
    current = next_step

    # Step 3: strip salts
    next_step = []
    for i, mol in current:
        stripped_mol, error = strip_salts(mol)
        if stripped_mol is None or stripped_mol.GetNumAtoms() == 0:
            reason = error if error else "Salt stripping removed entire molecule"
            removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "strip_salts", "reason": reason})
        else:
            next_step.append((i, stripped_mol))
    audit_trail.append({"step": "strip_salts", "input_count": len(current), "output_count": len(next_step)})
    current = next_step

    # Step 4: Lipinski filter
    next_step = []
    for i, mol in current:
        passes, descriptors, reason = check_lipinski(mol, max_violations=lipinski_max_violations)
        if not passes:
            removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_lipinski", "reason": reason})
        else:
            next_step.append((i, mol))
    audit_trail.append({"step": "check_lipinski", "input_count": len(current), "output_count": len(next_step)})
    current = next_step

    # Step 5: PAINS filter
    next_step = []
    for i, mol in current:
        is_pains, pains_name = check_pains(mol)
        if is_pains:
            removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_pains", "reason": f"PAINS match: {pains_name}"})
        else:
            next_step.append((i, mol))
    audit_trail.append({"step": "check_pains", "input_count": len(current), "output_count": len(next_step)})
    current = next_step

    # Step 5b: Brenk filter (optional)
    if enable_brenk:
        next_step = []
        for i, mol in current:
            is_brenk, brenk_name = check_brenk(mol)
            if is_brenk:
                removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_brenk", "reason": f"Brenk match: {brenk_name}"})
            else:
                next_step.append((i, mol))
        audit_trail.append({"step": "check_brenk", "input_count": len(current), "output_count": len(next_step)})
        current = next_step

    # Step 5c: Veber filter (optional)
    if enable_veber:
        next_step = []
        for i, mol in current:
            passes, descriptors, reason = check_veber(mol)
            if not passes:
                removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_veber", "reason": reason})
            else:
                next_step.append((i, mol))
        audit_trail.append({"step": "check_veber", "input_count": len(current), "output_count": len(next_step)})
        current = next_step

    # Step 5d: Ghose filter (optional)
    if enable_ghose:
        next_step = []
        for i, mol in current:
            passes, descriptors, reason = check_ghose(mol)
            if not passes:
                removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_ghose", "reason": reason})
            else:
                next_step.append((i, mol))
        audit_trail.append({"step": "check_ghose", "input_count": len(current), "output_count": len(next_step)})
        current = next_step

    # Step 5e: Egan filter (optional)
    if enable_egan:
        next_step = []
        for i, mol in current:
            passes, descriptors, reason = check_egan(mol)
            if not passes:
                removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_egan", "reason": reason})
            else:
                next_step.append((i, mol))
        audit_trail.append({"step": "check_egan", "input_count": len(current), "output_count": len(next_step)})
        current = next_step

    # Step 5f: Muegge filter (optional)
    if enable_muegge:
        next_step = []
        for i, mol in current:
            passes, descriptors, reason = check_muegge(mol)
            if not passes:
                removed_log.append({"original_index": i, "smiles": Chem.MolToSmiles(mol), "step": "check_muegge", "reason": reason})
            else:
                next_step.append((i, mol))
        audit_trail.append({"step": "check_muegge", "input_count": len(current), "output_count": len(next_step)})
        current = next_step

    # Step 6: deduplicate
    just_mols = [mol for i, mol in current]
    unique_mols, duplicate_info = deduplicate_molecules(just_mols)
    for dup in duplicate_info:
        original_index = current[dup["index"]][0]
        removed_log.append({"original_index": original_index, "smiles": dup["smiles"], "step": "deduplicate_molecules", "reason": f"Duplicate of index {current[dup['duplicate_of_index']][0]}"})
    audit_trail.append({"step": "deduplicate_molecules", "input_count": len(current), "output_count": len(unique_mols)})

    kept_smiles = [Chem.MolToSmiles(mol) for mol in unique_mols]

    sa_scores = None
    if enable_sa_score:
        sa_scores = []
        for mol in unique_mols:
            score, error = compute_synthetic_accessibility(mol)
            sa_scores.append(score)

    return {
        "kept_mols": unique_mols,
        "kept_smiles": kept_smiles,
        "audit_trail": audit_trail,
        "removed_log": removed_log,
        "sa_scores": sa_scores,
    }
