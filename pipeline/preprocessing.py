from rdkit import Chem
import sys
from rdkit.Chem import RDConfig
try:
    sys.path.append(RDConfig.RDContribDir + "/SA_Score")
    import sascorer
except Exception:
    sascorer = None
from rdkit.Chem import Descriptors
from molvs import Standardizer
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import FilterCatalog
from rdkit.Chem.QED import qed as _qed
import streamlit as st


@st.cache_resource
def _get_standardizer():
    return Standardizer()


@st.cache_resource
def _get_salt_remover():
    return SaltRemover()


@st.cache_resource
def _get_pains_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog.FilterCatalog(params)


@st.cache_resource
def _get_brenk_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


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
        standardized_mol = _get_standardizer().standardize(mol)
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
        stripped_mol = _get_salt_remover().StripMol(mol)
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
    match = _get_pains_catalog().GetFirstMatch(mol)

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
    match = _get_brenk_catalog().GetFirstMatch(mol)

    if match is not None:
        return True, match.GetDescription()

    return False, None


def compute_qed(mol):
    """
    Compute the Quantitative Estimate of Druglikeness (QED) for a molecule.

    Parameters:
        mol: RDKit Mol object

    Returns:
        tuple: (qed_score, error_message)
            qed_score (float): 0 = least drug-like, 1 = most drug-like
            error_message (str or None)
    """
    try:
        return _qed(mol), None
    except Exception as e:
        return None, f"QED computation failed: {str(e)}"


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
    if sascorer is None:
        return None, "SA Score unavailable (RDKit Contrib/SA_Score not found)"
    try:
        score = sascorer.calculateScore(mol)
        return score, None
    except Exception as e:
        return None, f"SA score computation failed: {str(e)}"


def clean_dataset(df, smiles_column):
    """
    Clean a bioactivity dataset by removing invalid rows and cross-filling
    pchembl_value / standard_value where possible.

    Parameters:
        df (pd.DataFrame): input DataFrame
        smiles_column (str): name of the column containing SMILES strings

    Returns:
        tuple: (cleaned_df, report)
            cleaned_df (pd.DataFrame): cleaned copy of df (original is not modified)
            report (dict): cleaning summary with keys:
                - rows_before (int)
                - missing_smiles_removed (int), missing_smiles_rows (list of int)
                - invalid_smiles_removed (int), invalid_smiles_rows (list of dicts)
                - pchembl_recomputed (int), pchembl_recomputed_rows (list of int)
                - stdval_computed (int), stdval_computed_rows (list of int)
                - inconsistent_count (int), inconsistent_rows (list of dicts)
                - both_missing_count (int), both_missing_rows (list of int)
                - rows_after (int)
    """
    import math
    import pandas as _pd

    df = df.copy()
    report = {}
    report["rows_before"] = len(df)

    # Step 1: Remove rows with missing/empty SMILES
    _missing_mask = df[smiles_column].isna() | (df[smiles_column].astype(str).str.strip() == "")
    _missing_indices = df[_missing_mask].index.tolist()
    report["missing_smiles_removed"] = len(_missing_indices)
    report["missing_smiles_rows"] = _missing_indices
    df = df[~_missing_mask].reset_index(drop=True)

    # Step 2: Remove rows where SMILES fails to parse
    _invalid = []
    _valid_mask = []
    for i, smi in enumerate(df[smiles_column]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            _invalid.append({"row": i, "smiles": str(smi)})
            _valid_mask.append(False)
        else:
            _valid_mask.append(True)
    report["invalid_smiles_removed"] = len(_invalid)
    report["invalid_smiles_rows"] = _invalid
    df = df[_valid_mask].reset_index(drop=True)

    # Step 3: Cross-fill pchembl_value <-> standard_value
    _has_pchembl = "pchembl_value" in df.columns
    _has_stdval = "standard_value" in df.columns

    report["stdval_computed"] = 0
    report["stdval_computed_rows"] = []
    report["inconsistent_count"] = 0
    report["inconsistent_rows"] = []
    report["both_missing_count"] = 0
    report["both_missing_rows"] = []
    report["pchembl_recomputed"] = 0
    report["pchembl_recomputed_rows"] = []

    if _has_pchembl and _has_stdval:
        # Pass 1: Back-compute standard_value where missing, and check
        # consistency of original pchembl vs standard_value BEFORE overwriting
        for i in range(len(df)):
            _pv = df.at[df.index[i], "pchembl_value"]
            _sv = df.at[df.index[i], "standard_value"]
            _pv_valid = _pv == _pv and _pv is not None  # not NaN
            _sv_valid = _sv == _sv and _sv is not None

            if _pv_valid and not _sv_valid:
                # pchembl present, standard_value missing -> back-compute
                try:
                    _pv_f = float(_pv)
                    _computed_sv = round(10 ** (9.0 - _pv_f), 3)
                    df.at[df.index[i], "standard_value"] = _computed_sv
                    report["stdval_computed"] += 1
                    report["stdval_computed_rows"].append(i)
                except (ValueError, TypeError):
                    pass
            elif _pv_valid and _sv_valid:
                # Both present -> check consistency before we overwrite
                try:
                    _sv_f = float(_sv)
                    _pv_f = float(_pv)
                    if _sv_f > 0:
                        _expected = 9.0 - math.log10(_sv_f)
                        if abs(_pv_f - _expected) > 0.1:
                            report["inconsistent_count"] += 1
                            report["inconsistent_rows"].append({
                                "row": i,
                                "original_pchembl": round(_pv_f, 3),
                                "standard_value": round(_sv_f, 3),
                                "recomputed_pchembl": round(_expected, 3),
                                "difference": round(abs(_pv_f - _expected), 3),
                            })
                except (ValueError, TypeError):
                    pass
            elif not _pv_valid and not _sv_valid:
                report["both_missing_count"] += 1
                report["both_missing_rows"].append(i)

        # Pass 2: Recompute ALL pchembl_value from standard_value
        for i in range(len(df)):
            _sv = df.at[df.index[i], "standard_value"]
            _sv_valid = _sv == _sv and _sv is not None
            if _sv_valid:
                try:
                    _sv_f = float(_sv)
                    if _sv_f > 0:
                        df.at[df.index[i], "pchembl_value"] = round(
                            9.0 - math.log10(_sv_f), 3
                        )
                        report["pchembl_recomputed"] += 1
                        report["pchembl_recomputed_rows"].append(i)
                except (ValueError, TypeError):
                    pass
    elif not _has_pchembl and _has_stdval:
        # Create pchembl_value column entirely from standard_value
        _pvals = []
        for i, _sv in enumerate(df["standard_value"]):
            try:
                _sv_f = float(_sv)
                if _sv_f > 0:
                    _pvals.append(round(9.0 - math.log10(_sv_f), 3))
                    report["pchembl_recomputed"] += 1
                    report["pchembl_recomputed_rows"].append(i)
                else:
                    _pvals.append(None)
            except (ValueError, TypeError):
                _pvals.append(None)
        df["pchembl_value"] = _pvals

    report["rows_after"] = len(df)
    return df, report


def label_activity(
    df,
    activity_column,
    activity_type="IC50",
    threshold_active=1000,
    threshold_inactive=10000,
    use_intermediate=False,
    is_log_scale=False,
):
    """
    Convert continuous bioactivity values into categorical labels for ML classification.

    Parameters:
        df (pd.DataFrame): input DataFrame with an activity column
        activity_column (str): column name containing activity values
        activity_type (str): label for the measurement type (e.g. "IC50", "Ki")
        threshold_active (float): threshold for "Active" label
        threshold_inactive (float): threshold for three-class mode "Inactive" boundary
        use_intermediate (bool): when True, adds an "Intermediate" class between the two thresholds
        is_log_scale (bool): when True, values are already on -log₁₀ scale (e.g. pIC50, pKi).
            Higher = more potent. Active = value >= threshold_active.
            When False (default), values are in nM. Lower = more potent. Active = value <= threshold_active.

    Returns:
        tuple: (labeled_df, skipped)
            labeled_df (pd.DataFrame): copy of df with new columns "Activity_Label" and "pActivity"
            skipped (list of dicts): rows skipped due to missing or invalid values
    """
    import math

    df = df.copy()
    labels = []
    p_values = []
    skipped = []

    for idx, val in df[activity_column].items():
        # Missing values
        if val != val or val is None:  # NaN check without importing pandas
            labels.append(None)
            p_values.append(None)
            skipped.append({"index": idx, "value": val, "reason": "missing value"})
            continue
        try:
            fval = float(val)
        except (ValueError, TypeError):
            labels.append(None)
            p_values.append(None)
            skipped.append({"index": idx, "value": val, "reason": "non-numeric value"})
            continue

        if is_log_scale:
            # Values already on -log₁₀ scale: higher = more potent
            p_values.append(round(fval, 3))

            if fval >= threshold_active:
                labels.append("Active")
            elif use_intermediate and fval >= threshold_inactive:
                labels.append("Intermediate")
            else:
                labels.append("Inactive")
        else:
            # Values in nM: lower = more potent
            if fval <= 0:
                labels.append(None)
                p_values.append(None)
                skipped.append({"index": idx, "value": fval, "reason": "non-positive value (cannot compute log)"})
                continue

            # pActivity = -log10(value in M) = -log10(value_nM × 1e-9) = 9 - log10(value_nM)
            p_val = round(9.0 - math.log10(fval), 3)
            p_values.append(p_val)

            if fval <= threshold_active:
                labels.append("Active")
            elif use_intermediate and fval <= threshold_inactive:
                labels.append("Intermediate")
            else:
                labels.append("Inactive")

    df["Activity_Label"] = labels
    df["pActivity"] = p_values
    return df, skipped


def labels_to_targets(labels, use_3class=False):
    """
    Convert string activity labels to numeric ML targets.

    Parameters:
        labels: list of str or None ("Active", "Intermediate", "Inactive")
        use_3class: if True, use ordinal mapping (Inactive=0, Intermediate=1, Active=2)
                    if False, use binary mapping (Active=1, Inactive=0)

    Returns:
        list of int or None (None for unmapped/missing labels)
    """
    if use_3class:
        mapping = {"Inactive": 0, "Intermediate": 1, "Active": 2}
    else:
        mapping = {"Active": 1, "Inactive": 0}
    return [mapping.get(label) for label in labels]


def analyze_scaffolds(mol_list):
    """
    Extract Murcko scaffolds and compute frequency statistics.

    Parameters:
        mol_list: list of RDKit Mol objects

    Returns:
        dict with keys: scaffold_smiles, scaffold_counts, top_scaffolds,
              singleton_count, unique_scaffold_count
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    scaffold_to_indices = {}

    for i, mol in enumerate(mol_list):
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(core)
            if not smi:
                smi = "No scaffold"
        except Exception:
            smi = "No scaffold"
        scaffolds.append(smi)
        scaffold_to_indices.setdefault(smi, []).append(i)

    scaffold_counts = dict(
        sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
    )
    counts_dict = {s: len(idxs) for s, idxs in scaffold_counts.items()}

    top_items = list(scaffold_counts.items())[:10]
    top_scaffolds = [(smi, len(idxs), idxs) for smi, idxs in top_items]

    singleton_count = sum(1 for idxs in scaffold_counts.values() if len(idxs) == 1)

    return {
        "scaffold_smiles": scaffolds,
        "scaffold_counts": counts_dict,
        "top_scaffolds": top_scaffolds,
        "singleton_count": singleton_count,
        "unique_scaffold_count": len(scaffold_counts),
    }


def run_preprocessing_pipeline(
    smiles_list,
    lipinski_max_violations=1,
    enable_veber=False,
    enable_ghose=False,
    enable_egan=False,
    enable_muegge=False,
    enable_brenk=False,
    enable_sa_score=False,
    enable_qed=False,
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
        enable_qed (bool): compute QED score for kept molecules (informational, not a filter)

    Returns:
        dict: {
            "kept_mols": list of surviving RDKit Mol objects,
            "kept_smiles": list of their canonical SMILES,
            "audit_trail": list of dicts, one per step, with counts,
            "removed_log": list of dicts, one per removed molecule, with reason,
            "sa_scores": list of SA scores per kept molecule (only if enable_sa_score=True), else None
            "qed_scores": list of QED scores per kept molecule (only if enable_qed=True), else None
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

    qed_scores = None
    if enable_qed:
        qed_scores = []
        for mol in unique_mols:
            score, error = compute_qed(mol)
            qed_scores.append(score)

    return {
        "kept_mols": unique_mols,
        "kept_smiles": kept_smiles,
        "audit_trail": audit_trail,
        "removed_log": removed_log,
        "sa_scores": sa_scores,
        "qed_scores": qed_scores,
    }
