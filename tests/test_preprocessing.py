"""
Tests for pipeline/preprocessing.py.
All descriptor values and filter pass/fail outcomes were verified empirically
before writing assertions.
"""
import pytest
from rdkit import Chem

from pipeline.preprocessing import (
    parse_smiles,
    standardize_molecule,
    strip_salts,
    check_lipinski,
    check_pains,
    check_brenk,
    check_veber,
    check_ghose,
    check_egan,
    check_muegge,
    compute_qed,
    compute_synthetic_accessibility,
    deduplicate_molecules,
    run_preprocessing_pipeline,
    labels_to_targets,
)

# ---------------------------------------------------------------------------
# Molecule SMILES (properties verified with RDKit before writing tests)
# ---------------------------------------------------------------------------
ASPIRIN       = "CC(=O)Oc1ccccc1C(=O)O"         # MW=180, LogP=1.31, HBD=1, HBA=3, 0 Lipinski violations; PAINS clean; Brenk hit (phenol_ester)
IBUPROFEN     = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"    # MW=206, Brenk clean, PAINS clean, Muegge pass
CIPROFLOXACIN = "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"  # MW=331, passes Ghose, Muegge, Veber, Egan; Brenk/PAINS clean
SORBITOL      = "OCC(O)C(O)C(O)C(O)CO"          # MW=182, HBD=6 (>5) → 1 Lipinski violation; Egan fail (LogP=-3.59<-1)
CATECHOL      = "Oc1ccccc1O"                     # PAINS hit (catechol_A), Brenk hit
CAPTOPRIL     = "CC(CS)C(=O)N1CCCC1C(=O)O"      # Brenk hit (thiol_2)
ETHANOL       = "CCO"                            # MW=46; fails Ghose, Muegge; Brenk/PAINS clean
TETRADECANE   = "CCCCCCCCCCCCCC"                 # 11 rotatable bonds → fails Veber


def _mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"Test setup error: could not parse {smiles!r}"
    return m


# ---------------------------------------------------------------------------
# parse_smiles
# ---------------------------------------------------------------------------
class TestParseSmiles:
    def test_valid_smiles_returns_mol_and_no_error(self):
        mol, err = parse_smiles(ASPIRIN)
        assert mol is not None
        assert err is None
        assert mol.GetNumAtoms() > 0

    def test_invalid_smiles_returns_none_and_error_message(self):
        mol, err = parse_smiles("not_a_smiles")
        assert mol is None
        assert isinstance(err, str)
        assert len(err) > 0

    def test_empty_string_returns_zero_atom_mol(self):
        # RDKit parses "" as a valid empty molecule (0 atoms) rather than None.
        # The pipeline handles this downstream (strip_salts removes 0-atom molecules).
        mol, err = parse_smiles("")
        assert err is None
        assert mol.GetNumAtoms() == 0


# ---------------------------------------------------------------------------
# standardize_molecule
# ---------------------------------------------------------------------------
class TestStandardizeMolecule:
    def test_valid_mol_returns_mol_and_no_error(self):
        result, err = standardize_molecule(_mol(ASPIRIN))
        assert result is not None
        assert err is None

    def test_standardized_mol_is_parseable(self):
        result, err = standardize_molecule(_mol(IBUPROFEN))
        assert result is not None
        smiles = Chem.MolToSmiles(result)
        assert Chem.MolFromSmiles(smiles) is not None


# ---------------------------------------------------------------------------
# strip_salts
# ---------------------------------------------------------------------------
class TestStripSalts:
    def test_pure_molecule_atom_count_unchanged(self):
        m = _mol(IBUPROFEN)
        result, err = strip_salts(m)
        assert err is None
        assert result is not None
        assert result.GetNumAtoms() == m.GetNumAtoms()

    def test_counterion_removed(self):
        # Na+ salt of aspirin: Na is the small counterion, aspirin is kept
        salt_smi = "[Na+].CC(=O)Oc1ccccc1C(=O)O"
        m = _mol(salt_smi)
        result, err = strip_salts(m)
        assert err is None
        assert result is not None
        # Na+ (1 atom) should be gone; result has fewer atoms than input
        assert result.GetNumAtoms() < m.GetNumAtoms()


# ---------------------------------------------------------------------------
# check_lipinski
# ---------------------------------------------------------------------------
class TestCheckLipinski:
    def test_aspirin_zero_violations(self):
        passes, desc, reason = check_lipinski(_mol(ASPIRIN), max_violations=1)
        assert passes is True
        assert desc["violations"] == 0
        assert reason is None

    def test_aspirin_passes_strict_zero_threshold(self):
        passes, _, _ = check_lipinski(_mol(ASPIRIN), max_violations=0)
        assert passes is True  # 0 violations ≤ 0

    def test_sorbitol_has_one_violation(self):
        # HBD=6 > 5 → 1 violation
        _, desc, _ = check_lipinski(_mol(SORBITOL))
        assert desc["violations"] == 1
        assert desc["HBD"] == 6

    def test_sorbitol_passes_with_threshold_one(self):
        passes, _, _ = check_lipinski(_mol(SORBITOL), max_violations=1)
        assert passes is True

    def test_sorbitol_fails_strict_threshold(self):
        passes, _, reason = check_lipinski(_mol(SORBITOL), max_violations=0)
        assert passes is False
        assert reason is not None

    def test_descriptor_dict_has_expected_keys(self):
        _, desc, _ = check_lipinski(_mol(ASPIRIN))
        assert set(desc.keys()) == {"MW", "LogP", "HBD", "HBA", "violations"}

    def test_aspirin_descriptor_values(self):
        _, desc, _ = check_lipinski(_mol(ASPIRIN))
        assert 179 < desc["MW"] < 182
        assert desc["HBD"] == 1
        assert desc["HBA"] == 3


# ---------------------------------------------------------------------------
# check_pains
# ---------------------------------------------------------------------------
class TestCheckPains:
    def test_catechol_is_pains_hit(self):
        is_pains, name = check_pains(_mol(CATECHOL))
        assert is_pains is True
        assert isinstance(name, str)

    def test_ibuprofen_is_pains_clean(self):
        is_pains, name = check_pains(_mol(IBUPROFEN))
        assert is_pains is False
        assert name is None

    def test_ciprofloxacin_is_pains_clean(self):
        is_pains, _ = check_pains(_mol(CIPROFLOXACIN))
        assert is_pains is False

    def test_return_type_is_bool_and_str_or_none(self):
        is_pains, name = check_pains(_mol(ASPIRIN))
        assert isinstance(is_pains, bool)
        assert name is None or isinstance(name, str)


# ---------------------------------------------------------------------------
# check_brenk
# ---------------------------------------------------------------------------
class TestCheckBrenk:
    def test_captopril_is_brenk_hit(self):
        # Captopril contains a free thiol (thiol_2 pattern)
        is_brenk, name = check_brenk(_mol(CAPTOPRIL))
        assert is_brenk is True
        assert "thiol" in name.lower()

    def test_ibuprofen_is_brenk_clean(self):
        is_brenk, name = check_brenk(_mol(IBUPROFEN))
        assert is_brenk is False
        assert name is None

    def test_ciprofloxacin_is_brenk_clean(self):
        is_brenk, _ = check_brenk(_mol(CIPROFLOXACIN))
        assert is_brenk is False

    def test_return_type_is_bool_and_str_or_none(self):
        is_brenk, name = check_brenk(_mol(IBUPROFEN))
        assert isinstance(is_brenk, bool)
        assert name is None or isinstance(name, str)


# ---------------------------------------------------------------------------
# check_veber
# ---------------------------------------------------------------------------
class TestCheckVeber:
    def test_aspirin_passes(self):
        # RotatableBonds=2, TPSA=63.6 — both within limits
        passes, desc, reason = check_veber(_mol(ASPIRIN))
        assert passes is True
        assert reason is None
        assert desc["RotatableBonds"] == 2
        assert desc["TPSA"] < 140

    def test_tetradecane_fails_rotatable_bonds(self):
        # 11 rotatable bonds > 10 → fails Veber
        passes, desc, reason = check_veber(_mol(TETRADECANE))
        assert passes is False
        assert desc["RotatableBonds"] == 11
        assert reason is not None

    def test_descriptor_keys(self):
        _, desc, _ = check_veber(_mol(ASPIRIN))
        assert set(desc.keys()) == {"RotatableBonds", "TPSA"}


# ---------------------------------------------------------------------------
# check_ghose
# ---------------------------------------------------------------------------
class TestCheckGhose:
    def test_ciprofloxacin_passes(self):
        # MW=331, LogP=1.58, MR=88.5, atoms=24 — all within Ghose range
        passes, desc, reason = check_ghose(_mol(CIPROFLOXACIN))
        assert passes is True
        assert reason is None

    def test_ethanol_fails_mw_and_atom_count(self):
        # MW=46 (<160), atoms=3 (<20) — fails on both
        passes, _, reason = check_ghose(_mol(ETHANOL))
        assert passes is False
        assert reason is not None

    def test_descriptor_keys(self):
        _, desc, _ = check_ghose(_mol(CIPROFLOXACIN))
        assert set(desc.keys()) == {"MW", "LogP", "MolarRefractivity", "AtomCount"}


# ---------------------------------------------------------------------------
# check_egan
# ---------------------------------------------------------------------------
class TestCheckEgan:
    def test_aspirin_passes(self):
        # LogP=1.31, TPSA=63.6 — both within Egan bounds
        passes, desc, reason = check_egan(_mol(ASPIRIN))
        assert passes is True
        assert reason is None

    def test_ciprofloxacin_passes(self):
        passes, _, _ = check_egan(_mol(CIPROFLOXACIN))
        assert passes is True

    def test_sorbitol_fails_logp(self):
        # LogP=-3.59 < -1 → fails Egan
        passes, desc, reason = check_egan(_mol(SORBITOL))
        assert passes is False
        assert desc["LogP"] < -1
        assert reason is not None

    def test_descriptor_keys(self):
        _, desc, _ = check_egan(_mol(ASPIRIN))
        assert set(desc.keys()) == {"LogP", "TPSA"}


# ---------------------------------------------------------------------------
# check_muegge
# ---------------------------------------------------------------------------
class TestCheckMuegge:
    def test_ciprofloxacin_passes(self):
        # MW=331, LogP=1.58, TPSA=74.6, rings=2, HBD=2, HBA=4, RotB=3 — all in range
        passes, desc, reason = check_muegge(_mol(CIPROFLOXACIN))
        assert passes is True
        assert reason is None

    def test_ibuprofen_passes(self):
        passes, _, _ = check_muegge(_mol(IBUPROFEN))
        assert passes is True

    def test_ethanol_fails_mw(self):
        # MW=46 < 200 → fails Muegge
        passes, desc, reason = check_muegge(_mol(ETHANOL))
        assert passes is False
        assert desc["MW"] < 200
        assert reason is not None

    def test_descriptor_keys(self):
        _, desc, _ = check_muegge(_mol(CIPROFLOXACIN))
        assert {"MW", "LogP", "TPSA", "Rings", "HBD", "HBA", "RotatableBonds"}.issubset(desc.keys())


# ---------------------------------------------------------------------------
# compute_qed
# ---------------------------------------------------------------------------
class TestComputeQED:
    def test_returns_float_in_0_to_1_range(self):
        score, err = compute_qed(_mol(ASPIRIN))
        assert err is None
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_ibuprofen_is_highly_drug_like(self):
        score, _ = compute_qed(_mol(IBUPROFEN))
        assert score > 0.7  # empirically ~0.82

    def test_aspirin_score_reasonable(self):
        score, _ = compute_qed(_mol(ASPIRIN))
        assert 0.4 <= score <= 0.7  # empirically ~0.55

    def test_all_test_molecules_valid(self):
        for smi in [ASPIRIN, IBUPROFEN, CIPROFLOXACIN, CATECHOL, ETHANOL]:
            score, err = compute_qed(_mol(smi))
            assert err is None
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_synthetic_accessibility
# ---------------------------------------------------------------------------
class TestComputeSA:
    def test_returns_float_in_1_to_10_range(self):
        score, err = compute_synthetic_accessibility(_mol(ASPIRIN))
        assert err is None
        assert isinstance(score, float)
        assert 1.0 <= score <= 10.0

    def test_ethanol_is_easily_synthesizable(self):
        # Simple molecule should have low SA score
        score, _ = compute_synthetic_accessibility(_mol(ETHANOL))
        assert score < 3.0

    def test_all_test_molecules_valid(self):
        for smi in [ASPIRIN, IBUPROFEN, CIPROFLOXACIN]:
            score, err = compute_synthetic_accessibility(_mol(smi))
            assert err is None
            assert 1.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# deduplicate_molecules
# ---------------------------------------------------------------------------
class TestDeduplicateMolecules:
    def test_no_duplicates_unchanged(self):
        mols = [_mol(ASPIRIN), _mol(IBUPROFEN), _mol(ETHANOL)]
        unique, dup_info = deduplicate_molecules(mols)
        assert len(unique) == 3
        assert len(dup_info) == 0

    def test_single_duplicate_removed(self):
        mols = [_mol(ASPIRIN), _mol(ETHANOL), _mol(ASPIRIN)]
        unique, dup_info = deduplicate_molecules(mols)
        assert len(unique) == 2
        assert len(dup_info) == 1

    def test_duplicate_info_correct_indices(self):
        mols = [_mol(ASPIRIN), _mol(ETHANOL), _mol(ASPIRIN)]
        _, dup_info = deduplicate_molecules(mols)
        assert dup_info[0]["index"] == 2
        assert dup_info[0]["duplicate_of_index"] == 0

    def test_all_same_returns_one(self):
        mols = [_mol(ASPIRIN)] * 5
        unique, dup_info = deduplicate_molecules(mols)
        assert len(unique) == 1
        assert len(dup_info) == 4

    def test_empty_list(self):
        unique, dup_info = deduplicate_molecules([])
        assert unique == []
        assert dup_info == []

    def test_first_occurrence_kept(self):
        mols = [_mol(ASPIRIN), _mol(ASPIRIN)]
        unique, dup_info = deduplicate_molecules(mols)
        assert len(unique) == 1
        # The duplicate was at index 1, the first at index 0
        assert dup_info[0]["duplicate_of_index"] == 0


# ---------------------------------------------------------------------------
# run_preprocessing_pipeline (end-to-end)
# ---------------------------------------------------------------------------
class TestRunPreprocessingPipeline:
    def test_result_has_all_expected_keys(self):
        result = run_preprocessing_pipeline([ASPIRIN])
        for key in ("kept_mols", "kept_smiles", "audit_trail", "removed_log", "sa_scores", "qed_scores"):
            assert key in result

    def test_valid_molecule_is_kept(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        assert len(result["kept_smiles"]) == 1

    def test_invalid_smiles_removed_at_parse_step(self):
        result = run_preprocessing_pipeline(["not_valid_smiles", IBUPROFEN])
        assert len(result["kept_smiles"]) == 1
        parse_removals = [e for e in result["removed_log"] if e["step"] == "parse_smiles"]
        assert len(parse_removals) == 1

    def test_pains_hit_removed(self):
        result = run_preprocessing_pipeline([IBUPROFEN, CATECHOL])
        assert len(result["kept_smiles"]) == 1
        pains_removals = [e for e in result["removed_log"] if e["step"] == "check_pains"]
        assert len(pains_removals) == 1

    def test_brenk_filter_optional_and_effective(self):
        # Captopril is a Brenk hit; without filter it is kept, with filter it is removed
        without_brenk = run_preprocessing_pipeline([CAPTOPRIL])
        with_brenk    = run_preprocessing_pipeline([CAPTOPRIL], enable_brenk=True)
        assert len(without_brenk["kept_smiles"]) == 1
        assert len(with_brenk["kept_smiles"])    == 0

    def test_duplicate_removed_at_dedup_step(self):
        result = run_preprocessing_pipeline([IBUPROFEN, IBUPROFEN])
        assert len(result["kept_smiles"]) == 1
        dedup_removals = [e for e in result["removed_log"] if e["step"] == "deduplicate_molecules"]
        assert len(dedup_removals) == 1

    def test_audit_trail_entry_structure(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        for entry in result["audit_trail"]:
            assert "step" in entry
            assert "input_count" in entry
            assert "output_count" in entry
            assert entry["output_count"] <= entry["input_count"]

    def test_audit_trail_contains_required_steps(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        steps = {e["step"] for e in result["audit_trail"]}
        assert {"parse_smiles", "standardize_molecule", "strip_salts",
                "check_lipinski", "check_pains", "deduplicate_molecules"}.issubset(steps)

    def test_optional_steps_absent_when_disabled(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        steps = {e["step"] for e in result["audit_trail"]}
        for step in ("check_brenk", "check_veber", "check_ghose", "check_egan", "check_muegge"):
            assert step not in steps

    def test_optional_steps_present_when_enabled(self):
        result = run_preprocessing_pipeline(
            [IBUPROFEN],
            enable_brenk=True, enable_veber=True, enable_ghose=True,
            enable_egan=True, enable_muegge=True,
        )
        steps = {e["step"] for e in result["audit_trail"]}
        for step in ("check_brenk", "check_veber", "check_ghose", "check_egan", "check_muegge"):
            assert step in steps

    def test_sa_scores_none_by_default(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        assert result["sa_scores"] is None

    def test_sa_scores_computed_when_enabled(self):
        result = run_preprocessing_pipeline([IBUPROFEN, CIPROFLOXACIN], enable_sa_score=True)
        assert result["sa_scores"] is not None
        assert len(result["sa_scores"]) == 2
        assert all(1.0 <= s <= 10.0 for s in result["sa_scores"])

    def test_qed_scores_none_by_default(self):
        result = run_preprocessing_pipeline([IBUPROFEN])
        assert result["qed_scores"] is None

    def test_qed_scores_computed_when_enabled(self):
        result = run_preprocessing_pipeline([IBUPROFEN, CIPROFLOXACIN], enable_qed=True)
        assert result["qed_scores"] is not None
        assert len(result["qed_scores"]) == 2
        assert all(0.0 <= s <= 1.0 for s in result["qed_scores"])

    def test_mixed_input_end_to_end(self):
        smiles_list = [
            IBUPROFEN,       # kept
            "invalid_smiles", # removed: parse
            CATECHOL,         # removed: PAINS
            CIPROFLOXACIN,    # kept
            IBUPROFEN,        # removed: duplicate
        ]
        result = run_preprocessing_pipeline(smiles_list)
        assert len(result["kept_smiles"]) == 2
        assert len(result["removed_log"]) == 3
        steps_removed = [e["step"] for e in result["removed_log"]]
        assert "parse_smiles" in steps_removed
        assert "check_pains" in steps_removed
        assert "deduplicate_molecules" in steps_removed

    def test_lipinski_threshold_controls_removal(self):
        # Sorbitol: 1 Lipinski violation (HBD=6)
        strict  = run_preprocessing_pipeline([SORBITOL, IBUPROFEN], lipinski_max_violations=0)
        lenient = run_preprocessing_pipeline([SORBITOL, IBUPROFEN], lipinski_max_violations=1)
        assert len(strict["kept_smiles"])  == 1   # sorbitol removed
        assert len(lenient["kept_smiles"]) == 2   # both pass


# ---------------------------------------------------------------------------
# labels_to_targets
# ---------------------------------------------------------------------------
class TestLabelsToTargets:
    def test_2class_basic(self):
        assert labels_to_targets(["Active", "Inactive", "Active"]) == [1, 0, 1]

    def test_3class_basic(self):
        result = labels_to_targets(
            ["Active", "Intermediate", "Inactive"], use_3class=True
        )
        assert result == [2, 1, 0]

    def test_none_labels(self):
        assert labels_to_targets(["Active", None, "Inactive"]) == [1, None, 0]

    def test_empty_list(self):
        assert labels_to_targets([]) == []

    def test_all_same_class(self):
        assert labels_to_targets(["Active", "Active"]) == [1, 1]
        assert labels_to_targets(["Active", "Active"], use_3class=True) == [2, 2]

    def test_3class_with_only_2_present(self):
        result = labels_to_targets(["Active", "Inactive"], use_3class=True)
        assert result == [2, 0]

    def test_unknown_label(self):
        assert labels_to_targets(["Active", "Unknown"]) == [1, None]
