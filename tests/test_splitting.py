"""
Tests for pipeline/splitting.py.
"""
import pytest
import numpy as np
from rdkit import Chem

from pipeline.splitting import (
    extract_murcko_scaffolds,
    scaffold_split,
    random_split,
    compute_split_pca,
)

# ── Test molecules ──
ASPIRIN       = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN     = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
NAPROXEN      = "CC(c1ccc2cccc(OC)c2c1)C(=O)O"
ACETAMINOPHEN = "CC(=O)Nc1ccc(O)cc1"
CAFFEINE      = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
METFORMIN     = "CN(C)C(=N)NC(=N)N"           # Acyclic — empty scaffold
DIAZEPAM      = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"
PROPRANOLOL   = "CC(C)NCC(O)COc1cccc2ccccc12"
CAPTOPRIL     = "CC(CS)C(=O)N1CCCC1C(=O)O"
ETHANOL       = "CCO"                          # Acyclic
# Molecules with guaranteed unique scaffolds (verified):
CIPROFLOXACIN = "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
OMEPRAZOLE    = "COc1ccc2[nH]c(CS(=O)c3ncc(C)c(OC)c3C)nc2c1"
FUROSEMIDE    = "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl"
WARFARIN      = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
SILDENAFIL    = "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"Failed to parse: {smiles}"
    return m


def _mols_and_smiles(smi_list):
    mols = [_mol(s) for s in smi_list]
    canon = [Chem.MolToSmiles(m) for m in mols]
    return mols, canon


# ---------------------------------------------------------------------------
# extract_murcko_scaffolds
# ---------------------------------------------------------------------------
class TestExtractMurckoScaffolds:
    def test_basic(self):
        mols = [_mol(ASPIRIN), _mol(IBUPROFEN), _mol(CAFFEINE)]
        scaffolds = extract_murcko_scaffolds(mols)
        assert len(scaffolds) == 3
        for s in scaffolds:
            assert isinstance(s, str)
            assert len(s) > 0  # All three have rings

    def test_acyclic_returns_empty_string(self):
        mols = [_mol(METFORMIN)]
        scaffolds = extract_murcko_scaffolds(mols)
        assert scaffolds == [""]

    def test_empty_input(self):
        assert extract_murcko_scaffolds([]) == []

    def test_handles_none(self):
        mols = [_mol(ASPIRIN), None, _mol(CAFFEINE)]
        scaffolds = extract_murcko_scaffolds(mols)
        assert len(scaffolds) == 3
        assert scaffolds[1] == ""
        assert scaffolds[0] != ""
        assert scaffolds[2] != ""


# ---------------------------------------------------------------------------
# scaffold_split
# ---------------------------------------------------------------------------
class TestScaffoldSplit:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN,
                    CAFFEINE, DIAZEPAM, PROPRANOLOL, CAPTOPRIL,
                    METFORMIN, ETHANOL]
        mols, canon = _mols_and_smiles(smi_list)
        result = scaffold_split(mols, canon, test_size=0.3, random_state=42)

        assert result["fallback"] is None
        # All indices accounted for
        all_idx = sorted(result["train_indices"] + result["test_indices"])
        assert all_idx == list(range(10))
        # No scaffold overlap between train and test
        train_scaffolds = set(result["train_df"]["Scaffold"])
        test_scaffolds = set(result["test_df"]["Scaffold"])
        assert train_scaffolds.isdisjoint(test_scaffolds)

    def test_single_molecule(self):
        mols, canon = _mols_and_smiles([ASPIRIN])
        result = scaffold_split(mols, canon, test_size=0.2)
        assert result["train_indices"] == [0]
        assert result["test_indices"] == []
        assert result["fallback"] == "single_scaffold"

    def test_all_same_scaffold(self):
        # Aspirin and its canonical form are the same scaffold
        # Use molecules that share the benzene scaffold
        smi_list = [
            "c1ccccc1",          # benzene
            "Cc1ccccc1",         # toluene — scaffold is benzene
            "CCc1ccccc1",        # ethylbenzene — scaffold is benzene
            "Oc1ccccc1",         # phenol — scaffold is benzene
            "Nc1ccccc1",         # aniline — scaffold is benzene
        ]
        mols, canon = _mols_and_smiles(smi_list)
        result = scaffold_split(mols, canon, test_size=0.2)
        assert result["fallback"] == "single_scaffold"
        assert len(result["test_indices"]) == 0
        assert len(result["train_indices"]) == 5

    def test_all_singletons(self):
        # 8 molecules each with a verified-unique Murcko scaffold
        smi_list = [CAFFEINE, DIAZEPAM, CAPTOPRIL, CIPROFLOXACIN,
                    OMEPRAZOLE, FUROSEMIDE, WARFARIN, SILDENAFIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = scaffold_split(mols, canon, test_size=0.25, random_state=42)
        assert result["fallback"] is None
        assert len(result["test_indices"]) > 0
        assert len(result["train_indices"]) > 0
        # All singletons → n_singletons == n_scaffolds
        assert result["n_singletons"] == result["n_scaffolds"]

    def test_reproducible(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN,
                    CAFFEINE, DIAZEPAM, PROPRANOLOL, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        r1 = scaffold_split(mols, canon, test_size=0.3, random_state=99)
        r2 = scaffold_split(mols, canon, test_size=0.3, random_state=99)
        assert r1["train_indices"] == r2["train_indices"]
        assert r1["test_indices"] == r2["test_indices"]

    def test_labels_included_when_provided(self):
        smi_list = [ASPIRIN, CAFFEINE, DIAZEPAM, PROPRANOLOL, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        labels = ["Active", "Inactive", "Active", "Inactive", "Active"]
        result = scaffold_split(mols, canon, test_size=0.3, labels=labels)
        assert "Label" in result["train_df"].columns
        if len(result["test_df"]) > 0:
            assert "Label" in result["test_df"].columns

    def test_scaffold_counts_dataframe(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, METFORMIN, ETHANOL]
        mols, canon = _mols_and_smiles(smi_list)
        result = scaffold_split(mols, canon, test_size=0.3)
        sc = result["scaffold_counts"]
        assert "Scaffold" in sc.columns
        assert "Count" in sc.columns
        assert sc["Count"].sum() == 5
        # Sorted descending
        assert list(sc["Count"]) == sorted(sc["Count"], reverse=True)

    def test_achieved_test_fraction_reported(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN,
                    CAFFEINE, DIAZEPAM, PROPRANOLOL, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = scaffold_split(mols, canon, test_size=0.25)
        assert 0.0 <= result["achieved_test_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------
class TestRandomSplit:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE,
                    DIAZEPAM, PROPRANOLOL, CAPTOPRIL, METFORMIN, ETHANOL]
        mols, canon = _mols_and_smiles(smi_list)
        result = random_split(mols, canon, test_size=0.2)
        assert len(result["test_indices"]) == 2
        assert len(result["train_indices"]) == 8
        assert result["fallback"] is None

    def test_deterministic(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE]
        mols, canon = _mols_and_smiles(smi_list)
        r1 = random_split(mols, canon, test_size=0.4, random_state=42)
        r2 = random_split(mols, canon, test_size=0.4, random_state=42)
        assert r1["train_indices"] == r2["train_indices"]
        assert r1["test_indices"] == r2["test_indices"]

    def test_stratified_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE,
                    DIAZEPAM, PROPRANOLOL, CAPTOPRIL, METFORMIN, ETHANOL]
        mols, canon = _mols_and_smiles(smi_list)
        labels = ["Active"] * 5 + ["Inactive"] * 5
        result = random_split(mols, canon, test_size=0.4, labels=labels, stratify=True)
        # Both classes should appear in both sets
        train_labels = result["train_df"]["Label"].tolist()
        test_labels = result["test_df"]["Label"].tolist()
        assert "Active" in train_labels and "Inactive" in train_labels
        assert "Active" in test_labels and "Inactive" in test_labels

    def test_stratified_no_labels_raises(self):
        mols, canon = _mols_and_smiles([ASPIRIN, IBUPROFEN])
        with pytest.raises(ValueError, match="labels"):
            random_split(mols, canon, stratify=True)

    def test_stratified_single_class_raises(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN]
        mols, canon = _mols_and_smiles(smi_list)
        labels = ["Active"] * 4
        with pytest.raises(ValueError, match="2 classes"):
            random_split(mols, canon, labels=labels, stratify=True)

    def test_labels_column_present(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE]
        mols, canon = _mols_and_smiles(smi_list)
        labels = ["A", "B", "A", "B", "A"]
        result = random_split(mols, canon, test_size=0.4, labels=labels)
        assert "Label" in result["train_df"].columns
        assert "Label" in result["test_df"].columns

    def test_stratified_with_none_labels_raises(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE]
        mols, canon = _mols_and_smiles(smi_list)
        labels = ["Active", None, "Inactive", "Active", None]
        with pytest.raises(ValueError, match="None"):
            random_split(mols, canon, labels=labels, stratify=True)

    def test_too_small(self):
        mols, canon = _mols_and_smiles([ASPIRIN])
        result = random_split(mols, canon, test_size=0.2)
        assert result["fallback"] == "too_small"
        assert result["train_indices"] == [0]
        assert result["test_indices"] == []


# ---------------------------------------------------------------------------
# compute_split_pca
# ---------------------------------------------------------------------------
class TestComputeSplitPCA:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE,
                    DIAZEPAM, PROPRANOLOL, CAPTOPRIL, METFORMIN, ETHANOL]
        mols = [_mol(s) for s in smi_list]
        train_idx = list(range(5))
        test_idx = list(range(5, 10))
        result = compute_split_pca(mols, train_idx, test_idx)
        df = result["pca_df"]
        assert len(df) == 10
        assert set(df.columns) == {"PC1", "PC2", "Split", "SMILES"}
        assert set(df["Split"].unique()) == {"Train", "Test"}
        assert len(result["var_ratio"]) == 2

    def test_variance_sums_reasonable(self):
        smi_list = [ASPIRIN, IBUPROFEN, NAPROXEN, ACETAMINOPHEN, CAFFEINE,
                    DIAZEPAM, PROPRANOLOL, CAPTOPRIL, METFORMIN, ETHANOL]
        mols = [_mol(s) for s in smi_list]
        result = compute_split_pca(mols, list(range(7)), list(range(7, 10)))
        total_var = sum(result["var_ratio"])
        # Two PCs should explain between 0% and 100% of variance
        assert 0.0 < total_var <= 1.0
        # For 10 diverse molecules with 1024-bit FPs, 2 PCs won't explain everything
        # but should capture a meaningful fraction
        assert result["var_ratio"][0] > 0.0
        assert result["var_ratio"][1] >= 0.0

    def test_fewer_than_3_points(self):
        mols = [_mol(ASPIRIN), _mol(IBUPROFEN)]
        result = compute_split_pca(mols, [0], [1])
        assert len(result["pca_df"]) == 2
        assert result["var_ratio"] == [0.0, 0.0]
