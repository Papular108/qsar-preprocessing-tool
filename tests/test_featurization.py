"""
Tests for pipeline/featurization.py.
"""
import pytest
import numpy as np
from rdkit import Chem

from pipeline.featurization import compute_descriptors, compute_fingerprint, featurize_dataset

ASPIRIN       = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN     = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CIPROFLOXACIN = "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
ETHANOL       = "CCO"

DESCRIPTOR_KEYS = {"MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "AromaticRings"}


def _mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return m


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------
class TestComputeDescriptors:
    def test_returns_dict_with_all_expected_keys(self):
        desc = compute_descriptors(_mol(ASPIRIN))
        assert set(desc.keys()) == DESCRIPTOR_KEYS

    def test_all_values_are_numeric(self):
        desc = compute_descriptors(_mol(IBUPROFEN))
        for val in desc.values():
            assert isinstance(val, (int, float))

    def test_aspirin_mw_in_range(self):
        desc = compute_descriptors(_mol(ASPIRIN))
        assert 179.0 < desc["MW"] < 182.0

    def test_aspirin_hbd_hba_correct(self):
        desc = compute_descriptors(_mol(ASPIRIN))
        assert desc["HBD"] == 1
        assert desc["HBA"] == 3

    def test_aspirin_aromatic_rings(self):
        desc = compute_descriptors(_mol(ASPIRIN))
        assert desc["AromaticRings"] == 1

    def test_ethanol_no_aromatic_rings(self):
        desc = compute_descriptors(_mol(ETHANOL))
        assert desc["AromaticRings"] == 0
        assert desc["HBD"] == 1

    def test_ciprofloxacin_multiple_rings(self):
        desc = compute_descriptors(_mol(CIPROFLOXACIN))
        assert desc["AromaticRings"] >= 1


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------
class TestComputeFingerprint:
    @pytest.mark.parametrize("fp_type,n_bits", [
        ("morgan",      2048),
        ("morgan",      1024),
        ("morgan",       512),
        ("topological", 2048),
        ("atom_pair",   1024),
        ("torsion",      512),
        ("avalon",      2048),
    ])
    def test_fingerprint_length_matches_n_bits(self, fp_type, n_bits):
        fp, err = compute_fingerprint(_mol(ASPIRIN), fp_type=fp_type, n_bits=n_bits)
        assert err is None
        assert isinstance(fp, np.ndarray)
        assert len(fp) == n_bits

    def test_maccs_always_167_bits(self):
        fp, err = compute_fingerprint(_mol(ASPIRIN), fp_type="maccs", n_bits=167)
        assert err is None
        assert len(fp) == 167

    def test_unknown_fp_type_returns_error(self):
        fp, err = compute_fingerprint(_mol(ASPIRIN), fp_type="unknown_type")
        assert fp is None
        assert err is not None
        assert "unknown" in err.lower() or "Unknown" in err

    def test_fingerprint_contains_only_0_and_1(self):
        fp, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", n_bits=2048)
        assert set(fp).issubset({0, 1})

    def test_different_molecules_produce_different_fingerprints(self):
        fp_a, _ = compute_fingerprint(_mol(ASPIRIN),   fp_type="morgan", n_bits=2048)
        fp_b, _ = compute_fingerprint(_mol(ETHANOL),   fp_type="morgan", n_bits=2048)
        assert not np.array_equal(fp_a, fp_b)

    def test_morgan_radius_2_and_3_differ(self):
        fp2, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", radius=2, n_bits=2048)
        fp3, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", radius=3, n_bits=2048)
        assert not np.array_equal(fp2, fp3)

    def test_same_molecule_same_fingerprint(self):
        fp1, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", n_bits=2048)
        fp2, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", n_bits=2048)
        assert np.array_equal(fp1, fp2)

    def test_fingerprint_not_all_zeros(self):
        # A real molecule should have some bits set
        fp, _ = compute_fingerprint(_mol(ASPIRIN), fp_type="morgan", n_bits=2048)
        assert fp.sum() > 0


# ---------------------------------------------------------------------------
# featurize_dataset
# ---------------------------------------------------------------------------
class TestFeaturizeDataset:
    def test_row_count_matches_mol_list(self):
        mols = [_mol(s) for s in [ASPIRIN, IBUPROFEN, ETHANOL]]
        df, errors = featurize_dataset(mols, fp_type="morgan", n_bits=512)
        assert len(errors) == 0
        assert df.shape[0] == 3

    def test_column_count_matches_1_smiles_plus_descriptors_plus_fp(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols, fp_type="morgan", n_bits=512)
        # 1 (SMILES) + 7 (descriptors) + 512 (fp bits) = 520
        assert df.shape[1] == 520

    def test_smiles_column_present(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols)
        assert "SMILES" in df.columns

    def test_all_descriptor_columns_present(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols)
        for col in DESCRIPTOR_KEYS:
            assert col in df.columns

    def test_fingerprint_columns_named_correctly(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols, fp_type="morgan", n_bits=64)
        fp_cols = [c for c in df.columns if c.startswith("morgan_bit_")]
        assert len(fp_cols) == 64
        assert "morgan_bit_0" in df.columns
        assert "morgan_bit_63" in df.columns

    @pytest.mark.parametrize("fp_type", ["morgan", "maccs", "topological", "atom_pair", "torsion", "avalon"])
    def test_all_fingerprint_types_produce_valid_output(self, fp_type):
        mols = [_mol(ASPIRIN), _mol(IBUPROFEN)]
        df, errors = featurize_dataset(mols, fp_type=fp_type)
        assert len(errors) == 0
        assert df.shape[0] == 2
        assert "SMILES" in df.columns

    def test_no_errors_for_valid_molecules(self):
        mols = [_mol(s) for s in [ASPIRIN, IBUPROFEN, CIPROFLOXACIN, ETHANOL]]
        _, errors = featurize_dataset(mols)
        assert len(errors) == 0

    def test_smiles_column_values_are_canonical(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols)
        smiles_val = df["SMILES"].iloc[0]
        # Value should be parseable back into a molecule
        assert Chem.MolFromSmiles(smiles_val) is not None

    def test_maccs_column_count(self):
        mols = [_mol(ASPIRIN)]
        df, _ = featurize_dataset(mols, fp_type="maccs", n_bits=167)
        fp_cols = [c for c in df.columns if c.startswith("maccs_bit_")]
        assert len(fp_cols) == 167
