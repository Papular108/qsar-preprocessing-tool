"""
Tests for pipeline/clustering.py.
"""
import pytest
import numpy as np
from rdkit import Chem

from pipeline.clustering import (
    compute_distance_matrix,
    cluster_butina,
    cluster_hierarchical,
    compute_cluster_pca,
)

# ── Test molecules (diverse scaffolds for meaningful clustering) ──
ASPIRIN       = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN     = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE      = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
METFORMIN     = "CN(C)C(=N)NC(=N)N"
DIAZEPAM      = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"
PROPRANOLOL   = "CC(C)NCC(O)COc1cccc2ccccc12"
CAPTOPRIL     = "CC(CS)C(=O)N1CCCC1C(=O)O"
CIPROFLOXACIN = "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
OMEPRAZOLE    = "COc1ccc2[nH]c(CS(=O)c3ncc(C)c(OC)c3C)nc2c1"
SILDENAFIL    = "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"
ETHANOL       = "CCO"
ACETAMINOPHEN = "CC(=O)Nc1ccc(O)cc1"


def _mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"Failed to parse: {smiles}"
    return m


def _mols_and_smiles(smi_list):
    mols = [_mol(s) for s in smi_list]
    canon = [Chem.MolToSmiles(m) for m in mols]
    return mols, canon


# ---------------------------------------------------------------------------
# compute_distance_matrix
# ---------------------------------------------------------------------------
class TestComputeDistanceMatrix:
    def test_basic_shape(self):
        mols, _ = _mols_and_smiles([ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL])
        dm = compute_distance_matrix(mols)
        assert len(dm["condensed"]) == 10  # 5*(5-1)/2
        assert dm["square"].shape == (5, 5)

    def test_self_distance_zero(self):
        mols, _ = _mols_and_smiles([ASPIRIN, IBUPROFEN, CAFFEINE])
        dm = compute_distance_matrix(mols)
        for i in range(3):
            assert dm["square"][i][i] == 0.0

    def test_symmetric(self):
        mols, _ = _mols_and_smiles([ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM])
        dm = compute_distance_matrix(mols)
        sq = dm["square"]
        assert np.allclose(sq, sq.T)

    def test_distances_in_0_to_1(self):
        mols, _ = _mols_and_smiles([ASPIRIN, IBUPROFEN, CAFFEINE])
        dm = compute_distance_matrix(mols)
        assert all(0.0 <= d <= 1.0 for d in dm["condensed"])
        assert np.all(dm["square"] >= 0.0)
        assert np.all(dm["square"] <= 1.0)

    def test_identical_molecules_zero(self):
        mols, _ = _mols_and_smiles([ASPIRIN, ASPIRIN])
        dm = compute_distance_matrix(mols)
        assert dm["condensed"] == [0.0]
        assert dm["square"][0][1] == 0.0
        assert dm["square"][1][0] == 0.0


# ---------------------------------------------------------------------------
# cluster_butina
# ---------------------------------------------------------------------------
class TestClusterButina:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL, PROPRANOLOL, METFORMIN]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_butina(mols, canon, cutoff=0.4)
        assert result["n_clusters"] >= 1
        assert result["method"] == "butina"
        assert len(result["assignments"]) == 10

    def test_all_singletons_tight_cutoff(self):
        smi_list = [ASPIRIN, CAFFEINE, DIAZEPAM, CIPROFLOXACIN, SILDENAFIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_butina(mols, canon, cutoff=0.01)
        assert result["n_clusters"] == len(smi_list)
        assert result["n_singletons"] == len(smi_list)

    def test_one_cluster_loose_cutoff(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_butina(mols, canon, cutoff=0.99)
        assert result["n_clusters"] == 1

    def test_representative_is_in_cluster(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, PROPRANOLOL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_butina(mols, canon, cutoff=0.4)
        assign = result["assignments"]
        for _, row in result["representatives"].iterrows():
            cid = row["Cluster_ID"]
            rep_smi = row["SMILES"]
            cluster_members = assign[assign["Cluster_ID"] == cid]["SMILES"].tolist()
            assert rep_smi in cluster_members

    def test_assignments_cover_all_mols(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_butina(mols, canon, cutoff=0.4)
        assert len(result["assignments"]) == 5
        assert set(result["assignments"]["SMILES"]) == set(canon)

    def test_empty_input(self):
        result = cluster_butina([], [])
        assert result["n_clusters"] == 0
        assert len(result["assignments"]) == 0
        assert len(result["representatives"]) == 0

    def test_single_molecule(self):
        mols, canon = _mols_and_smiles([ASPIRIN])
        result = cluster_butina(mols, canon, cutoff=0.4)
        assert result["n_clusters"] == 1
        assert result["n_singletons"] == 1
        assert len(result["representatives"]) == 1

    def test_two_identical_molecules(self):
        mols, canon = _mols_and_smiles([ASPIRIN, ASPIRIN])
        result = cluster_butina(mols, canon, cutoff=0.4)
        # Distance is 0 which is < cutoff, so they should be in the same cluster
        assert result["n_clusters"] == 1

    def test_reproducible(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL]
        mols, canon = _mols_and_smiles(smi_list)
        r1 = cluster_butina(mols, canon, cutoff=0.4)
        r2 = cluster_butina(mols, canon, cutoff=0.4)
        assert r1["assignments"]["Cluster_ID"].tolist() == r2["assignments"]["Cluster_ID"].tolist()
        assert r1["n_clusters"] == r2["n_clusters"]


# ---------------------------------------------------------------------------
# cluster_hierarchical
# ---------------------------------------------------------------------------
class TestClusterHierarchical:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL, PROPRANOLOL, METFORMIN]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_hierarchical(mols, canon, n_clusters=3)
        assert result["n_clusters"] == 3
        assert result["method"] == "hierarchical"
        assert len(result["assignments"]) == 10

    def test_auto_n_clusters(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL, PROPRANOLOL, METFORMIN]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_hierarchical(mols, canon, n_clusters=None)
        assert result["n_clusters"] >= 2
        assert result["silhouette_score"] is not None
        assert -1.0 <= result["silhouette_score"] <= 1.0

    def test_assignments_cover_all_mols(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_hierarchical(mols, canon, n_clusters=2)
        assert len(result["assignments"]) == 5
        assert set(result["assignments"]["SMILES"]) == set(canon)

    def test_representative_is_in_cluster(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, PROPRANOLOL]
        mols, canon = _mols_and_smiles(smi_list)
        result = cluster_hierarchical(mols, canon, n_clusters=3)
        assign = result["assignments"]
        for _, row in result["representatives"].iterrows():
            cid = row["Cluster_ID"]
            rep_smi = row["SMILES"]
            cluster_members = assign[assign["Cluster_ID"] == cid]["SMILES"].tolist()
            assert rep_smi in cluster_members

    def test_single_molecule(self):
        mols, canon = _mols_and_smiles([ASPIRIN])
        result = cluster_hierarchical(mols, canon, n_clusters=1)
        assert result["n_clusters"] == 1
        assert result["silhouette_score"] is None

    def test_two_molecules(self):
        mols, canon = _mols_and_smiles([ASPIRIN, CAFFEINE])
        result = cluster_hierarchical(mols, canon, n_clusters=None)
        assert result["n_clusters"] == 1
        assert result["silhouette_score"] is None

    def test_all_identical(self):
        mols, canon = _mols_and_smiles([ASPIRIN, ASPIRIN, ASPIRIN, ASPIRIN, ASPIRIN])
        result = cluster_hierarchical(mols, canon, n_clusters=None)
        assert result["n_clusters"] == 1
        assert result["silhouette_score"] is None
        assert all(cid == 0 for cid in result["assignments"]["Cluster_ID"])


# ---------------------------------------------------------------------------
# compute_cluster_pca
# ---------------------------------------------------------------------------
class TestComputeClusterPCA:
    def test_basic(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL, PROPRANOLOL, METFORMIN]
        mols = [_mol(s) for s in smi_list]
        cluster_ids = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
        result = compute_cluster_pca(mols, cluster_ids)
        df = result["pca_df"]
        assert len(df) == 10
        assert set(df.columns) == {"PC1", "PC2", "Cluster_ID", "SMILES"}
        assert len(result["var_ratio"]) == 2

    def test_var_ratio_reasonable(self):
        smi_list = [ASPIRIN, IBUPROFEN, CAFFEINE, DIAZEPAM, CAPTOPRIL,
                    CIPROFLOXACIN, OMEPRAZOLE, SILDENAFIL, PROPRANOLOL, METFORMIN]
        mols = [_mol(s) for s in smi_list]
        cluster_ids = [0] * 5 + [1] * 5
        result = compute_cluster_pca(mols, cluster_ids)
        total_var = sum(result["var_ratio"])
        assert 0.0 < total_var <= 1.0
        assert result["var_ratio"][0] > 0.0

    def test_fewer_than_3(self):
        mols = [_mol(ASPIRIN), _mol(CAFFEINE)]
        result = compute_cluster_pca(mols, [0, 1])
        assert len(result["pca_df"]) == 2
        assert result["var_ratio"] == [0.0, 0.0]
