"""
Molecular clustering using fingerprint-based Tanimoto distance.

Provides Butina (Taylor-Butina) and agglomerative hierarchical clustering,
plus PCA projection for visualizing cluster assignments.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline.featurization import _mol_to_fp_obj


def compute_distance_matrix(mols, fp_type="morgan", radius=2, n_bits=2048):
    """
    Compute pairwise Tanimoto distance matrix.

    Parameters:
        mols: list of RDKit Mol objects
        fp_type, radius, n_bits: fingerprint settings

    Returns:
        dict:
            "condensed": list[float] — lower-triangle distances for Butina
            "square": np.ndarray shape (n, n) — full matrix for sklearn
    """
    n = len(mols)
    if n == 0:
        return {"condensed": [], "square": np.zeros((0, 0))}

    fps = [_mol_to_fp_obj(mol, fp_type, radius, n_bits) for mol in mols]

    square = np.zeros((n, n))
    condensed = []

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        for j, sim in enumerate(sims):
            dist = 1.0 - sim
            condensed.append(dist)
            square[i][j] = dist
            square[j][i] = dist

    return {"condensed": condensed, "square": square}


def cluster_butina(mols, smiles, fp_type="morgan", radius=2, n_bits=2048,
                   cutoff=0.4):
    """
    Butina (Taylor-Butina) clustering using RDKit.

    Parameters:
        mols: list of RDKit Mol objects
        smiles: list of SMILES strings (parallel to mols)
        fp_type, radius, n_bits: fingerprint settings
        cutoff: Tanimoto distance cutoff (0.4 means similarity >= 0.6 to cluster)

    Returns:
        dict:
            "assignments": DataFrame [SMILES, Cluster_ID, Is_Representative]
            "cluster_sizes": DataFrame [Cluster_ID, Size] sorted descending
            "representatives": DataFrame [Cluster_ID, Size, SMILES]
            "n_clusters": int
            "n_singletons": int
            "method": "butina"
            "params": dict of settings used
    """
    n = len(mols)
    if n == 0:
        empty_assign = pd.DataFrame(columns=["SMILES", "Cluster_ID", "Is_Representative"])
        empty_sizes = pd.DataFrame(columns=["Cluster_ID", "Size"])
        empty_reps = pd.DataFrame(columns=["Cluster_ID", "Size", "SMILES"])
        return {
            "assignments": empty_assign, "cluster_sizes": empty_sizes,
            "representatives": empty_reps, "n_clusters": 0, "n_singletons": 0,
            "method": "butina",
            "params": {"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                       "cutoff": cutoff},
        }

    dm = compute_distance_matrix(mols, fp_type, radius, n_bits)
    clusters = Butina.ClusterData(dm["condensed"], n, cutoff,
                                  isDistData=True, reordering=True)

    # Build assignments — clusters are already sorted largest-first by RDKit
    cluster_ids = [0] * n
    is_rep = [False] * n
    sizes_list = []
    reps_list = []

    for cid, cluster in enumerate(clusters):
        centroid = cluster[0]
        for idx in cluster:
            cluster_ids[idx] = cid
        is_rep[centroid] = True
        size = len(cluster)
        sizes_list.append({"Cluster_ID": cid, "Size": size})
        reps_list.append({"Cluster_ID": cid, "Size": size, "SMILES": smiles[centroid]})

    assignments = pd.DataFrame({
        "SMILES": smiles,
        "Cluster_ID": cluster_ids,
        "Is_Representative": is_rep,
    })
    cluster_sizes = pd.DataFrame(sizes_list)
    representatives = pd.DataFrame(reps_list)
    n_singletons = sum(1 for s in sizes_list if s["Size"] == 1)

    return {
        "assignments": assignments,
        "cluster_sizes": cluster_sizes,
        "representatives": representatives,
        "n_clusters": len(clusters),
        "n_singletons": n_singletons,
        "method": "butina",
        "params": {"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                   "cutoff": cutoff},
    }


def cluster_hierarchical(mols, smiles, fp_type="morgan", radius=2, n_bits=2048,
                         n_clusters=None):
    """
    Agglomerative clustering on Tanimoto distance matrix.

    Parameters:
        mols: list of RDKit Mol objects
        smiles: list of SMILES strings (parallel to mols)
        fp_type, radius, n_bits: fingerprint settings
        n_clusters: int or None. If None, auto-select via silhouette score.

    Returns:
        dict:
            "assignments": DataFrame [SMILES, Cluster_ID, Is_Representative]
            "cluster_sizes": DataFrame [Cluster_ID, Size] sorted descending
            "representatives": DataFrame [Cluster_ID, Size, SMILES]
            "n_clusters": int
            "n_singletons": int
            "silhouette_score": float or None
            "method": "hierarchical"
            "params": dict of settings used
    """
    n = len(mols)
    if n == 0:
        empty_assign = pd.DataFrame(columns=["SMILES", "Cluster_ID", "Is_Representative"])
        empty_sizes = pd.DataFrame(columns=["Cluster_ID", "Size"])
        empty_reps = pd.DataFrame(columns=["Cluster_ID", "Size", "SMILES"])
        return {
            "assignments": empty_assign, "cluster_sizes": empty_sizes,
            "representatives": empty_reps, "n_clusters": 0, "n_singletons": 0,
            "silhouette_score": None, "method": "hierarchical",
            "params": {"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                       "n_clusters": n_clusters},
        }

    dm = compute_distance_matrix(mols, fp_type, radius, n_bits)
    sq = dm["square"]

    # Too few molecules for meaningful clustering
    if n < 3:
        labels = [0] * n
        return _build_hierarchical_result(
            labels, smiles, sq, n_clusters=1, sil_score=None,
            params={"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                    "n_clusters": n_clusters},
        )

    # All distances zero (identical molecules) — skip auto-selection
    if np.allclose(sq, 0.0):
        labels = [0] * n
        return _build_hierarchical_result(
            labels, smiles, sq, n_clusters=1, sil_score=None,
            params={"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                    "n_clusters": n_clusters},
        )

    if n_clusters is None:
        n_clusters_final = _auto_select_n_clusters(sq, max_k=min(20, n - 1))
    else:
        n_clusters_final = min(n_clusters, n)

    model = AgglomerativeClustering(
        n_clusters=n_clusters_final, metric="precomputed", linkage="average",
    )
    labels = model.fit_predict(sq).tolist()

    # Compute silhouette
    sil_score = None
    unique_labels = set(labels)
    if len(unique_labels) >= 2 and len(unique_labels) < n:
        sil_score = round(float(silhouette_score(sq, labels, metric="precomputed")), 4)

    # Renumber clusters so cluster 0 is largest
    labels = _renumber_by_size(labels)

    return _build_hierarchical_result(
        labels, smiles, sq, n_clusters=len(set(labels)), sil_score=sil_score,
        params={"fp_type": fp_type, "radius": radius, "n_bits": n_bits,
                "n_clusters": n_clusters},
    )


def compute_cluster_pca(mols, cluster_ids, fp_type="morgan", radius=2,
                        n_bits=1024):
    """
    PCA projection colored by cluster assignment.

    Parameters:
        mols: list of RDKit Mol objects
        cluster_ids: list of int (parallel to mols)
        fp_type, radius, n_bits: fingerprint settings

    Returns:
        dict:
            "pca_df": DataFrame [PC1, PC2, Cluster_ID, SMILES]
            "var_ratio": [float, float]
    """
    n = len(mols)
    if n < 3:
        rows = []
        for i in range(n):
            rows.append({
                "PC1": 0.0, "PC2": 0.0,
                "Cluster_ID": cluster_ids[i],
                "SMILES": Chem.MolToSmiles(mols[i]) if mols[i] else "",
            })
        return {"pca_df": pd.DataFrame(rows), "var_ratio": [0.0, 0.0]}

    fp_arrays = []
    valid_indices = []
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = _mol_to_fp_obj(mol, fp_type, radius, n_bits)
        if fp is None:
            continue
        arr = np.zeros(n_bits if fp_type != "maccs" else 167)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_arrays.append(arr)
        valid_indices.append(i)

    if len(fp_arrays) < 3:
        rows = []
        for i in valid_indices:
            rows.append({
                "PC1": 0.0, "PC2": 0.0,
                "Cluster_ID": cluster_ids[i],
                "SMILES": Chem.MolToSmiles(mols[i]),
            })
        return {"pca_df": pd.DataFrame(rows), "var_ratio": [0.0, 0.0]}

    X = np.array(fp_arrays)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    rows = []
    for j, i in enumerate(valid_indices):
        rows.append({
            "PC1": float(coords[j, 0]),
            "PC2": float(coords[j, 1]),
            "Cluster_ID": cluster_ids[i],
            "SMILES": Chem.MolToSmiles(mols[i]),
        })

    return {
        "pca_df": pd.DataFrame(rows),
        "var_ratio": [float(pca.explained_variance_ratio_[0]),
                      float(pca.explained_variance_ratio_[1])],
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _select_medoid(member_indices, square_distance_matrix):
    """Pick the molecule with lowest average distance to all others in the cluster."""
    if len(member_indices) == 1:
        return member_indices[0]
    idx_arr = np.array(member_indices)
    sub = square_distance_matrix[np.ix_(idx_arr, idx_arr)]
    avg_dists = sub.mean(axis=1)
    return member_indices[int(np.argmin(avg_dists))]


def _auto_select_n_clusters(distance_matrix, max_k=20):
    """Try k=2..max_k, return k with highest silhouette score."""
    n = len(distance_matrix)
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2

    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average",
        )
        labels = model.fit_predict(distance_matrix)
        unique = set(labels)
        if len(unique) < 2 or len(unique) >= n:
            continue
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _renumber_by_size(labels):
    """Renumber cluster labels so cluster 0 is the largest."""
    from collections import Counter
    counts = Counter(labels)
    # Sort by count descending, then by original label for stability
    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], k))
    mapping = {old: new for new, old in enumerate(ranked)}
    return [mapping[l] for l in labels]


def _build_hierarchical_result(labels, smiles, sq, n_clusters, sil_score, params):
    """Build the standardized result dict for hierarchical clustering."""
    n = len(labels)

    # Find medoid for each cluster
    cluster_members = {}
    for i, cid in enumerate(labels):
        cluster_members.setdefault(cid, []).append(i)

    is_rep = [False] * n
    sizes_list = []
    reps_list = []

    for cid in sorted(cluster_members.keys()):
        members = cluster_members[cid]
        size = len(members)
        medoid = _select_medoid(members, sq)
        is_rep[medoid] = True
        sizes_list.append({"Cluster_ID": cid, "Size": size})
        reps_list.append({"Cluster_ID": cid, "Size": size, "SMILES": smiles[medoid]})

    # Sort by size descending
    sizes_list.sort(key=lambda x: x["Size"], reverse=True)
    reps_list.sort(key=lambda x: x["Size"], reverse=True)

    assignments = pd.DataFrame({
        "SMILES": smiles,
        "Cluster_ID": labels,
        "Is_Representative": is_rep,
    })
    cluster_sizes = pd.DataFrame(sizes_list)
    representatives = pd.DataFrame(reps_list)
    n_singletons = sum(1 for s in sizes_list if s["Size"] == 1)

    return {
        "assignments": assignments,
        "cluster_sizes": cluster_sizes,
        "representatives": representatives,
        "n_clusters": n_clusters,
        "n_singletons": n_singletons,
        "silhouette_score": sil_score,
        "method": "hierarchical",
        "params": params,
    }
