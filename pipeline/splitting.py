"""
Train/test splitting strategies for QSAR datasets.

Provides scaffold-based, random, and stratified-random splitting,
plus PCA projection for visualizing chemical space coverage.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_murcko_scaffolds(mols):
    """
    Extract Bemis-Murcko scaffold SMILES for each molecule.

    Parameters:
        mols: list of RDKit Mol objects (may contain None)

    Returns:
        list of scaffold SMILES strings. Empty string for acyclic
        molecules or if extraction fails. None entries yield "".
    """
    scaffolds = []
    for mol in mols:
        if mol is None:
            scaffolds.append("")
            continue
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(core)
            scaffolds.append(smi if smi else "")
        except Exception:
            scaffolds.append("")
    return scaffolds


def scaffold_split(mols, smiles, test_size=0.2, random_state=42, labels=None):
    """
    Split molecules so no Murcko scaffold appears in both train and test.

    Scaffolds are sorted largest-first, then greedily assigned to test
    until the test budget is met; remaining go to train. Tie-breaking
    uses random_state for reproducibility.

    Parameters:
        mols: list of RDKit Mol objects
        smiles: list of SMILES strings (parallel to mols)
        test_size: target fraction for test set (0.0–1.0)
        random_state: seed for reproducible tie-breaking
        labels: optional list of activity labels (parallel to mols).
                Included as a "Label" column in output DataFrames if provided.

    Returns:
        dict with keys:
            train_indices, test_indices: list[int]
            train_df, test_df: DataFrame with [SMILES, Scaffold, Split] (+ Label if provided)
            scaffold_counts: DataFrame [Scaffold, Count] sorted descending
            n_scaffolds: int
            n_singletons: int (scaffolds with exactly 1 molecule)
            achieved_test_fraction: float
            fallback: str or None — set if split couldn't produce a non-empty test set
    """
    n = len(mols)
    scaffolds = extract_murcko_scaffolds(mols)

    # Group molecule indices by scaffold
    scaffold_to_indices = {}
    for i, scaf in enumerate(scaffolds):
        scaffold_to_indices.setdefault(scaf, []).append(i)

    # Scaffold counts table
    scaffold_counts = pd.DataFrame([
        {"Scaffold": scaf, "Count": len(idxs)}
        for scaf, idxs in scaffold_to_indices.items()
    ]).sort_values("Count", ascending=False).reset_index(drop=True)

    n_scaffolds = len(scaffold_to_indices)
    n_singletons = sum(1 for idxs in scaffold_to_indices.values() if len(idxs) == 1)

    # Edge case: only one scaffold group — can't split
    if n_scaffolds <= 1:
        all_indices = list(range(n))
        train_df = _build_split_df(all_indices, smiles, scaffolds, "Train", labels)
        test_df = _build_split_df([], smiles, scaffolds, "Test", labels)
        return {
            "train_indices": all_indices,
            "test_indices": [],
            "train_df": train_df,
            "test_df": test_df,
            "scaffold_counts": scaffold_counts,
            "n_scaffolds": n_scaffolds,
            "n_singletons": n_singletons,
            "achieved_test_fraction": 0.0,
            "fallback": "single_scaffold",
        }

    # Sort scaffold groups: largest first, then by scaffold SMILES for determinism
    rng = np.random.RandomState(random_state)
    groups = list(scaffold_to_indices.items())
    # Shuffle first so equal-size groups get random order
    rng.shuffle(groups)
    # Stable sort by size descending
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    test_target = int(np.ceil(n * test_size))
    test_indices = []
    train_indices = []

    for _scaf, idxs in groups:
        if len(test_indices) < test_target:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    # If greedy assigned everything to test (very small dataset), move overflow to train
    if not train_indices and test_indices:
        # Keep only enough in test; move the rest to train
        test_indices_sorted = sorted(test_indices)
        actual_test = test_indices_sorted[:test_target]
        train_indices = test_indices_sorted[test_target:]
        test_indices = actual_test

    test_indices.sort()
    train_indices.sort()

    achieved = len(test_indices) / n if n > 0 else 0.0

    train_df = _build_split_df(train_indices, smiles, scaffolds, "Train", labels)
    test_df = _build_split_df(test_indices, smiles, scaffolds, "Test", labels)

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_df": train_df,
        "test_df": test_df,
        "scaffold_counts": scaffold_counts,
        "n_scaffolds": n_scaffolds,
        "n_singletons": n_singletons,
        "achieved_test_fraction": round(achieved, 4),
        "fallback": None,
    }


def random_split(mols, smiles, test_size=0.2, random_state=42,
                 labels=None, stratify=False):
    """
    Random or stratified-random split.

    Parameters:
        mols: list of RDKit Mol objects (unused, accepted for API symmetry)
        smiles: list of SMILES strings
        test_size: fraction for test set
        random_state: seed
        labels: activity labels (required if stratify=True)
        stratify: if True, use sklearn's stratified split on labels

    Returns:
        dict with keys:
            train_indices, test_indices: list[int]
            train_df, test_df: DataFrame [SMILES, Split] (+ Label if provided)
            achieved_test_fraction: float
            fallback: str or None

    Raises:
        ValueError: if stratify=True but labels is None or has < 2 classes
    """
    n = len(smiles)

    if n < 2:
        train_df = pd.DataFrame({"SMILES": smiles, "Split": ["Train"] * n})
        if labels is not None:
            train_df["Label"] = labels
        return {
            "train_indices": list(range(n)),
            "test_indices": [],
            "train_df": train_df,
            "test_df": pd.DataFrame(columns=train_df.columns),
            "achieved_test_fraction": 0.0,
            "fallback": "too_small",
        }

    indices = np.arange(n)
    stratify_arr = None

    if stratify:
        if labels is None:
            raise ValueError("stratify=True requires labels to be provided")
        if any(l is None for l in labels):
            raise ValueError(
                "Labels contain None values. Remove or exclude unlabeled "
                "molecules before stratified splitting."
            )
        unique = set(labels)
        if len(unique) < 2:
            raise ValueError(
                f"Stratified split requires at least 2 classes, got {len(unique)}"
            )
        stratify_arr = labels

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=stratify_arr,
    )
    train_idx = sorted(train_idx.tolist())
    test_idx = sorted(test_idx.tolist())

    achieved = len(test_idx) / n if n > 0 else 0.0

    train_df = pd.DataFrame({
        "SMILES": [smiles[i] for i in train_idx],
        "Split": "Train",
    })
    test_df = pd.DataFrame({
        "SMILES": [smiles[i] for i in test_idx],
        "Split": "Test",
    })
    if labels is not None:
        train_df["Label"] = [labels[i] for i in train_idx]
        test_df["Label"] = [labels[i] for i in test_idx]

    return {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "train_df": train_df,
        "test_df": test_df,
        "achieved_test_fraction": round(achieved, 4),
        "fallback": None,
    }


def compute_split_pca(mols, train_indices, test_indices,
                      fp_type="morgan", radius=2, n_bits=1024):
    """
    Compute 2D PCA of molecular fingerprints, labeled by split assignment.

    Parameters:
        mols: list of RDKit Mol objects
        train_indices: list of int
        test_indices: list of int
        fp_type: fingerprint type for projection
        radius: Morgan radius
        n_bits: fingerprint length

    Returns:
        dict with keys:
            pca_df: DataFrame [PC1, PC2, Split, SMILES]
            var_ratio: [float, float] — explained variance ratio for PC1, PC2
    """
    from pipeline.featurization import _mol_to_fp_obj

    all_indices = sorted(set(train_indices) | set(test_indices))
    if len(all_indices) < 3:
        # PCA needs at least 3 points
        rows = []
        for i in all_indices:
            split = "Train" if i in set(train_indices) else "Test"
            rows.append({"PC1": 0.0, "PC2": 0.0, "Split": split,
                         "SMILES": Chem.MolToSmiles(mols[i]) if mols[i] else ""})
        return {"pca_df": pd.DataFrame(rows), "var_ratio": [0.0, 0.0]}

    # Compute fingerprints
    from rdkit import DataStructs
    fp_arrays = []
    valid_indices = []
    for i in all_indices:
        mol = mols[i]
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
        train_set = set(train_indices)
        for i in valid_indices:
            rows.append({"PC1": 0.0, "PC2": 0.0,
                         "Split": "Train" if i in train_set else "Test",
                         "SMILES": Chem.MolToSmiles(mols[i])})
        return {"pca_df": pd.DataFrame(rows), "var_ratio": [0.0, 0.0]}

    X = np.array(fp_arrays)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    train_set = set(train_indices)
    rows = []
    for j, i in enumerate(valid_indices):
        rows.append({
            "PC1": float(coords[j, 0]),
            "PC2": float(coords[j, 1]),
            "Split": "Train" if i in train_set else "Test",
            "SMILES": Chem.MolToSmiles(mols[i]),
        })

    return {
        "pca_df": pd.DataFrame(rows),
        "var_ratio": [float(pca.explained_variance_ratio_[0]),
                      float(pca.explained_variance_ratio_[1])],
    }


def _build_split_df(indices, smiles, scaffolds, split_label, labels=None):
    """Build a DataFrame for a train or test partition."""
    if not indices:
        cols = ["SMILES", "Scaffold", "Split"]
        if labels is not None:
            cols.append("Label")
        return pd.DataFrame(columns=cols)
    data = {
        "SMILES": [smiles[i] for i in indices],
        "Scaffold": [scaffolds[i] for i in indices],
        "Split": split_label,
    }
    if labels is not None:
        data["Label"] = [labels[i] for i in indices]
    return pd.DataFrame(data)
