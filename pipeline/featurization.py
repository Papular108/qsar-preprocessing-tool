from rdkit.Chem import Descriptors


def compute_descriptors(mol):
    """
    Compute key physicochemical descriptors for a molecule.

    Parameters:
        mol: RDKit Mol object

    Returns:
        dict: descriptor name -> value
    """
    descriptors = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
    }

    return descriptors
