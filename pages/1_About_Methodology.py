import streamlit as st

st.title("About & Methodology")

st.header("What this tool does")
st.write(
    """
    This tool helps cheminformatics students and researchers preprocess and featurize
    molecules for QSAR (Quantitative Structure-Activity Relationship) modeling and
    virtual screening workflows. It takes raw SMILES strings, cleans and filters them
    using established cheminformatics rules, and converts the survivors into
    machine-learning-ready numerical features (descriptors and fingerprints).
    """
)

st.header("Preprocessing steps explained")

st.subheader("Standardization")
st.write(
    """
    Different people can draw the same molecule with slightly different SMILES strings
    (different tautomers, charge states, etc.). Standardization normalizes molecules into
    a consistent representation so that comparisons and deduplication work correctly.
    """
)

st.subheader("Salt stripping")
st.write(
    """
    Many real-world compounds are stored as salts (e.g., "drug hydrochloride"). The salt
    counterion (like Cl- or Na+) is not the biologically active part of the molecule, so
    it is removed before further analysis.
    """
)

st.subheader("Lipinski's Rule of Five")
st.write(
    """
    A widely used rule of thumb for oral drug-likeness, based on molecular weight, LogP
    (lipophilicity), hydrogen bond donors, and hydrogen bond acceptors. Molecules that
    violate too many of these properties are less likely to be orally bioavailable.
    """
)

st.subheader("PAINS (Pan-Assay Interference Compounds)")
st.write(
    """
    PAINS filtering removes compounds that show structural patterns known to cause false
    positives in biological assays due to assay interference, not genuine binding. This
    does not mean the compound is inactive, just that hits involving it should be treated
    with extra caution.
    """
)

st.subheader("Brenk filter")
st.write(
    """
    A broader set of structural alerts beyond PAINS, flagging functional groups that are
    often reactive, unstable, or otherwise undesirable in drug candidates.
    """
)

st.subheader("Veber, Ghose, Egan, and Muegge rules")
st.write(
    """
    These are additional, independently published druglikeness rules, each using slightly
    different property thresholds (rotatable bonds, TPSA, molar refractivity, ring count,
    etc.). No single rule is definitive; using several together gives a more complete
    picture of a molecule's drug-like character.
    """
)

st.subheader("Synthetic Accessibility (SA) Score")
st.write(
    """
    Estimates how easy or difficult a molecule would be to synthesize, on a scale from
    1 (easy) to 10 (very difficult), based on fragment contributions and structural
    complexity (Ertl & Schuffenhauer, 2009). This is informational and does not filter
    out molecules automatically.
    """
)

st.subheader("QED (Quantitative Estimate of Druglikeness)")
st.write(
    """
    QED (Bickerton et al., 2012) combines eight physicochemical properties — molecular
    weight, LogP, H-bond donors, H-bond acceptors, PSA, number of rotatable bonds,
    number of aromatic rings, and number of structural alerts — into a single score
    between 0 (least drug-like) and 1 (most drug-like). Unlike the rule-based filters
    above, QED uses a continuous desirability function for each property. It is
    informational and does not filter out molecules automatically.
    """
)

st.header("Fingerprint guide")
st.write(
    """
    - **Morgan (ECFP)**: the most common default for machine learning; captures circular
      atom neighborhoods up to a chosen radius (radius=2 is equivalent to ECFP4, the most
      widely used setting in QSAR literature).
    - **MACCS**: a fixed set of 166 predefined structural keys; faster to compute but
      less detailed than Morgan fingerprints.
    - **Topological (RDKit)**: based on the molecular graph's connectivity, capturing
      atom paths.
    - **Atom Pair**: encodes pairs of atoms and the topological distance between them.
    - **Torsion**: encodes four-atom sequences (torsion angles), capturing local 3D-like
      connectivity from a 2D structure.
    - **Avalon**: a fingerprint combining several substructure feature types, used in
      several commercial and open-source cheminformatics tools.
    """
)

st.header("Descriptor glossary")
st.write(
    """
    - **MW (Molecular Weight)**: the sum of atomic weights, in g/mol.
    - **LogP**: a measure of lipophilicity (fat solubility vs water solubility); higher
      values mean more lipophilic.
    - **TPSA (Topological Polar Surface Area)**: the surface area contributed by polar
      atoms, in Å²; relevant to membrane permeability.
    - **HBD / HBA**: counts of hydrogen bond donors and acceptors.
    - **Rotatable Bonds**: bonds that can rotate freely, related to molecular flexibility.
    - **Aromatic Rings**: count of aromatic ring systems in the molecule.
    """
)

st.header("References")
st.write(
    """
    - Landrum, G. *RDKit: Open-source cheminformatics.* https://www.rdkit.org
    - Lipinski, C. A., et al. (2001). Experimental and computational approaches to
      estimate solubility and permeability in drug discovery and development settings.
      *Advanced Drug Delivery Reviews*, 46(1-3), 3-26.
    - Baell, J. B., & Holloway, G. A. (2010). New substructure filters for removal of pan
      assay interference compounds (PAINS). *Journal of Medicinal Chemistry*, 53(7),
      2719-2740.
    - Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of
      Chemical Information and Modeling*, 50(5), 742-754.
    - Ertl, P., & Schuffenhauer, A. (2009). Estimation of synthetic accessibility score of
      drug-like molecules based on molecular complexity and fragment contributions.
      *Journal of Cheminformatics*, 1(1), 8.
    - Bickerton, G. R., et al. (2012). Quantifying the chemical beauty of drugs.
      *Nature Chemistry*, 4(2), 90-98.
    - MolVS: Molecule Validation and Standardization. https://github.com/mcs07/MolVS
    """
)
