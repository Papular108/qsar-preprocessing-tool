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

st.header("Similarity Search (Tanimoto)")
st.write(
    """
    The Similarity Search feature in Molecule Explorer finds the most similar molecules
    in your preprocessed dataset to a query molecule, using Tanimoto similarity on
    molecular fingerprints.

    **Tanimoto coefficient** (also called Jaccard index for bit vectors) measures the
    overlap between two fingerprints:

    `Tanimoto(A, B) = |A ∩ B| / |A ∪ B|`

    where |A ∩ B| is the number of bits set in both fingerprints and |A ∪ B| is the
    number of bits set in either. The score ranges from 0 (no shared features) to 1
    (identical fingerprints).

    **Interpretation guidelines:**
    - **> 0.85**: Near-duplicates or same Murcko scaffold; likely same chemical series
    - **0.7 – 0.85**: Close analogs; often share the same core with minor substitutions
    - **0.4 – 0.7**: Moderate similarity; may share pharmacophoric features
    - **< 0.4**: Structurally dissimilar; different scaffolds

    **Activity cliffs** occur when structurally similar molecules (high Tanimoto) have
    very different biological activities. These are important for SAR analysis — small
    structural changes that dramatically affect potency can reveal key binding interactions.

    The fingerprint type matters: Morgan (ECFP) fingerprints tend to give lower similarity
    scores than MACCS keys for the same molecule pair, because Morgan captures finer
    structural detail. Choose the fingerprint type that matches your downstream ML model.
    """
)

st.header("Multi-Reference Similarity Screening")
st.write(
    """
    The Similarity Screening tab extends single-query similarity search into a
    virtual screening workflow. You provide a **set of reference molecules**
    (e.g., known actives, lead compounds) and a **target library** (e.g., a
    vendor catalog or your preprocessed dataset), and the tool computes Tanimoto
    similarity between every reference-target pair.

    **Key outputs:**
    - **Per-reference hit list:** the top-N most similar targets for each reference,
      ranked by Tanimoto score
    - **All-hits table:** a deduplicated list of every target that exceeded the
      similarity threshold for at least one reference, annotated with its best-matching
      reference and similarity score
    - **Similarity heatmap:** a color-coded matrix (references x targets) showing
      pairwise similarities at a glance — useful for spotting clusters of related
      compounds and identifying which references share common hits
    - **Downloadable CSV:** the all-hits table can be exported for downstream analysis

    **When to use this:**
    - **Hit expansion:** given a few confirmed actives, find structurally similar
      compounds in a larger library that may also be active
    - **SAR exploration:** compare multiple analogs against a common target set to
      understand structure-activity relationships
    - **Library profiling:** assess how well a screening library covers the chemical
      space around your references

    All fingerprint types supported elsewhere (Morgan, FCFP, MACCS, topological,
    atom pair, torsion, avalon) are available here. Target fingerprints are computed
    once and reused across all references for efficiency.
    """
)

st.header("Train/Test Splitting")
st.write(
    """
    The Train/Test Split tab generates train/test partitions for QSAR modeling.
    Three strategies are available:

    **Scaffold split (recommended for QSAR benchmarking):**
    Molecules are grouped by their Bemis-Murcko scaffold — the core ring system
    with linkers, but stripped of side chains. No scaffold appears in both the
    training and test sets. This forces the model to generalize to entirely new
    chemotypes at test time, giving a realistic estimate of performance on novel
    compounds.

    The algorithm sorts scaffold groups by size (largest first) and greedily
    assigns them to the test set until the target fraction is reached. Because
    entire scaffold groups move together, the achieved test fraction may deviate
    from the target — this is expected and reported in the summary.

    **Random split:**
    Standard random partitioning. Train and test sets will share scaffolds,
    making the task easier and performance estimates optimistic.

    **Stratified random split:**
    Like random, but preserves the class distribution of activity labels in
    both sets. Useful for imbalanced classification tasks.

    **Interpreting the PCA visualization:**
    The chemical space plot projects molecular fingerprints into 2D using PCA.
    In a scaffold split, train (blue) and test (orange) points tend to cluster
    separately — the model must extrapolate to new chemical regions. In a random
    split, they overlap — the model can interpolate, which is easier but less
    realistic.

    **References:**
    - Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs.
      1. Molecular frameworks. *J. Med. Chem.*, 39(15), 2887-2893.
    - Wu, Z., et al. (2018). MoleculeNet: a benchmark for molecular machine
      learning. *Chemical Science*, 9(2), 513-530.
    """
)

st.header("Cluster Analysis")
st.write(
    """
    The Cluster Analysis tab groups molecules by fingerprint similarity,
    complementing the scaffold-based analysis in preprocessing. While
    scaffolds group by core ring system, clustering captures overall
    structural similarity including side chains and functional groups.

    **Butina clustering (Taylor-Butina):**
    A distance-based algorithm native to RDKit. You set a Tanimoto distance
    cutoff (e.g., 0.4, meaning molecules need similarity >= 0.6 to cluster).
    The algorithm finds the molecule with the most neighbors within the cutoff
    (the centroid), assigns it and its neighbors to a cluster, then repeats.
    The number of clusters emerges naturally from the data and cutoff.

    - **Tight cutoff (e.g., 0.2):** many small, highly similar clusters
    - **Loose cutoff (e.g., 0.7):** fewer, broader clusters
    - Best for: exploring natural groupings, when you don't know how many
      clusters to expect

    **Hierarchical (agglomerative) clustering:**
    Builds a hierarchy by iteratively merging the closest clusters using
    average linkage on the Tanimoto distance matrix. You specify the number
    of clusters, or let the tool auto-select it by maximizing the silhouette
    score over a range of k values.

    - **Silhouette score** ranges from -1 to 1. Higher values mean clusters
      are well-separated and internally cohesive. Scores above 0.5 indicate
      strong structure; below 0.25 may mean the data doesn't cluster cleanly.
    - Best for: when you need a specific number of groups, or want a quality
      metric for cluster assignments

    **Representative (medoid) selection:**
    For each cluster, the representative is the medoid — the molecule with
    the lowest average Tanimoto distance to all other cluster members. This
    is the most "central" molecule in the cluster, making it a good choice
    for diverse subset selection.

    **Diverse subset selection:**
    Picking one representative per cluster gives a maximally diverse subset
    of your dataset. This is useful for screening library design (maximize
    chemical coverage with fewer compounds), selecting training sets, or
    prioritizing compounds for experimental testing.

    **Reference:**
    - Butina, D. (1999). Unsupervised Database Clustering Based on Daylight's
      Fingerprint and Tanimoto Similarity: A Fast and Automated Way to Cluster
      Small and Large Data Sets. *J. Chem. Inf. Comput. Sci.*, 39(4), 747-750.
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
