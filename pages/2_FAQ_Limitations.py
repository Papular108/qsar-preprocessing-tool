import streamlit as st

st.title("FAQ & Limitations")

st.header("What this tool does NOT do")
st.write(
    """
    - This tool performs **2D-based preprocessing and featurization only**. It does not
      perform 3D conformer generation, molecular docking, or ADMET property prediction
      (e.g., GI absorption, BBB permeability, CYP inhibition).
    - PAINS and Brenk filtering identify **known interference/alert patterns**, but do
      not guarantee that a flagged compound is biologically inactive, or that an
      unflagged compound is active.
    - Druglikeness rules (Lipinski, Veber, Ghose, Egan, Muegge) are **heuristics based on
      historical drug datasets**, not hard biological laws. Exceptions exist in real,
      approved drugs.
    - The Synthetic Accessibility (SA) score is an **estimate**, not a guarantee of
      synthesizability or cost.
    - This tool does not replace expert medicinal chemistry judgment, wet-lab validation,
      or regulatory assessment.
    """
)

st.header("Frequently Asked Questions")

st.subheader("Why was my molecule removed?")
st.write(
    """
    Check the "Removed Molecules" table after running the pipeline — it shows exactly
    which step removed each molecule and why (e.g., a specific PAINS pattern, a Lipinski
    violation with the actual values, or a parsing error).
    """
)

st.subheader("Should I always enable every filter?")
st.write(
    """
    Not necessarily. PAINS and Lipinski are commonly used defaults, but Veber, Ghose,
    Egan, Muegge, and Brenk are more specialized or stricter. Consider your specific
    research context — for example, fragment-based screening often uses more permissive
    thresholds than late-stage lead optimization.
    """
)

st.subheader("Which fingerprint should I use?")
st.write(
    """
    Morgan (ECFP, radius=2) is the most common default for general QSAR/ML work. MACCS
    is faster but less detailed. See the About/Methodology page for a fuller comparison.
    """
)

st.subheader("Can I trust the exact descriptor values for publication?")
st.write(
    """
    Always validate results against your specific assay context and, where possible,
    cross-check critical values independently before using them in downstream analysis
    or publication. The reproducibility metadata included in exported files (RDKit
    version, settings used, timestamp) is provided to support this.
    """
)

st.subheader("Is this tool affiliated with SwissADME or any other commercial tool?")
st.write(
    """
    No. This is an independent, open-source tool built for the cheminformatics community,
    conceptually inspired by tools like SwissADME but built from scratch using RDKit and
    MolVS.
    """
)
