import streamlit as st
import pandas as pd
from pipeline.preprocessing import parse_smiles, run_preprocessing_pipeline
from pipeline.featurization import featurize_dataset

st.title("QSAR Preprocessing Tool")
st.write("Welcome! This tool helps preprocess and featurize molecules for QSAR/virtual screening workflows.")

st.header("Try it: parse a SMILES string")

smiles_input = st.text_input(
    "Enter a SMILES string",
    value="CCO",
    help="SMILES (Simplified Molecular Input Line Entry System) is a text notation for chemical structures, e.g. CCO represents ethanol.",
)

if st.button("Parse"):
    mol, error = parse_smiles(smiles_input)

    if error:
        st.error(error)
    else:
        st.success("Successfully parsed!")
        st.write(f"Number of atoms: {mol.GetNumAtoms()}")

st.header("Batch preprocessing")
st.write("Upload a file with one SMILES string per line, or paste them below.")

uploaded_file = st.file_uploader("Upload a .txt or .csv file with SMILES (one per line)", type=["txt", "csv"])
pasted_smiles = st.text_area("Or paste SMILES here (one per line)")

max_violations = st.number_input(
    "Max Lipinski violations allowed",
    min_value=0,
    max_value=4,
    value=1,
    step=1,
    help="Lipinski's Rule of Five flags molecules unlikely to be orally bioavailable. The standard threshold is 1 violation; higher values are more permissive and may include less drug-like compounds.",
)

if st.button("Run Pipeline"):
    smiles_list = []

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        smiles_list = [line.strip() for line in content.splitlines() if line.strip()]
    elif pasted_smiles.strip():
        smiles_list = [line.strip() for line in pasted_smiles.splitlines() if line.strip()]

    if not smiles_list:
        st.warning("Please upload a file or paste at least one SMILES string.")
    else:
        result = run_preprocessing_pipeline(smiles_list, lipinski_max_violations=max_violations)
        st.session_state["kept_mols_for_featurization"] = result["kept_mols"]

        st.subheader("Results")
        st.write(f"Input molecules: {len(smiles_list)}")
        st.write(f"Kept after preprocessing: {len(result['kept_smiles'])}")

        st.subheader("Audit Trail")
        audit_df = pd.DataFrame(result["audit_trail"])
        st.dataframe(audit_df)

        if result["removed_log"]:
            st.subheader("Removed Molecules (with reasons)")
            removed_df = pd.DataFrame(result["removed_log"])
            st.dataframe(removed_df)

        st.subheader("Kept Molecules (SMILES)")
        kept_df = pd.DataFrame({"SMILES": result["kept_smiles"]})
        st.dataframe(kept_df)

        csv_data = kept_df.to_csv(index=False)
        st.download_button("Download kept molecules as CSV", data=csv_data, file_name="kept_molecules.csv", mime="text/csv")


st.header("Featurization")
st.write("After preprocessing, generate descriptors and fingerprints for your kept molecules.")

fp_type = st.selectbox(
    "Fingerprint type",
    ["morgan", "maccs", "topological", "atom_pair", "torsion", "avalon"],
    help="Morgan (ECFP) is the most common default for machine learning. MACCS is faster but less detailed (166 fixed structural keys). Others are specialized for specific use cases.",
)
if fp_type == "maccs":
    st.caption("MACCS keys are a fixed, predefined set of 166 structural patterns — bit size is not adjustable.")
    n_bits = 167
else:
    n_bits = st.selectbox(
    "Fingerprint size (bits)",
    [512, 1024, 2048],
    index=2,
    help="Larger bit vectors capture more structural detail but increase memory and computation. 2048 is the most common choice in QSAR literature.",
)

if fp_type == "morgan":
    radius = st.number_input(
        "Morgan radius",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Radius=2 (equivalent to ECFP4) is the most widely used setting in QSAR literature. Higher radius captures larger structural neighborhoods.",
    )
else:
    radius = 2

if st.button("Run Featurization"):
    if "kept_mols_for_featurization" not in st.session_state:
        st.warning("Please run the preprocessing pipeline first, above.")
    else:
        feature_df, feat_errors = featurize_dataset(
            st.session_state["kept_mols_for_featurization"],
            fp_type=fp_type,
            radius=radius,
            n_bits=n_bits,
        )

        st.subheader("Featurized Data")
        st.write(f"Shape: {feature_df.shape}")
        st.dataframe(feature_df.head(20))

        if feat_errors:
            st.warning(f"{len(feat_errors)} molecules failed featurization.")
            st.dataframe(pd.DataFrame(feat_errors))

        csv_data = feature_df.to_csv(index=False)
        st.download_button(
            "Download featurized data as CSV",
            data=csv_data,
            file_name="featurized_data.csv",
            mime="text/csv",
        )
