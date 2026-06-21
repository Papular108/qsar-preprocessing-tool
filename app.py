import streamlit as st
from pipeline.preprocessing import parse_smiles

st.title("QSAR Preprocessing Tool")
st.write("Welcome! This tool helps preprocess and featurize molecules for QSAR/virtual screening workflows.")

st.header("Try it: parse a SMILES string")

smiles_input = st.text_input("Enter a SMILES string", value="CCO")

if st.button("Parse"):
    mol, error = parse_smiles(smiles_input)

    if error:
        st.error(error)
    else:
        st.success("Successfully parsed!")
        st.write(f"Number of atoms: {mol.GetNumAtoms()}")
