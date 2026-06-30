import streamlit as st
import pandas as pd
from datetime import datetime
from rdkit import rdBase, Chem
from pipeline.preprocessing import parse_smiles, run_preprocessing_pipeline
from pipeline.featurization import featurize_dataset, compute_descriptors
import altair as alt
from pipeline.visualization import mol_to_base64_png
from pipeline.methodology import generate_methods_text
from pipeline.example_data import get_fda_approved_drugs, get_pains_demo_set

def _mol_table_html(rows, smiles_key, extra_keys, img_size=(150, 150)):
    """Render a list of dicts as an HTML table with an embedded structure image column."""
    header_cells = "".join(f"<th style='padding:6px 10px;text-align:left;border-bottom:1px solid #ddd'>{k}</th>" for k in [smiles_key] + extra_keys)
    header_cells += "<th style='padding:6px 10px;text-align:left;border-bottom:1px solid #ddd'>Structure</th>"

    html_rows = []
    for row in rows:
        cells = "".join(
            f"<td style='padding:6px 10px;vertical-align:middle'>{row.get(k, '')}</td>"
            for k in [smiles_key] + extra_keys
        )
        mol = Chem.MolFromSmiles(str(row.get(smiles_key, "")))
        if mol:
            b64 = mol_to_base64_png(mol, size=img_size)
            img_tag = f'<img src="data:image/png;base64,{b64}" style="display:block" />'
        else:
            img_tag = "<em>invalid SMILES</em>"
        cells += f"<td style='padding:4px 10px;vertical-align:middle'>{img_tag}</td>"
        html_rows.append(f"<tr style='border-bottom:1px solid #eee'>{cells}</tr>")

    body = "".join(html_rows)
    return (
        f"<div style='overflow-x:auto'>"
        f"<table style='border-collapse:collapse;width:100%'>"
        f"<thead><tr style='background:#f5f5f5'>{header_cells}</tr></thead>"
        f"<tbody>{body}</tbody>"
        f"</table></div>"
    )


@st.cache_data
def _cached_descriptors(smiles_tuple):
    rows = []
    for smi in smiles_tuple:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rows.append(compute_descriptors(mol))
    return pd.DataFrame(rows)


@st.cache_data
def _cached_featurize(smiles_tuple, fp_type, radius, n_bits):
    mols = [Chem.MolFromSmiles(s) for s in smiles_tuple]
    return featurize_dataset(mols, fp_type=fp_type, radius=radius, n_bits=n_bits)


def build_metadata_block(settings):
    lines = ['# === Reproducibility Metadata ===']
    lines.append(f'# Generated: {datetime.now().isoformat()}')
    lines.append(f'# RDKit version: {rdBase.rdkitVersion}')
    for key, value in settings.items():
        lines.append(f'# {key}: {value}')
    lines.append('# ================================')
    return chr(10).join(lines) + chr(10)

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

if "pasted_smiles" not in st.session_state:
    st.session_state["pasted_smiles"] = ""

with st.expander("Try with example data", expanded=False):
    _EXAMPLE_OPTIONS = {
        "FDA-approved drugs (20 molecules)": get_fda_approved_drugs,
        "PAINS-rich demo set (15 molecules)": get_pains_demo_set,
    }
    example_choice = st.selectbox(
        "Select a dataset",
        list(_EXAMPLE_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    smiles_list_ex, description_ex = _EXAMPLE_OPTIONS[example_choice]()
    st.caption(description_ex)
    if st.button("Load example data"):
        st.session_state["pasted_smiles"] = "\n".join(smiles_list_ex)
        st.rerun()

uploaded_file = st.file_uploader("Upload a .txt or .csv file with SMILES (one per line)", type=["txt", "csv", "xlsx"])
pasted_smiles = st.text_area("Or paste SMILES here (one per line)", key="pasted_smiles")

max_violations = st.number_input(
    "Max Lipinski violations allowed",
    min_value=0,
    max_value=4,
    value=1,
    step=1,
    help="Lipinski's Rule of Five flags molecules unlikely to be orally bioavailable. The standard threshold is 1 violation; higher values are more permissive and may include less drug-like compounds.",
)

st.write("Optional additional filters:")
col1, col2, col3 = st.columns(3)
with col1:
    enable_veber = st.checkbox("Veber", help="Rotatable bonds <=10 and TPSA <=140")
    enable_ghose = st.checkbox("Ghose", help="Drug-like property ranges (MW, LogP, molar refractivity, atom count)")
with col2:
    enable_egan = st.checkbox("Egan", help="LogP and TPSA within the Egan egg boundary")
    enable_muegge = st.checkbox("Muegge", help="Combined pharmacophore-like property rule")
with col3:
    enable_brenk = st.checkbox("Brenk", help="Flags additional structural alerts beyond PAINS")
    enable_sa_score = st.checkbox("SA Score", help="Report synthetic accessibility score (1=easy, 10=hard) for kept molecules. Informational only, does not filter.")
    enable_qed = st.checkbox("QED Score", help="Quantitative Estimate of Druglikeness (0=least drug-like, 1=most drug-like; Bickerton et al., 2012). Informational only, does not filter.")

if st.button("Run Pipeline"):
    smiles_list = []
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv") or file_name.endswith(".xlsx"):
            if file_name.endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
            smiles_column = None
            for candidate in ["canonical_smiles", "Smiles", "SMILES", "smiles"]:
                if candidate in uploaded_df.columns:
                    smiles_column = candidate
                    break
            if smiles_column is None:
                st.warning("Could not auto-detect a SMILES column. Please select it manually below.")
                smiles_column = st.selectbox("Which column contains the SMILES strings?", uploaded_df.columns.tolist())
            smiles_list = uploaded_df[smiles_column].dropna().astype(str).tolist()
        else:
            file_content = uploaded_file.read().decode("utf-8")
            smiles_list = [line.strip() for line in file_content.splitlines() if line.strip()]
    elif pasted_smiles.strip():
        smiles_list = [line.strip() for line in pasted_smiles.splitlines() if line.strip()]

    if not smiles_list:
        st.warning("Please upload a file or paste at least one SMILES string.")
    else:
        result = run_preprocessing_pipeline(
            smiles_list,
            lipinski_max_violations=max_violations,
            enable_veber=enable_veber,
            enable_ghose=enable_ghose,
            enable_egan=enable_egan,
            enable_muegge=enable_muegge,
            enable_brenk=enable_brenk,
            enable_sa_score=enable_sa_score,
            enable_qed=enable_qed,
        )
        st.session_state["kept_mols_for_featurization"] = result["kept_mols"]
        st.session_state["kept_smiles_for_featurization"] = result["kept_smiles"]
        st.session_state["pipeline_settings"] = {
            "lipinski_max_violations": max_violations,
            "enable_veber": enable_veber,
            "enable_ghose": enable_ghose,
            "enable_egan": enable_egan,
            "enable_muegge": enable_muegge,
            "enable_brenk": enable_brenk,
            "enable_sa_score": enable_sa_score,
            "enable_qed": enable_qed,
            "input_count": len(smiles_list),
            "kept_count": len(result["kept_smiles"]),
        }

        st.subheader("Results")
        st.write(f"Input molecules: {len(smiles_list)}")
        st.write(f"Kept after preprocessing: {len(result['kept_smiles'])}")

        st.subheader("Audit Trail")
        audit_df = pd.DataFrame(result["audit_trail"])
        st.dataframe(audit_df)

        if result["removed_log"]:
            st.subheader("Removed Molecules (with reasons)")
            st.markdown(
                _mol_table_html(result["removed_log"], "smiles", ["original_index", "step", "reason"]),
                unsafe_allow_html=True,
            )

        st.subheader("Kept Molecules (SMILES)")
        kept_data = {"SMILES": result["kept_smiles"]}
        if result["sa_scores"] is not None:
            kept_data["SA_Score"] = result["sa_scores"]
        if result["qed_scores"] is not None:
            kept_data["QED_Score"] = result["qed_scores"]
        kept_df = pd.DataFrame(kept_data)
        extra_kept = [c for c in ["SA_Score", "QED_Score"] if c in kept_df.columns]
        st.markdown(
            _mol_table_html(kept_df.to_dict("records"), "SMILES", extra_kept),
            unsafe_allow_html=True,
        )

        metadata = build_metadata_block({
            "Lipinski max violations": max_violations,
            "Veber filter": enable_veber,
            "Ghose filter": enable_ghose,
            "Egan filter": enable_egan,
            "Muegge filter": enable_muegge,
            "Brenk filter": enable_brenk,
            "SA score reported": enable_sa_score,
            "QED score reported": enable_qed,
            "Input molecule count": len(smiles_list),
            "Kept molecule count": len(result["kept_smiles"]),
        })
        csv_data = metadata + kept_df.to_csv(index=False)
        st.download_button("Download kept molecules as CSV", data=csv_data, file_name="kept_molecules.csv", mime="text/csv")

        st.subheader("Descriptor Distributions")
        if len(result["kept_smiles"]) < 2:
            st.info("At least 2 kept molecules are needed to show distributions.")
        else:
            desc_df = _cached_descriptors(tuple(result["kept_smiles"]))
            if result["qed_scores"] is not None:
                desc_df["QED"] = result["qed_scores"]
            st.caption(
                "Physicochemical property distributions for the kept molecules. "
                "Hover over bars for exact counts."
            )
            _DIST_FIELDS = [
                ("MW",             "Molecular Weight (Da)"),
                ("LogP",           "LogP"),
                ("TPSA",           "TPSA (Å²)"),
                ("HBD",            "H-bond Donors"),
                ("HBA",            "H-bond Acceptors"),
                ("RotatableBonds", "Rotatable Bonds"),
            ]
            if result["qed_scores"] is not None:
                _DIST_FIELDS.append(("QED", "QED Score (0–1)"))
            _grid_rows = [st.columns(3) for _ in range((len(_DIST_FIELDS) + 2) // 3)]
            for i, (field, label) in enumerate(_DIST_FIELDS):
                with _grid_rows[i // 3][i % 3]:
                    chart = (
                        alt.Chart(desc_df)
                        .mark_bar(color="#4C72B0")
                        .encode(
                            alt.X(f"{field}:Q", bin=alt.Bin(maxbins=15), title=label),
                            alt.Y("count()", title="Count"),
                            tooltip=[
                                alt.Tooltip(f"{field}:Q", bin=True, title=label),
                                alt.Tooltip("count()", title="Count"),
                            ],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(chart, use_container_width=True)


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
    if "kept_smiles_for_featurization" not in st.session_state:
        st.warning("Please run the preprocessing pipeline first, above.")
    else:
        feature_df, feat_errors = _cached_featurize(
            tuple(st.session_state["kept_smiles_for_featurization"]),
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

        feat_metadata = build_metadata_block({
            "Fingerprint type": fp_type,
            "Fingerprint bits": n_bits,
            "Morgan radius": radius,
            "Molecule count": feature_df.shape[0],
            "Feature column count": feature_df.shape[1],
        })
        csv_data = feat_metadata + feature_df.to_csv(index=False)
        st.download_button(
            "Download featurized data as CSV",
            data=csv_data,
            file_name="featurized_data.csv",
            mime="text/csv",
        )

st.header("Generate Methods Section")
st.write("Generate a publication-ready paragraph describing the preprocessing and featurization steps you used.")

if st.button("Generate Methods Section"):
    if "pipeline_settings" not in st.session_state:
        st.warning("Please run the preprocessing pipeline first.")
    else:
        settings = dict(st.session_state["pipeline_settings"])
        settings["fp_type"] = fp_type
        settings["n_bits"] = n_bits
        settings["radius"] = radius
        methods_text = generate_methods_text(settings)
        st.text_area(
            "Methods paragraph (select all and copy)",
            value=methods_text,
            height=250,
        )
        st.caption("Tip: click inside the text box, then press Ctrl+A / Cmd+A to select all.")
