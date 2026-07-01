import streamlit as st

st.set_page_config(page_title="QSAR Preprocessing Tool", page_icon="🧪", layout="wide")

import pandas as pd
from datetime import datetime
from rdkit import rdBase, Chem
from pipeline.preprocessing import run_preprocessing_pipeline, label_activity
from pipeline.featurization import featurize_dataset, compute_descriptors
import altair as alt
from pipeline.visualization import mol_to_base64_png
from pipeline.methodology import generate_methods_text
from pipeline.example_data import get_fda_approved_drugs, get_pains_demo_set
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


@st.cache_data
def _cached_chemical_space_pca(smiles_tuple):
    desc_cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds"]

    user_rows = []
    for smi in smiles_tuple:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            d = compute_descriptors(mol)
            d["SMILES"] = smi
            d["Group"] = "Your molecules"
            user_rows.append(d)

    fda_smiles, _ = get_fda_approved_drugs()
    fda_rows = []
    for smi in fda_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            d = compute_descriptors(mol)
            d["SMILES"] = smi
            d["Group"] = "FDA-approved drugs"
            fda_rows.append(d)

    combined = pd.DataFrame(user_rows + fda_rows)
    X = combined[desc_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    combined["PC1"] = coords[:, 0]
    combined["PC2"] = coords[:, 1]

    var = pca.explained_variance_ratio_
    pc1_label = f"PC1 ({var[0] * 100:.1f}%)"
    pc2_label = f"PC2 ({var[1] * 100:.1f}%)"

    return combined, pc1_label, pc2_label


@st.cache_data
def _cached_corr_matrix(smiles_tuple, sa_scores_tuple=None, qed_scores_tuple=None):
    desc_df = _cached_descriptors(smiles_tuple).copy()
    if sa_scores_tuple is not None:
        desc_df["SA_Score"] = list(sa_scores_tuple)
    if qed_scores_tuple is not None:
        desc_df["QED"] = list(qed_scores_tuple)
    corr = desc_df.corr(numeric_only=True).round(2)
    corr_long = corr.reset_index().melt(id_vars="index")
    corr_long.columns = ["descriptor_x", "descriptor_y", "correlation"]
    return corr_long


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

uploaded_file = st.file_uploader("Upload a .txt, .csv, or .xlsx file with SMILES", type=["txt", "csv", "xlsx"])
pasted_smiles = st.text_area("Or paste SMILES here (one per line)", key="pasted_smiles")

# ── Early parse of uploaded CSV/XLSX for column detection ────────────────────
_uploaded_df = None
_uploaded_smiles_col = None
if uploaded_file is not None:
    _fname = uploaded_file.name.lower()
    if _fname.endswith(".csv"):
        _uploaded_df = pd.read_csv(uploaded_file)
    elif _fname.endswith(".xlsx"):
        _uploaded_df = pd.read_excel(uploaded_file)
    if _uploaded_df is not None:
        for _c in ["canonical_smiles", "Smiles", "SMILES", "smiles"]:
            if _c in _uploaded_df.columns:
                _uploaded_smiles_col = _c
                break
        if _uploaded_smiles_col is None:
            st.warning("Could not auto-detect a SMILES column. Please select it manually.")
            _uploaded_smiles_col = st.selectbox(
                "Which column contains the SMILES strings?",
                _uploaded_df.columns.tolist(),
                key="smiles_col_early",
            )

# ── Auto-fill pchembl_value from standard_value (ChEMBL convention) ──────────
_pchembl_fill_map = {}
_pchembl_fill_summary = None

if (
    _uploaded_df is not None
    and "pchembl_value" in _uploaded_df.columns
    and "standard_value" in _uploaded_df.columns
):
    import math as _math
    _pchembl_missing = _uploaded_df["pchembl_value"].isna()
    _n_missing = int(_pchembl_missing.sum())
    if _n_missing > 0:
        _uploaded_df = _uploaded_df.copy()
        _n_computed = 0
        for _idx in _uploaded_df[_pchembl_missing].index:
            _sv = _uploaded_df.at[_idx, "standard_value"]
            try:
                _sv_f = float(_sv)
                if _sv_f > 0:
                    _uploaded_df.at[_idx, "pchembl_value"] = round(9.0 - _math.log10(_sv_f), 3)
                    _n_computed += 1
            except (ValueError, TypeError):
                pass
        _pchembl_fill_summary = (
            f"{_n_computed} of {_n_missing} pchembl_value entries were missing "
            f"and have been computed from standard_value."
        )
    if _uploaded_smiles_col:
        for _, _row in _uploaded_df.iterrows():
            _pv = _row.get("pchembl_value")
            if pd.notna(_pv):
                _mol_tmp = Chem.MolFromSmiles(str(_row[_uploaded_smiles_col]))
                if _mol_tmp:
                    _pchembl_fill_map[Chem.MolToSmiles(_mol_tmp)] = _pv

# ── Activity Labeling (CSV/XLSX only) ────────────────────────────────────────
_activity_label_map = {}
_activity_pval_map = {}

if _uploaded_df is not None:
    with st.expander("Activity Labeling (optional)", expanded=False):

        if _pchembl_fill_summary:
            st.info(_pchembl_fill_summary)

        _numeric_cols = [
            c for c in _uploaded_df.select_dtypes(include="number").columns
            if c != _uploaded_smiles_col
        ]
        if not _numeric_cols:
            st.info("No numeric columns found in the uploaded file for activity labeling.")
        else:
            _act_keywords = ["ic50", "ki", "ec50", "kd", "activity", "potency", "standard_value"]
            _auto_act_col = next(
                (c for c in _numeric_cols if any(kw in c.lower() for kw in _act_keywords)),
                None,
            )
            _act_default_idx = _numeric_cols.index(_auto_act_col) if _auto_act_col else 0

            _la_c1, _la_c2 = st.columns(2)
            with _la_c1:
                _act_col = st.selectbox("Activity column", _numeric_cols, index=_act_default_idx)
                _act_type = st.selectbox(
                    "Activity type", ["IC50", "Ki", "EC50", "Kd"],
                    help="Select the measurement type.",
                )
            with _la_c2:
                # Auto-detect log-scale columns
                _log_keywords = ["pchembl", "pic50", "pki", "pec50", "pkd", "pactivity"]
                _auto_log = (
                    _act_col.lower().startswith("p")
                    or any(kw in _act_col.lower() for kw in _log_keywords)
                )
                _is_log_scale = st.checkbox(
                    "Values are already on −log₁₀ scale (e.g. pIC50, pKi, pchembl_value)",
                    value=_auto_log,
                    help="Check this if your column contains pIC50/pKi/pchembl values "
                         "where higher = more potent. Leave unchecked for raw nM values "
                         "where lower = more potent.",
                )

            # Threshold inputs adapt to scale
            if _is_log_scale:
                _unit_label = ""  # unitless
                _scale_note = "Values on −log₁₀ scale (higher = more potent)."
                _thr_col1, _thr_col2 = st.columns(2)
                with _thr_col1:
                    _thr_active = st.number_input(
                        "Active threshold (−log₁₀)", value=7.0, min_value=0.0,
                        step=0.5, format="%.1f",
                        help="Molecules with pActivity ≥ this value are labeled 'Active'.",
                    )
                with _thr_col2:
                    _use_3class = st.checkbox(
                        "Use three-class labeling (Active / Intermediate / Inactive)",
                        help="Molecules between the two thresholds are labeled 'Intermediate'.",
                    )
                _thr_inactive = _thr_active - 1.0  # default: one log unit below
                if _use_3class:
                    _thr_inactive = st.number_input(
                        "Inactive threshold (−log₁₀)", value=_thr_active - 1.0,
                        min_value=0.0, step=0.5, format="%.1f",
                        help="Molecules with pActivity < this value are labeled 'Inactive'.",
                    )
            else:
                _unit_label = " nM"
                _scale_note = "Values in **nanomolar (nM)** (lower = more potent)."
                _thr_col1, _thr_col2 = st.columns(2)
                with _thr_col1:
                    _thr_active = st.number_input(
                        "Active threshold (nM)", value=1000, min_value=1,
                        help="Molecules with activity ≤ this value are labeled 'Active'.",
                    )
                with _thr_col2:
                    _use_3class = st.checkbox(
                        "Use three-class labeling (Active / Intermediate / Inactive)",
                        help="Molecules between the two thresholds are labeled 'Intermediate'.",
                    )
                _thr_inactive = 10000
                if _use_3class:
                    _thr_inactive = st.number_input(
                        "Inactive threshold (nM)", value=10000, min_value=1,
                        help="Molecules above this value are labeled 'Inactive'.",
                    )

            st.write(_scale_note)

            # Threshold validation
            if _use_3class:
                if _is_log_scale and _thr_inactive >= _thr_active:
                    st.warning(
                        f"Inactive threshold ({_thr_inactive:.1f}) should be **less than** "
                        f"active threshold ({_thr_active:.1f}) on −log₁₀ scale "
                        f"(higher = more potent)."
                    )
                elif not _is_log_scale and _thr_inactive <= _thr_active:
                    st.warning(
                        f"Inactive threshold ({_thr_inactive:,} nM) should be **greater than** "
                        f"active threshold ({_thr_active:,} nM) on nM scale "
                        f"(lower = more potent)."
                    )

            # Dynamic formula expander based on selected activity type
            _p_label = f"p{_act_type}"
            if _is_log_scale:
                with st.expander(f"About {_p_label} scale", expanded=False):
                    st.markdown(
                        f"Your column appears to contain **{_p_label}** values "
                        f"(already on −log₁₀ scale).\n\n"
                        f"Higher {_p_label} = more potent compound.\n\n"
                        f"| {_p_label} | {_act_type} (nM) | Potency tier |\n"
                        "|:---:|---:|---|\n"
                        "| 5 | 10,000 nM | Weak |\n"
                        "| 6 | 1,000 nM | Moderate |\n"
                        "| 7 | 100 nM | Good |\n"
                        "| 8 | 10 nM | High |\n"
                        "| 9 | 1 nM | Very high |\n"
                    )
            else:
                with st.expander(f"About {_p_label} conversion", expanded=False):
                    st.markdown(
                        f"**{_p_label} = −log₁₀({_act_type} × 10⁻⁹) = 9 − log₁₀({_act_type} in nM)**\n\n"
                        f"Higher {_p_label} = more potent compound. "
                        f"A {_p_label} of 6 corresponds to a {_act_type} of 1,000 nM (1 μM).\n\n"
                        f"| {_p_label} | {_act_type} (nM) | Potency tier |\n"
                        "|:---:|---:|---|\n"
                        "| 5 | 10,000 nM | Weak |\n"
                        "| 6 | 1,000 nM | Moderate |\n"
                        "| 7 | 100 nM | Good |\n"
                        "| 8 | 10 nM | High |\n"
                        "| 9 | 1 nM | Very high |\n"
                    )

            # Visual threshold zone indicator
            if _is_log_scale:
                # Log scale: higher = more potent (left=Active high, right=Inactive low)
                if _use_3class:
                    st.markdown(
                        f"<div style='display:flex;border-radius:6px;overflow:hidden;font-size:0.82em;"
                        f"font-weight:600;margin:8px 0 4px 0;'>"
                        f"<div style='flex:1;background:#2ca02c22;border:1px solid #2ca02c;"
                        f"color:#1a6b1a;padding:6px 8px;text-align:center;'>"
                        f"Active<br><span style='font-weight:400'>&ge; {_thr_active:.1f}</span></div>"
                        f"<div style='flex:1;background:#ff7f0e22;border:1px solid #ff7f0e;"
                        f"color:#a85200;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Intermediate<br><span style='font-weight:400'>{_thr_inactive:.1f} – {_thr_active:.1f}</span></div>"
                        f"<div style='flex:1;background:#d6272822;border:1px solid #d62728;"
                        f"color:#8b0000;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Inactive<br><span style='font-weight:400'>&lt; {_thr_inactive:.1f}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='display:flex;border-radius:6px;overflow:hidden;font-size:0.82em;"
                        f"font-weight:600;margin:8px 0 4px 0;'>"
                        f"<div style='flex:1;background:#2ca02c22;border:1px solid #2ca02c;"
                        f"color:#1a6b1a;padding:6px 8px;text-align:center;'>"
                        f"Active<br><span style='font-weight:400'>&ge; {_thr_active:.1f}</span></div>"
                        f"<div style='flex:1;background:#d6272822;border:1px solid #d62728;"
                        f"color:#8b0000;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Inactive<br><span style='font-weight:400'>&lt; {_thr_active:.1f}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                # nM scale: lower = more potent (left=Active low, right=Inactive high)
                if _use_3class:
                    st.markdown(
                        f"<div style='display:flex;border-radius:6px;overflow:hidden;font-size:0.82em;"
                        f"font-weight:600;margin:8px 0 4px 0;'>"
                        f"<div style='flex:1;background:#2ca02c22;border:1px solid #2ca02c;"
                        f"color:#1a6b1a;padding:6px 8px;text-align:center;'>"
                        f"Active<br><span style='font-weight:400'>&le; {_thr_active:,} nM</span></div>"
                        f"<div style='flex:1;background:#ff7f0e22;border:1px solid #ff7f0e;"
                        f"color:#a85200;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Intermediate<br><span style='font-weight:400'>{_thr_active:,} – {_thr_inactive:,} nM</span></div>"
                        f"<div style='flex:1;background:#d6272822;border:1px solid #d62728;"
                        f"color:#8b0000;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Inactive<br><span style='font-weight:400'>&gt; {_thr_inactive:,} nM</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='display:flex;border-radius:6px;overflow:hidden;font-size:0.82em;"
                        f"font-weight:600;margin:8px 0 4px 0;'>"
                        f"<div style='flex:1;background:#2ca02c22;border:1px solid #2ca02c;"
                        f"color:#1a6b1a;padding:6px 8px;text-align:center;'>"
                        f"Active<br><span style='font-weight:400'>&le; {_thr_active:,} nM</span></div>"
                        f"<div style='flex:1;background:#d6272822;border:1px solid #d62728;"
                        f"color:#8b0000;padding:6px 8px;text-align:center;border-left:none;'>"
                        f"Inactive<br><span style='font-weight:400'>&gt; {_thr_active:,} nM</span></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            _labeled_df, _skipped = label_activity(
                _uploaded_df, _act_col, _act_type, _thr_active, _thr_inactive,
                _use_3class, is_log_scale=_is_log_scale,
            )

            # Build canonical SMILES → label map for downstream use
            if _uploaded_smiles_col:
                for _, _row in _labeled_df.iterrows():
                    if _row["Activity_Label"] is None:
                        continue
                    _mol_tmp = Chem.MolFromSmiles(str(_row[_uploaded_smiles_col]))
                    if _mol_tmp:
                        _csmi = Chem.MolToSmiles(_mol_tmp)
                        _activity_label_map[_csmi] = _row["Activity_Label"]
                        _activity_pval_map[_csmi] = _row["pActivity"]

            # Summary line + bar chart
            _label_counts = (
                _labeled_df["Activity_Label"].dropna().value_counts().reset_index()
            )
            _label_counts.columns = ["Label", "Count"]
            # Enforce Active → Intermediate → Inactive order for summary text
            _ordered_labels = [l for l in ["Active", "Intermediate", "Inactive"]
                               if l in _label_counts["Label"].values]
            _summary_parts = [
                f"**{_label_counts.loc[_label_counts['Label'] == l, 'Count'].iloc[0]} {l}**"
                for l in _ordered_labels
            ]
            st.write(
                ", ".join(_summary_parts)
                + (f" — {len(_skipped)} skipped (missing / zero / non-numeric)" if _skipped else "")
            )

            if not _label_counts.empty:
                _bar = (
                    alt.Chart(_label_counts)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Label:N", title=None,
                                sort=["Active", "Intermediate", "Inactive"]),
                        y=alt.Y("Count:Q", title="Molecules"),
                        color=alt.Color(
                            "Label:N",
                            scale=alt.Scale(
                                domain=["Active", "Intermediate", "Inactive"],
                                range=["#2ca02c", "#ff7f0e", "#d62728"],
                            ),
                            legend=None,
                        ),
                        tooltip=["Label:N", "Count:Q"],
                    )
                    .properties(height=160)
                )
                st.altair_chart(_bar, use_container_width=True)

                _total = _label_counts["Count"].sum()
                if (_label_counts["Count"] / _total < 0.1).any():
                    st.warning(
                        "Class imbalance detected — consider balancing techniques "
                        "(e.g. oversampling, SMOTE, or class weights) before ML training."
                    )

            if _skipped:
                with st.expander(f"View {len(_skipped)} skipped rows"):
                    st.dataframe(pd.DataFrame(_skipped))

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
    if _uploaded_df is not None:
        smiles_list = _uploaded_df[_uploaded_smiles_col].dropna().astype(str).tolist()
    elif uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        smiles_list = [line.strip() for line in file_content.splitlines() if line.strip()]
    elif pasted_smiles.strip():
        smiles_list = [line.strip() for line in pasted_smiles.splitlines() if line.strip()]

    if not smiles_list:
        st.warning("Please upload a file or paste at least one SMILES string.")
    else:
        with st.spinner("Processing molecules..."):
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
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            st.subheader("Removed Molecules")
            st.write(f"{len(result['removed_log'])} molecules removed across all steps")
            removed_df = pd.DataFrame(result["removed_log"])[["original_index", "smiles", "step", "reason"]]
            st.download_button(
                "Download removed molecules as CSV",
                data=removed_df.to_csv(index=False),
                file_name=f"removed_molecules_{_ts}.csv",
                mime="text/csv",
            )
            with st.expander("View removed molecules"):
                st.dataframe(removed_df, use_container_width=True)

        st.subheader("Kept Molecules")
        kept_data = {"SMILES": result["kept_smiles"]}
        if result["sa_scores"] is not None:
            kept_data["SA_Score"] = result["sa_scores"]
        if result["qed_scores"] is not None:
            kept_data["QED_Score"] = result["qed_scores"]
        if _activity_label_map:
            kept_data["Activity_Label"] = [_activity_label_map.get(s) for s in result["kept_smiles"]]
            kept_data["pActivity"] = [_activity_pval_map.get(s) for s in result["kept_smiles"]]
        if _pchembl_fill_map:
            kept_data["pchembl_value"] = [_pchembl_fill_map.get(s) for s in result["kept_smiles"]]
        kept_df = pd.DataFrame(kept_data)
        extra_kept = [c for c in ["SA_Score", "QED_Score", "Activity_Label", "pActivity", "pchembl_value"] if c in kept_df.columns]

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

        st.write(f"{len(kept_df)} molecules kept after preprocessing")
        st.download_button("Download kept molecules as CSV", data=csv_data, file_name=f"kept_molecules_{_ts}.csv", mime="text/csv")
        with st.expander("View kept molecules (first 20)"):
            preview_records = kept_df.head(20).to_dict("records")
            st.markdown(
                _mol_table_html(preview_records, "SMILES", extra_kept),
                unsafe_allow_html=True,
            )

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

        if len(result["kept_smiles"]) >= 5:
            st.subheader("Descriptor Correlations")
            _sa_tuple = tuple(result["sa_scores"]) if result["sa_scores"] is not None else None
            _qed_tuple = tuple(result["qed_scores"]) if result["qed_scores"] is not None else None
            corr_long = _cached_corr_matrix(tuple(result["kept_smiles"]), _sa_tuple, _qed_tuple)

            corr_rect = (
                alt.Chart(corr_long)
                .mark_rect()
                .encode(
                    x=alt.X("descriptor_x:N", title=None, sort=None),
                    y=alt.Y("descriptor_y:N", title=None, sort=None),
                    color=alt.Color(
                        "correlation:Q",
                        scale=alt.Scale(domain=[-1, 0, 1], range=["#d73027", "#ffffff", "#4575b4"]),
                        legend=alt.Legend(title="Pearson r"),
                    ),
                    tooltip=[
                        alt.Tooltip("descriptor_x:N", title="Descriptor X"),
                        alt.Tooltip("descriptor_y:N", title="Descriptor Y"),
                        alt.Tooltip("correlation:Q", title="Pearson r", format=".2f"),
                    ],
                )
            )
            corr_text = (
                alt.Chart(corr_long)
                .mark_text(fontSize=11)
                .encode(
                    x=alt.X("descriptor_x:N", sort=None),
                    y=alt.Y("descriptor_y:N", sort=None),
                    text=alt.Text("correlation:Q", format=".2f"),
                    color=alt.condition(
                        "abs(datum.correlation) > 0.6",
                        alt.value("white"),
                        alt.value("#333333"),
                    ),
                )
            )
            st.altair_chart((corr_rect + corr_text).properties(height=350), use_container_width=True)
            st.caption(
                "This heatmap shows Pearson correlations between physicochemical descriptors. "
                "Strong correlations (close to +1 or -1) indicate redundant features — consider removing "
                "one of a highly correlated pair before training ML models to reduce multicollinearity."
            )

        if len(result["kept_smiles"]) >= 5:
            st.subheader("Chemical Space Visualization")
            pca_df, pc1_label, pc2_label = _cached_chemical_space_pca(tuple(result["kept_smiles"]))
            color_scale = alt.Scale(
                domain=["Your molecules", "FDA-approved drugs"],
                range=["#1f77b4", "#BBBBBB"],
            )
            pca_chart = (
                alt.Chart(pca_df)
                .mark_circle()
                .encode(
                    x=alt.X("PC1:Q", title=pc1_label),
                    y=alt.Y("PC2:Q", title=pc2_label),
                    color=alt.Color("Group:N", scale=color_scale, legend=alt.Legend(title="Dataset")),
                    opacity=alt.condition(
                        alt.datum["Group"] == "Your molecules",
                        alt.value(0.9),
                        alt.value(0.35),
                    ),
                    size=alt.condition(
                        alt.datum["Group"] == "Your molecules",
                        alt.value(80),
                        alt.value(40),
                    ),
                    tooltip=[
                        alt.Tooltip("SMILES:N"),
                        alt.Tooltip("MW:Q", format=".1f"),
                        alt.Tooltip("LogP:Q", format=".2f"),
                        alt.Tooltip("TPSA:Q", format=".1f"),
                        alt.Tooltip("HBD:Q"),
                        alt.Tooltip("HBA:Q"),
                        alt.Tooltip("Group:N"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(pca_chart, use_container_width=True)
            st.caption(
                "This plot reduces your molecules' physicochemical properties to 2 dimensions using PCA. "
                "Molecules closer together have more similar properties. "
                "The grey dots represent FDA-approved drugs as a reference for 'drug-like' chemical space."
            )


st.divider()

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
        with st.spinner("Computing fingerprints and descriptors..."):
            feature_df, feat_errors = _cached_featurize(
                tuple(st.session_state["kept_smiles_for_featurization"]),
                fp_type=fp_type,
                radius=radius,
                n_bits=n_bits,
            )
        _feat_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            file_name=f"featurized_data_{_feat_ts}.csv",
            mime="text/csv",
        )

st.divider()

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
