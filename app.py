import streamlit as st

st.set_page_config(page_title="QSAR Preprocessing Tool", page_icon="🧪", layout="wide")

import sys
import pandas as pd
from datetime import datetime
from rdkit import rdBase, Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, RDConfig
from rdkit.Chem.inchi import MolToInchi, MolFromInchi, InchiToInchiKey
sys.path.append(RDConfig.RDContribDir + "/SA_Score")
import sascorer
from pipeline.preprocessing import (
    run_preprocessing_pipeline, label_activity, clean_dataset,
    check_lipinski, check_veber, check_ghose, check_egan, check_muegge,
    check_pains, check_brenk, compute_qed, analyze_scaffolds,
)
from pipeline.featurization import featurize_dataset, compute_descriptors, compute_fingerprint, compute_esol
import altair as alt
from pipeline.visualization import mol_to_base64_png, mol_to_image, plot_boiled_egg, plot_radar_chart, plot_mini_radar
from pipeline.methodology import generate_methods_text
from pipeline.pains_catalog import get_pains_explanation, get_brenk_explanation
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
def _cached_preview_descriptors(smiles_tuple):
    """Compute descriptors + FractionCsp3 for preview molecule cards."""
    results = []
    for smi in smiles_tuple:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            d = compute_descriptors(mol)
            d["FractionCsp3"] = rdMolDescriptors.CalcFractionCSP3(mol)
            d["SMILES"] = smi
            results.append(d)
    return results


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

st.markdown(
    """<style>
    /* ── Nav card buttons: scoped to nav_cards container only ── */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_cards"])
        [data-testid="stButton"] > button {
        min-height: 140px !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: 2px solid #e0e3e8 !important;
        background: white !important;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        transition: all 0.2s ease !important;
        white-space: pre-wrap !important;
        line-height: 1.6 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_cards"])
        [data-testid="stButton"] > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12) !important;
        border-color: #ff4b4b !important;
    }
    /* Active nav card */
    .nav-active [data-testid="stButton"] > button {
        background: #fafbff !important;
        border-bottom: 4px solid #ff4b4b !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("QSAR Preprocessing Tool")
st.write("Welcome! This tool helps preprocess and featurize molecules for QSAR/virtual screening workflows.")

# ── Navigation cards ──
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "preprocessing"

_NAV_CARDS = [
    ("preprocessing", "🔬", "Preprocessing",     "Clean, filter & featurize"),
    ("explorer",      "🧬", "Molecule Explorer",  "Analyze a single molecule"),
    ("comparison",    "⚖️", "Filter Comparison",  "Compare settings side by side"),
    ("converter",     "🔄", "Molecule Converter",  "SMILES, InChI & more"),
]

with st.container(key="nav_cards"):
    nav_cols = st.columns(4)
    for col, (key, icon, title, subtitle) in zip(nav_cols, _NAV_CARDS):
        is_active = st.session_state["active_tab"] == key
        with col:
            if is_active:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            if st.button(
                f"{icon} {title}",
                key=f"nav_{key}",
                use_container_width=True,
            ):
                if not is_active:
                    st.session_state["active_tab"] = key
                    st.rerun()
            st.caption(subtitle)
            if is_active:
                st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "preprocessing":

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

    # ── Data Quality Summary (CSV/XLSX only) ──────────────────────────────────────
    _pchembl_fill_map = {}


    def _build_missing_data_html(df, smiles_col):
        """Build an HTML table showing missing-data stats for key columns."""
        _key_cols = [c for c in [smiles_col, "standard_value", "pchembl_value", "value"]
                     if c and c in df.columns]
        if not _key_cols:
            return None
        _rows = []
        for _col in _key_cols:
            _total = len(df)
            _miss = int(df[_col].isna().sum())
            _pct = (_miss / _total * 100) if _total else 0
            _complete_pct = 100 - _pct
            _bar = (
                f"<div style='background:#e0e0e0;border-radius:4px;height:14px;"
                f"width:120px;display:inline-block;overflow:hidden'>"
                f"<div style='background:#2ca02c;height:100%;"
                f"width:{_complete_pct:.0f}%'></div></div>"
            )
            _rows.append({
                "Column": _col, "Total": _total, "Missing": _miss,
                "Missing %": f"{_pct:.1f}%", "Completeness": _bar,
            })
        _headers = ["Column", "Total", "Missing", "Missing %", "Completeness"]
        _html = (
            "<table style='border-collapse:collapse;width:100%'>"
            "<thead><tr style='background:#f5f5f5'>"
            + "".join(
                f"<th style='padding:6px 10px;text-align:left;"
                f"border-bottom:1px solid #ddd'>{h}</th>" for h in _headers
            )
            + "</tr></thead><tbody>"
        )
        for _r in _rows:
            _html += (
                "<tr style='border-bottom:1px solid #eee'>"
                + "".join(f"<td style='padding:6px 10px'>{_r[h]}</td>" for h in _headers)
                + "</tr>"
            )
        _html += "</tbody></table>"
        return _html


    if _uploaded_df is not None:
        with st.expander("Data Quality Summary", expanded=True):
            _n_rows, _n_cols = _uploaded_df.shape
            _numeric_cols_all = list(
                _uploaded_df.select_dtypes(include="number").columns
            )

            # ── Column overview ──
            st.subheader("Column overview")
            _ov_c1, _ov_c2, _ov_c3 = st.columns(3)
            _ov_c1.metric("Rows", f"{_n_rows:,}")
            _ov_c2.metric("Columns", f"{_n_cols:,}")
            _ov_c3.metric("SMILES column", _uploaded_smiles_col or "—")
            if _numeric_cols_all:
                st.write(
                    "**Numeric columns:** "
                    + ", ".join(f"`{c}`" for c in _numeric_cols_all)
                )
            else:
                st.write("No numeric columns detected.")

            st.divider()

            # ── Missing data table (before cleaning) ──
            st.subheader("Missing data")
            _miss_html = _build_missing_data_html(_uploaded_df, _uploaded_smiles_col)
            if _miss_html:
                st.markdown(_miss_html, unsafe_allow_html=True)
            else:
                st.info("No key columns found to display missing-data statistics.")

            st.divider()

            # ── Clean Dataset ──
            st.subheader("Clean Dataset")
            st.write(
                "Remove rows with missing or invalid SMILES, and cross-fill "
                "`pchembl_value` / `standard_value` where possible."
            )

            _cleaning_key = "cleaning_report"
            _cleaned_key = "cleaned_df"

            if st.button("Clean Dataset", key="btn_clean_dataset", type="primary"):
                _cleaned, _report = clean_dataset(
                    _uploaded_df, _uploaded_smiles_col,
                )
                st.session_state[_cleaning_key] = _report
                st.session_state[_cleaned_key] = _cleaned

            if _cleaning_key in st.session_state:
                _report = st.session_state[_cleaning_key]
                _uploaded_df = st.session_state[_cleaned_key]

                # Summary metrics
                _m1, _m2, _m3 = st.columns(3)
                _m1.metric("Rows before", f"{_report['rows_before']:,}")
                _m2.metric(
                    "Rows removed",
                    f"{_report['missing_smiles_removed'] + _report['invalid_smiles_removed']:,}",
                )
                _m3.metric("Rows after", f"{_report['rows_after']:,}")

                # Detailed results
                _details = []
                if _report["missing_smiles_removed"]:
                    _details.append(
                        f"Rows removed (missing/empty SMILES): **{_report['missing_smiles_removed']}**"
                    )
                if _report["invalid_smiles_removed"]:
                    _details.append(
                        f"Rows removed (unparseable SMILES): **{_report['invalid_smiles_removed']}**"
                    )
                if _report["pchembl_recomputed"]:
                    _details.append(
                        f"pchembl_value recomputed from standard_value: "
                        f"**{_report['pchembl_recomputed']}** "
                        f"(`pchembl = 9 - log`_`10`_`(standard_value)`)"
                    )
                if _report["stdval_computed"]:
                    _details.append(
                        f"standard_value computed from pchembl_value: "
                        f"**{_report['stdval_computed']}** "
                        f"(`standard_value = 10^(9 - pchembl)`)"
                    )
                if _report["inconsistent_count"]:
                    _details.append(
                        f"Inconsistent value pairs flagged: "
                        f"**{_report['inconsistent_count']}** "
                        f"(|pchembl - expected| > 0.1)"
                    )
                if _report["both_missing_count"]:
                    _details.append(
                        f"Rows with both values missing: **{_report['both_missing_count']}**"
                    )
                if not _details:
                    _details.append("No issues found — dataset is already clean.")

                for _d in _details:
                    st.markdown(f"- {_d}")

                # Expandable detail sections
                if _report["invalid_smiles_removed"]:
                    with st.expander(
                        f"View {_report['invalid_smiles_removed']} removed invalid SMILES"
                    ):
                        st.dataframe(
                            pd.DataFrame(_report["invalid_smiles_rows"]),
                            hide_index=True,
                        )
                if _report["inconsistent_count"]:
                    with st.expander(
                        f"View {_report['inconsistent_count']} inconsistent value pairs"
                    ):
                        st.dataframe(
                            pd.DataFrame(_report["inconsistent_rows"]),
                            hide_index=True,
                        )

                # Updated missing data table (after cleaning)
                st.divider()
                st.subheader("Missing data (after cleaning)")
                _miss_html_after = _build_missing_data_html(
                    _uploaded_df, _uploaded_smiles_col
                )
                if _miss_html_after:
                    st.markdown(_miss_html_after, unsafe_allow_html=True)
                else:
                    st.info("No key columns found.")

                st.success("Cleaned dataset is now used for all downstream steps.")

            # Build canonical SMILES -> pchembl map for downstream use
            if "pchembl_value" in _uploaded_df.columns and _uploaded_smiles_col:
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
            # Store serializable pipeline results for persistent display
            st.session_state["pipeline_result"] = {
                "kept_smiles": result["kept_smiles"],
                "audit_trail": result["audit_trail"],
                "removed_log": result["removed_log"],
                "sa_scores": result["sa_scores"],
                "qed_scores": result["qed_scores"],
                "input_count": len(smiles_list),
            }
            st.session_state["pipeline_ts"] = _ts
            # Snapshot label maps at pipeline run time
            st.session_state["pipeline_label_map"] = dict(_activity_label_map)
            st.session_state["pipeline_pval_map"] = dict(_activity_pval_map)
            st.session_state["pipeline_pchembl_map"] = dict(_pchembl_fill_map)

    # ── Display pipeline results (persistent) ────────────────────────────────────
    if "pipeline_result" in st.session_state:
        result = st.session_state["pipeline_result"]
        _ts = st.session_state["pipeline_ts"]
        _pl_label_map = st.session_state.get("pipeline_label_map", {})
        _pl_pval_map = st.session_state.get("pipeline_pval_map", {})
        _pl_pchembl_map = st.session_state.get("pipeline_pchembl_map", {})

        st.subheader("Results")
        st.write(f"Input molecules: {result['input_count']}")
        st.write(f"Kept after preprocessing: {len(result['kept_smiles'])}")

        st.subheader("Audit Trail")
        audit_df = pd.DataFrame(result["audit_trail"])
        st.dataframe(audit_df)

        # Build kept DataFrame
        kept_data = {"SMILES": result["kept_smiles"]}
        if result["sa_scores"] is not None:
            kept_data["SA_Score"] = result["sa_scores"]
        if result["qed_scores"] is not None:
            kept_data["QED_Score"] = result["qed_scores"]
        if _pl_label_map:
            kept_data["Activity_Label"] = [_pl_label_map.get(s) for s in result["kept_smiles"]]
            kept_data["pActivity"] = [_pl_pval_map.get(s) for s in result["kept_smiles"]]
        if _pl_pchembl_map:
            kept_data["pchembl_value"] = [_pl_pchembl_map.get(s) for s in result["kept_smiles"]]
        # Compute ESOL LogS for kept molecules
        _esol_logs_list = []
        for _smi in result["kept_smiles"]:
            _mol_tmp = Chem.MolFromSmiles(_smi)
            if _mol_tmp:
                _ls, _, _, _, _err = compute_esol(_mol_tmp)
                _esol_logs_list.append(round(_ls, 2) if _ls is not None else None)
            else:
                _esol_logs_list.append(None)
        kept_data["ESOL_LogS"] = _esol_logs_list

        kept_df = pd.DataFrame(kept_data)
        extra_kept = [c for c in ["ESOL_LogS", "SA_Score", "QED_Score", "Activity_Label", "pActivity", "pchembl_value"] if c in kept_df.columns]

        _pl_settings = st.session_state.get("pipeline_settings", {})
        metadata = build_metadata_block({
            "Lipinski max violations": _pl_settings.get("lipinski_max_violations", ""),
            "Veber filter": _pl_settings.get("enable_veber", ""),
            "Ghose filter": _pl_settings.get("enable_ghose", ""),
            "Egan filter": _pl_settings.get("enable_egan", ""),
            "Muegge filter": _pl_settings.get("enable_muegge", ""),
            "Brenk filter": _pl_settings.get("enable_brenk", ""),
            "SA score reported": _pl_settings.get("enable_sa_score", ""),
            "QED score reported": _pl_settings.get("enable_qed", ""),
            "Input molecule count": result["input_count"],
            "Kept molecule count": len(result["kept_smiles"]),
        })
        csv_data = metadata + kept_df.to_csv(index=False)

        # Side-by-side: Kept (left, wider) | Removed (right, narrower)
        _mol_left, _mol_right = st.columns([2, 1])
        with _mol_left:
            st.subheader("Kept Molecules")
            st.write(f"{len(kept_df)} molecules kept after preprocessing")

            # Activity class breakdown (inside kept column)
            if _pl_label_map:
                _kept_labels = [_pl_label_map.get(s) for s in result["kept_smiles"]]
                _class_names = ["Active", "Intermediate", "Inactive"]
                _active_classes = [c for c in _class_names if c in _kept_labels]
                if _active_classes:
                    _kept_metric_cols = st.columns(len(_active_classes))
                    for _ci, _cls in enumerate(_active_classes):
                        _kept_metric_cols[_ci].metric(_cls, _kept_labels.count(_cls))
                    _n_unlabeled = sum(1 for l in _kept_labels if l is None)
                    if _n_unlabeled:
                        st.caption(f"{_n_unlabeled} molecules had no activity label.")

            st.download_button(
                "Download kept molecules as CSV",
                data=csv_data,
                file_name=f"kept_molecules_{_ts}.csv",
                mime="text/csv",
            )
            with st.expander("View kept molecules (first 20)"):
                _preview_smiles = list(kept_df["SMILES"].head(20))
                _preview_descs = _cached_preview_descriptors(tuple(_preview_smiles))
                _preview_labels = {s: _pl_label_map.get(s) for s in _preview_smiles} if _pl_label_map else {}
                for _card_i in range(0, len(_preview_descs), 2):
                    _card_cols = st.columns(2)
                    for _ci, _col in enumerate(_card_cols):
                        _idx = _card_i + _ci
                        if _idx >= len(_preview_descs):
                            break
                        _desc = _preview_descs[_idx]
                        _smi = _desc["SMILES"]
                        with _col:
                            _trunc_smi = _smi[:40] + ("..." if len(_smi) > 40 else "")
                            st.code(_trunc_smi, language=None)
                            _img_col, _radar_col = st.columns([1, 1])
                            with _img_col:
                                _card_mol = Chem.MolFromSmiles(_smi)
                                if _card_mol:
                                    st.image(mol_to_image(_card_mol, size=(150, 150)), width=150)
                            with _radar_col:
                                _mini_fig = plot_mini_radar(_desc)
                                st.plotly_chart(_mini_fig, use_container_width=True, config={"displayModeBar": False}, key=f"mini_radar_{_idx}")
                            st.caption(
                                f"MW={_desc['MW']:.0f} | LogP={_desc['LogP']:.2f} | TPSA={_desc['TPSA']:.0f}"
                            )
                            _label = _preview_labels.get(_smi)
                            if _label:
                                _badge_color = {"Active": "green", "Intermediate": "orange", "Inactive": "red"}.get(_label, "gray")
                                st.markdown(
                                    f"<span style='background:{_badge_color};color:white;padding:2px 8px;border-radius:4px;font-size:12px'>{_label}</span>",
                                    unsafe_allow_html=True,
                                )
                    if _card_i + 2 < len(_preview_descs):
                        st.divider()

        with _mol_right:
            st.subheader("Removed Molecules")
            if result["removed_log"]:
                st.write(f"{len(result['removed_log'])} molecules removed")
                removed_df = pd.DataFrame(result["removed_log"])[
                    ["original_index", "smiles", "step", "reason"]
                ]
                # Add explanation column for PAINS/Brenk removals
                def _removal_explanation(row):
                    if row["step"] == "check_pains":
                        _pat = row["reason"].replace("PAINS match: ", "", 1)
                        return get_pains_explanation(_pat)["description"]
                    if row["step"] == "check_brenk":
                        _pat = row["reason"].replace("Brenk match: ", "", 1)
                        return get_brenk_explanation(_pat)["description"]
                    return ""
                removed_df["explanation"] = removed_df.apply(_removal_explanation, axis=1)
                _has_explanations = removed_df["explanation"].any()
                if _has_explanations:
                    st.dataframe(
                        removed_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "explanation": st.column_config.TextColumn(
                                "Why?", help="Why this pattern is problematic", width="large",
                            ),
                        },
                    )
                st.download_button(
                    "Download removed molecules as CSV",
                    data=removed_df.drop(columns=["explanation"]).to_csv(index=False),
                    file_name=f"removed_molecules_{_ts}.csv",
                    mime="text/csv",
                )
            else:
                st.write("No molecules were removed.")

        st.subheader("Descriptor Distributions")
        if len(result["kept_smiles"]) < 2:
            st.info("At least 2 kept molecules are needed to show distributions.")
        else:
            desc_df = _cached_descriptors(tuple(result["kept_smiles"]))
            if result["qed_scores"] is not None:
                desc_df["QED"] = result["qed_scores"]

            # Add activity labels to descriptor df if available
            _dist_labels = [_pl_label_map.get(s) for s in result["kept_smiles"]]
            _dist_has_labels = _pl_label_map and any(l is not None for l in _dist_labels)
            if _dist_has_labels:
                desc_df["Activity_Class"] = _dist_labels
                _dist_classes = [c for c in ["Active", "Intermediate", "Inactive"]
                                 if c in _dist_labels]
                _dist_color_scale = alt.Scale(
                    domain=_dist_classes,
                    range=[{"Active": "#2ca02c", "Intermediate": "#ff7f0e",
                            "Inactive": "#d62728"}[c] for c in _dist_classes],
                )

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
                    if _dist_has_labels:
                        _desc_filtered = desc_df[desc_df["Activity_Class"].notna()]
                        chart = (
                            alt.Chart(_desc_filtered)
                            .mark_bar(opacity=0.6)
                            .encode(
                                alt.X(f"{field}:Q", bin=alt.Bin(maxbins=15), title=label),
                                alt.Y("count()", title="Count", stack=None),
                                color=alt.Color(
                                    "Activity_Class:N",
                                    scale=_dist_color_scale,
                                    legend=alt.Legend(title="Class"),
                                ),
                                tooltip=[
                                    alt.Tooltip(f"{field}:Q", bin=True, title=label),
                                    alt.Tooltip("count()", title="Count"),
                                    alt.Tooltip("Activity_Class:N", title="Class"),
                                ],
                            )
                            .properties(height=180)
                        )
                    else:
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

            # Assign activity-class-aware groups when labels exist
            _pca_labels = [_pl_label_map.get(s) for s in result["kept_smiles"]]
            _pca_has_labels = _pl_label_map and any(l is not None for l in _pca_labels)

            if _pca_has_labels:
                # Replace generic "Your molecules" with specific class names
                _pca_group_map = {}
                for smi, lbl in zip(result["kept_smiles"], _pca_labels):
                    _pca_group_map[smi] = lbl if lbl else "Unlabeled"
                pca_df = pca_df.copy()
                pca_df["Group"] = pca_df.apply(
                    lambda r: _pca_group_map.get(r["SMILES"], r["Group"])
                    if r["Group"] == "Your molecules" else r["Group"],
                    axis=1,
                )
                _pca_classes = [c for c in ["Active", "Intermediate", "Inactive"]
                                if c in pca_df["Group"].values]
                _pca_domain = _pca_classes + ["FDA-approved drugs"]
                _class_colors = {"Active": "#2ca02c", "Intermediate": "#ff7f0e",
                                 "Inactive": "#d62728"}
                _pca_range = [_class_colors[c] for c in _pca_classes] + ["#9467bd"]
                # FDA dots should be behind user molecules
                _pca_order = ["FDA-approved drugs"] + _pca_classes
            else:
                _pca_domain = ["Your molecules", "FDA-approved drugs"]
                _pca_range = ["#1f77b4", "#BBBBBB"]
                _pca_order = ["FDA-approved drugs", "Your molecules"]

            color_scale = alt.Scale(domain=_pca_domain, range=_pca_range)

            pca_chart = (
                alt.Chart(pca_df)
                .mark_circle()
                .encode(
                    x=alt.X("PC1:Q", title=pc1_label),
                    y=alt.Y("PC2:Q", title=pc2_label),
                    color=alt.Color("Group:N", scale=color_scale, legend=alt.Legend(title="Dataset")),
                    opacity=alt.condition(
                        alt.datum["Group"] == "FDA-approved drugs",
                        alt.value(0.35),
                        alt.value(0.9),
                    ),
                    size=alt.condition(
                        alt.datum["Group"] == "FDA-approved drugs",
                        alt.value(40),
                        alt.value(80),
                    ),
                    order=alt.Order("_sort:Q"),
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
                .transform_calculate(
                    _sort="indexof(" + str(_pca_order) + ", datum.Group)"
                )
                .properties(height=420)
            )
            st.altair_chart(pca_chart, use_container_width=True)
            if _pca_has_labels:
                st.caption(
                    "This plot reduces your molecules' physicochemical properties to 2 dimensions using PCA. "
                    "Molecules are colored by activity class. "
                    "The purple dots represent FDA-approved drugs as a reference for 'drug-like' chemical space."
                )
            else:
                st.caption(
                    "This plot reduces your molecules' physicochemical properties to 2 dimensions using PCA. "
                    "Molecules closer together have more similar properties. "
                    "The grey dots represent FDA-approved drugs as a reference for 'drug-like' chemical space."
                )


    if "pipeline_result" in st.session_state and len(st.session_state["pipeline_result"]["kept_smiles"]) >= 5:
        result = st.session_state["pipeline_result"]
        _pl_label_map = st.session_state.get("pipeline_label_map", {})
        st.divider()
        st.header("\U0001F3D7\uFE0F Scaffold Analysis")
        st.caption(
            "Murcko scaffold analysis identifies the core ring systems shared across molecules. "
            "Scaffolds appearing frequently define the dominant chemical series in your dataset "
            "\u2014 important for understanding structural diversity and avoiding scaffold bias in ML models."
        )

        @st.cache_data
        def _cached_scaffolds(smiles_tuple):
            mols = [Chem.MolFromSmiles(s) for s in smiles_tuple]
            return analyze_scaffolds(mols)

        _scaf = _cached_scaffolds(tuple(result["kept_smiles"]))
        _n_mols = len(result["kept_smiles"])

        # a) Summary metrics
        _sc1, _sc2, _sc3 = st.columns([1, 1, 1])
        _sc1.metric("Unique Scaffolds", _scaf["unique_scaffold_count"])
        _top_smi, _top_cnt = list(_scaf["scaffold_counts"].items())[0]
        _sc2.metric("Most Common Scaffold", f"{_top_cnt}x",
                    help=_top_smi[:50])
        _sc3.metric("Singleton Scaffolds", _scaf["singleton_count"])

        # b) Scaffold frequency chart
        _scaf_labels = [_pl_label_map.get(s) for s in result["kept_smiles"]]
        _scaf_has_labels = _pl_label_map and any(l is not None for l in _scaf_labels)

        _bar_data = []
        for smi, count in list(_scaf["scaffold_counts"].items())[:15]:
            _display = smi if len(smi) <= 30 else smi[:27] + "..."
            if _scaf_has_labels:
                _idxs = [i for i, s in enumerate(_scaf["scaffold_smiles"]) if s == smi]
                for idx in _idxs:
                    lbl = _scaf_labels[idx] if _scaf_labels[idx] else "Unlabeled"
                    _bar_data.append({"Scaffold": _display, "Count": 1, "Activity": lbl})
            else:
                _bar_data.append({"Scaffold": _display, "Count": count})

        _bar_df = pd.DataFrame(_bar_data)
        if _scaf_has_labels:
            _act_classes = [c for c in ["Active", "Intermediate", "Inactive", "Unlabeled"]
                           if c in _bar_df["Activity"].values]
            _act_colors = {"Active": "#2ca02c", "Intermediate": "#ff7f0e",
                           "Inactive": "#d62728", "Unlabeled": "#999999"}
            _bar_chart = (
                alt.Chart(_bar_df)
                .mark_bar()
                .encode(
                    x=alt.X("sum(Count):Q", title="Count"),
                    y=alt.Y("Scaffold:N", sort="-x", title=""),
                    color=alt.Color("Activity:N",
                                    scale=alt.Scale(domain=_act_classes,
                                                    range=[_act_colors[c] for c in _act_classes])),
                )
                .properties(height=min(len(list(_scaf["scaffold_counts"].items())[:15]) * 28, 420))
            )
        else:
            _bar_chart = (
                alt.Chart(_bar_df)
                .mark_bar(color="#4C72B0")
                .encode(
                    x=alt.X("Count:Q", title="Count"),
                    y=alt.Y("Scaffold:N", sort="-x", title=""),
                )
                .properties(height=min(len(list(_scaf["scaffold_counts"].items())[:15]) * 28, 420))
            )
        st.subheader("Scaffold Frequency")
        st.altair_chart(_bar_chart, use_container_width=True)

        # c) Top scaffolds table
        with st.expander("Top 10 scaffolds (click to expand)", expanded=False):
            for _rank, (_ts_smi, _ts_cnt, _ts_idxs) in enumerate(_scaf["top_scaffolds"], 1):
                _ts_pct = _ts_cnt / _n_mols * 100
                _ts_left, _ts_right = st.columns([1, 2])
                with _ts_left:
                    if _ts_smi != "No scaffold":
                        _ts_mol = Chem.MolFromSmiles(_ts_smi)
                        if _ts_mol:
                            st.image(mol_to_image(_ts_mol, size=(200, 200)), width=150)
                with _ts_right:
                    st.markdown(f"**#{_rank}** — {_ts_cnt} molecules ({_ts_pct:.1f}%)")
                    st.code(_ts_smi, language=None)
                    if _scaf_has_labels:
                        _act_breakdown = {}
                        for idx in _ts_idxs:
                            lbl = _scaf_labels[idx] if _scaf_labels[idx] else "Unlabeled"
                            _act_breakdown[lbl] = _act_breakdown.get(lbl, 0) + 1
                        st.write(", ".join(f"{cnt} {lbl}" for lbl, cnt in _act_breakdown.items()))
                if _rank < len(_scaf["top_scaffolds"]):
                    st.divider()

        # d) Scaffold diversity
        st.subheader("Scaffold Diversity")
        _diversity = _scaf["unique_scaffold_count"] / _n_mols if _n_mols > 0 else 0
        st.metric("Diversity Score", f"{_diversity:.2f}")
        st.progress(min(_diversity, 1.0))
        st.caption(
            "High scaffold diversity (>0.5) indicates a structurally diverse dataset. "
            "Low diversity suggests the dataset is dominated by a few chemical series."
        )

    st.divider()

    st.header("Featurization")
    st.write("After preprocessing, generate descriptors and fingerprints for your kept molecules.")

    fp_type = st.selectbox(
        "Fingerprint type",
        ["morgan", "fcfp", "maccs", "topological", "atom_pair", "torsion", "avalon", "pattern", "layered"],
        help="Morgan (ECFP) is the most common default for machine learning. "
             "FCFP: Like Morgan but uses pharmacophoric features (H-bond donor/acceptor, aromatic, etc.) instead of atom types. "
             "MACCS is faster but less detailed (166 fixed structural keys). "
             "Pattern: Substructure pattern-based fingerprint. "
             "Layered: Encodes different molecular features in separate layers. "
             "Others are specialized for specific use cases.",
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

    if fp_type in ("morgan", "fcfp"):
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
            st.session_state["featurization_result"] = {
                "feature_df": feature_df,
                "feat_errors": feat_errors,
                "fp_type": fp_type,
                "n_bits": n_bits,
                "radius": radius,
                "ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }

    # ── Display featurization results (persistent) ───────────────────────────────
    if "featurization_result" in st.session_state:
        _fr = st.session_state["featurization_result"]
        feature_df = _fr["feature_df"]
        feat_errors = _fr["feat_errors"]
        _feat_ts = _fr["ts"]

        st.subheader("Featurized Data")
        st.write(f"Shape: {feature_df.shape}")
        st.dataframe(feature_df.head(20))

        if feat_errors:
            st.warning(f"{len(feat_errors)} molecules failed featurization.")
            st.dataframe(pd.DataFrame(feat_errors))

        feat_metadata = build_metadata_block({
            "Fingerprint type": _fr["fp_type"],
            "Fingerprint bits": _fr["n_bits"],
            "Morgan radius": _fr["radius"],
            "Molecule count": feature_df.shape[0],
            "Feature column count": feature_df.shape[1],
        })
        csv_data = feat_metadata + feature_df.to_csv(index=False)
        st.download_button(
            "Download all featurized data as CSV",
            data=csv_data,
            file_name=f"featurized_data_{_feat_ts}.csv",
            mime="text/csv",
        )

        # Activity-aware downloads and preview
        _pl_label_map_feat = st.session_state.get("pipeline_label_map", {})
        _feat_smiles = list(st.session_state.get("kept_smiles_for_featurization", []))
        _feat_labels = [_pl_label_map_feat.get(s) for s in _feat_smiles]
        _has_feat_labels = any(l is not None for l in _feat_labels)

        if _has_feat_labels:
            _feat_df_with_label = feature_df.copy()
            _feat_df_with_label.insert(0, "_Activity_Label", _feat_labels)

            _class_order = ["Active", "Intermediate", "Inactive"]
            _present_classes = [c for c in _class_order if c in _feat_labels]

            st.write("**Download by activity class:**")
            _dl_cols = st.columns(len(_present_classes))
            for _di, _cls in enumerate(_present_classes):
                _subset = _feat_df_with_label[
                    _feat_df_with_label["_Activity_Label"] == _cls
                ].drop(columns=["_Activity_Label"])
                _dl_cols[_di].download_button(
                    f"Download {_cls} ({len(_subset)})",
                    data=_subset.to_csv(index=False),
                    file_name=f"featurized_{_cls.lower()}_{_feat_ts}.csv",
                    mime="text/csv",
                    key=f"dl_feat_{_cls}",
                )

            with st.expander("View by activity class"):
                _tab_names = ["All"] + _present_classes
                _tabs = st.tabs(_tab_names)
                with _tabs[0]:
                    st.write(f"{len(feature_df)} molecules")
                    st.dataframe(feature_df.head(10))
                for _ti, _cls in enumerate(_present_classes):
                    with _tabs[_ti + 1]:
                        _sub = _feat_df_with_label[
                            _feat_df_with_label["_Activity_Label"] == _cls
                        ].drop(columns=["_Activity_Label"])
                        st.write(f"{len(_sub)} molecules")
                        st.dataframe(_sub.head(10))

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
            if settings.get("kept_count", 0) >= 2:
                settings["enable_boiled_egg"] = True
            st.session_state["methods_text"] = generate_methods_text(settings)

    if "methods_text" in st.session_state:
        st.text_area(
            "Methods paragraph (select all and copy)",
            value=st.session_state["methods_text"],
            height=250,
        )
        st.caption("Tip: click inside the text box, then press Ctrl+A / Cmd+A to select all.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Molecule Explorer
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "explorer":
    st.header("Molecule Explorer")
    st.write("Enter one or more SMILES strings to see a full physicochemical and druglikeness profile. "
             "Optionally add a name after each SMILES, separated by a space or tab.")

    _EXPLORER_PICKS = {
        "Aspirin":    "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine":   "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Ibuprofen":  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Metformin":  "CN(C)C(=N)NC(=N)N",
    }

    _exp_cols = st.columns(len(_EXPLORER_PICKS))
    for _exp_col, (_exp_name, _exp_smi) in zip(_exp_cols, _EXPLORER_PICKS.items()):
        if _exp_col.button(_exp_name, use_container_width=True, key=f"explorer_pick_{_exp_name}"):
            st.session_state["explorer_smiles"] = f"{_exp_smi} {_exp_name}"
            st.rerun()

    _explorer_input = st.text_area(
        "SMILES (one per line, optionally followed by a name)",
        placeholder="CC(=O)Oc1ccccc1C(=O)O Aspirin\nCn1cnc2c1c(=O)n(C)c(=O)n2C Caffeine\nc1ccc(O)c(O)c1",
        key="explorer_smiles",
        height=120,
    )

    _explorer_analyze = st.button("Analyze", type="primary", key="explorer_analyze")

    # Parse multi-molecule input
    _explorer_molecules = []  # list of (smiles, name, mol)
    if (_explorer_analyze or _explorer_input) and _explorer_input and _explorer_input.strip():
        _parse_errors = []
        for _line_i, _line in enumerate(_explorer_input.strip().splitlines()):
            _line = _line.strip()
            if not _line:
                continue
            _parts = _line.split(None, 1)
            _smi = _parts[0]
            _name = _parts[1] if len(_parts) > 1 else f"Molecule {_line_i + 1}"
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is None:
                _parse_errors.append(f"Could not parse SMILES: `{_smi}`")
            else:
                _explorer_molecules.append((_smi, _name, _mol))
        for _err in _parse_errors:
            st.error(_err)

    if _explorer_molecules:
        st.session_state["explorer_molecules"] = _explorer_molecules

        # Molecule selector
        _exp_names = [f"{name} ({smi[:30]}{'...' if len(smi) > 30 else ''})"
                      for smi, name, _ in _explorer_molecules]
        if len(_explorer_molecules) > 1:
            _exp_sel_idx = st.selectbox(
                f"Showing molecule 1 of {len(_explorer_molecules)} — select to switch:",
                range(len(_explorer_molecules)),
                format_func=lambda i: _exp_names[i],
                key="explorer_mol_idx",
            )
        else:
            _exp_sel_idx = 0

        _exp_smi_sel, _exp_name_sel, _exp_mol = _explorer_molecules[_exp_sel_idx]

        if len(_explorer_molecules) > 1:
            st.caption(f"Viewing: **{_explorer_molecules[_exp_sel_idx][1]}**")

        if _exp_mol is not None:
            # Compute descriptors early (needed for radar chart)
            _exp_mw = Descriptors.MolWt(_exp_mol)
            _exp_logp = Descriptors.MolLogP(_exp_mol)
            _exp_tpsa = Descriptors.TPSA(_exp_mol)
            _exp_hbd = Descriptors.NumHDonors(_exp_mol)
            _exp_hba = Descriptors.NumHAcceptors(_exp_mol)
            _exp_rotb = Descriptors.NumRotatableBonds(_exp_mol)
            _exp_arom = Descriptors.NumAromaticRings(_exp_mol)
            _exp_fcsp3 = rdMolDescriptors.CalcFractionCSP3(_exp_mol)
            _exp_heavy = _exp_mol.GetNumHeavyAtoms()
            _exp_rings = _exp_mol.GetRingInfo().NumRings()
            # Structure & Radar Chart
            st.divider()
            st.subheader("Structure & Identifiers")

            _exp_canonical = Chem.MolToSmiles(_exp_mol)
            _exp_insatu = 1.0 - _exp_fcsp3
            _exp_insolu_val = (_exp_logp + 2.0) / 7.0

            st.markdown(
                '<div style="border:1px solid #e0e0e0; border-radius:10px; padding:16px; background:#fff">',
                unsafe_allow_html=True,
            )
            _exp_struct_col, _exp_radar_col = st.columns([1, 1])
            with _exp_struct_col:
                _exp_png = mol_to_image(_exp_mol, size=(350, 350))
                st.image(_exp_png, use_container_width=True)
            with _exp_radar_col:
                _radar_fig = plot_radar_chart({
                    "MW": _exp_mw,
                    "TPSA": _exp_tpsa,
                    "LogP": _exp_logp,
                    "RotatableBonds": _exp_rotb,
                    "FractionCsp3": _exp_fcsp3,
                })
                st.plotly_chart(_radar_fig, use_container_width=True, config={"displayModeBar": False})
                st.caption(
                    f"LIPO={_exp_logp:.2f} | SIZE={_exp_mw:.0f} | "
                    f"POLAR={_exp_tpsa:.0f} | FLEX={_exp_rotb:.0f} | "
                    f"INSATU={_exp_insatu:.2f} | INSOLU={_exp_insolu_val:.2f}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.code(_exp_canonical, language=None)

            _exp_formula = rdMolDescriptors.CalcMolFormula(_exp_mol)
            _exp_inchi = MolToInchi(_exp_mol)
            _exp_inchikey = InchiToInchiKey(_exp_inchi) if _exp_inchi else "N/A"
            _exp_id_c1, _exp_id_c2, _exp_id_c3 = st.columns(3)
            with _exp_id_c1:
                st.markdown("**Molecular Formula**")
                st.code(_exp_formula, language=None)
            with _exp_id_c2:
                st.markdown("**InChI**")
                st.code(_exp_inchi or "N/A", language=None)
            with _exp_id_c3:
                st.markdown("**InChIKey**")
                st.code(_exp_inchikey, language=None)

            # Physicochemical Properties
            st.divider()
            st.subheader("Physicochemical Properties")
            _exp_props = [
                ("MW (g/mol)", f"{_exp_mw:.2f}"), ("LogP", f"{_exp_logp:.2f}"),
                ("TPSA (A^2)", f"{_exp_tpsa:.1f}"), ("HBD", str(_exp_hbd)),
                ("HBA", str(_exp_hba)), ("Rotatable Bonds", str(_exp_rotb)),
                ("Aromatic Rings", str(_exp_arom)), ("Fraction Csp3", f"{_exp_fcsp3:.3f}"),
                ("Heavy Atom Count", str(_exp_heavy)), ("Ring Count", str(_exp_rings)),
            ]
            _exp_r1, _exp_r2 = _exp_props[:5], _exp_props[5:]
            _exp_ca = st.columns(5)
            _exp_cb = st.columns(5)
            for col, (lbl, val) in zip(_exp_ca, _exp_r1):
                col.metric(lbl, val)
            for col, (lbl, val) in zip(_exp_cb, _exp_r2):
                col.metric(lbl, val)

            # Druglikeness Rules
            st.divider()
            st.subheader("Druglikeness Rules")
            _exp_rules = []
            _lip_p, _, _lip_r = check_lipinski(_exp_mol, max_violations=1)
            _exp_rules.append(("Lipinski (<=1 violation)", _lip_p, _lip_r or ""))
            _veb_p, _, _veb_r = check_veber(_exp_mol)
            _exp_rules.append(("Veber", _veb_p, _veb_r or ""))
            _gho_p, _, _gho_r = check_ghose(_exp_mol)
            _exp_rules.append(("Ghose", _gho_p, _gho_r or ""))
            _ega_p, _, _ega_r = check_egan(_exp_mol)
            _exp_rules.append(("Egan", _ega_p, _ega_r or ""))
            _mue_p, _, _mue_r = check_muegge(_exp_mol)
            _exp_rules.append(("Muegge", _mue_p, _mue_r or ""))
            _exp_rule_rows = [{"Rule": n, "Result": "Pass" if p else "Fail", "Detail": r}
                              for n, p, r in _exp_rules]
            _exp_rules_df = pd.DataFrame(_exp_rule_rows)

            def _style_result(val):
                if val == "Pass":
                    return "color: green; font-weight: bold"
                if val == "Fail":
                    return "color: red; font-weight: bold"
                return ""

            st.dataframe(
                _exp_rules_df.style.map(_style_result, subset=["Result"]),
                use_container_width=True, hide_index=True,
            )

            # Scores & Alerts
            st.divider()
            st.subheader("Scores & Alerts")
            _exp_sl, _exp_sr = st.columns(2)
            _exp_qed, _exp_qed_err = compute_qed(_exp_mol)
            with _exp_sl:
                st.markdown("**QED (Quantitative Estimate of Druglikeness)**")
                if _exp_qed_err:
                    st.error(_exp_qed_err)
                else:
                    st.write(f"Score: **{_exp_qed:.3f}** (0 = least drug-like, 1 = most)")
                    st.progress(float(_exp_qed))
                    if _exp_qed >= 0.67:
                        st.caption("High druglikeness")
                    elif _exp_qed >= 0.34:
                        st.caption("Moderate druglikeness")
                    else:
                        st.caption("Low druglikeness")
            try:
                _exp_sa = sascorer.calculateScore(_exp_mol)
                _exp_sa_err = None
            except Exception as e:
                _exp_sa = None
                _exp_sa_err = str(e)
            with _exp_sr:
                st.markdown("**SA Score (Synthetic Accessibility)**")
                if _exp_sa_err:
                    st.error(_exp_sa_err)
                else:
                    st.write(f"Score: **{_exp_sa:.2f}** (1 = easy, 10 = very difficult)")
                    st.progress(float(1.0 - (_exp_sa - 1) / 9.0))
                    if _exp_sa <= 3:
                        st.caption("Easy to synthesize")
                    elif _exp_sa <= 6:
                        st.caption("Moderate synthetic difficulty")
                    else:
                        st.caption("Difficult to synthesize")
            _exp_al, _exp_ar = st.columns(2)
            _exp_is_pains, _exp_pains_name = check_pains(_exp_mol)
            with _exp_al:
                st.markdown("**PAINS Filter**")
                if _exp_is_pains:
                    st.warning(f"PAINS hit: **{_exp_pains_name}**", icon="\u26A0\uFE0F")
                    _pains_info = get_pains_explanation(_exp_pains_name)
                    with st.expander("Why is this pattern problematic?", expanded=True):
                        st.markdown(f"**{_pains_info['name']}**")
                        st.markdown(_pains_info["description"])
                        st.markdown(f"**Mechanism:** {_pains_info['mechanism']}")
                        st.markdown(f"**Affected assays:** {_pains_info.get('affected_assays', 'Multiple assay types')}")
                        st.markdown(f"**Recommendation:** {_pains_info['recommendation']}")
                        st.caption(_pains_info["reference"])
                else:
                    st.success("No PAINS alerts detected")
            _exp_is_brenk, _exp_brenk_name = check_brenk(_exp_mol)
            with _exp_ar:
                st.markdown("**Brenk Filter**")
                if _exp_is_brenk:
                    st.warning(f"Brenk alert: **{_exp_brenk_name}**", icon="\u26A0\uFE0F")
                    _brenk_info = get_brenk_explanation(_exp_brenk_name)
                    with st.expander("Why is this pattern problematic?", expanded=True):
                        st.markdown(f"**{_brenk_info['name']}**")
                        st.markdown(_brenk_info["description"])
                        st.markdown(f"**Mechanism:** {_brenk_info.get('mechanism', 'Various')}")
                        st.markdown(f"**Recommendation:** {_brenk_info['recommendation']}")
                        st.caption(_brenk_info["reference"])
                else:
                    st.success("No Brenk alerts detected")

            # Water Solubility (ESOL)
            st.divider()
            st.subheader("Water Solubility (ESOL)")
            _esol_logs, _esol_mg, _esol_mol, _esol_class, _esol_err = compute_esol(_exp_mol)
            if _esol_err:
                st.error(_esol_err)
            else:
                _esol_left, _esol_right = st.columns([1, 1])
                with _esol_left:
                    st.metric("LogS (log mol/L)", f"{_esol_logs:.2f}")
                    _class_colors = {
                        "Highly soluble": "green", "Soluble": "green",
                        "Moderately soluble": "blue",
                        "Slightly soluble": "orange",
                        "Insoluble": "red", "Poorly soluble": "red",
                    }
                    _cc = _class_colors.get(_esol_class, "grey")
                    st.markdown(f"Class: :{_cc}[**{_esol_class}**]")
                with _esol_right:
                    st.metric("Solubility (mg/mL)", f"{_esol_mg:.4g}")
                    st.metric("Solubility (mol/L)", f"{_esol_mol:.4g}")
                # Gauge bar: LogS from -8 to +1
                _gauge_val = max(min((_esol_logs + 8) / 9.0, 1.0), 0.0)
                st.markdown("**Solubility scale** (LogS: -8 to +1)")
                st.progress(_gauge_val)
                st.caption(
                    "LogS predicted using the ESOL model (Delaney, 2004). "
                    "LogS = log\u2081\u2080 of aqueous solubility in mol/L. "
                    "Higher values indicate better water solubility."
                )

            # Fingerprint Preview
            st.divider()
            _prev_fp_type = st.selectbox(
                "Fingerprint type (preview)",
                ["morgan", "fcfp", "maccs", "topological", "atom_pair", "torsion", "avalon", "pattern", "layered"],
                key="explorer_fp_type",
            )
            if _prev_fp_type == "maccs":
                _prev_n_bits = 167
            else:
                _prev_n_bits = st.number_input(
                    "Number of bits (preview)", min_value=128, max_value=4096, value=2048, step=128,
                    key="explorer_fp_nbits",
                )
            if _prev_fp_type in ("morgan", "fcfp"):
                _prev_radius = st.number_input(
                    "Radius (preview)", min_value=1, max_value=4, value=2, step=1,
                    key="explorer_fp_radius",
                )
            else:
                _prev_radius = 2

            _prev_label = f"{_prev_fp_type}, {_prev_n_bits} bits"
            st.subheader(f"Fingerprint Preview ({_prev_label})")
            _exp_fp, _exp_fp_err = compute_fingerprint(_exp_mol, fp_type=_prev_fp_type, radius=_prev_radius, n_bits=_prev_n_bits)
            if _exp_fp_err:
                st.error(_exp_fp_err)
            else:
                _exp_bits_on = int(_exp_fp.sum())
                _exp_total_bits = len(_exp_fp)
                _exp_density = _exp_bits_on / _exp_total_bits
                _fp_c1, _fp_c2, _fp_c3 = st.columns(3)
                _fp_c1.metric("Bits ON", _exp_bits_on)
                _fp_c2.metric("Bits OFF", _exp_total_bits - _exp_bits_on)
                _fp_c3.metric("Bit Density", f"{_exp_density:.1%}")
                st.markdown("**Bit density**")
                st.progress(_exp_density)
                st.caption(
                    "Bit density reflects how structurally rich the fingerprint is. "
                    "Values around 3-10% are typical for drug-like molecules with Morgan (r=2, 2048 bits)."
                )

    # ── Boiled-Egg Diagram (only shown when molecules are being analyzed) ────
    _be_mols = st.session_state.get("explorer_molecules", [])

    if _be_mols:
        import plotly.graph_objects as go

        st.divider()
        st.header("\U0001F95A Boiled-Egg Diagram")

        _be_has_pipeline = (
            "pipeline_result" in st.session_state
            and len(st.session_state["pipeline_result"]["kept_smiles"]) >= 2
        )

        @st.cache_data
        def _cached_boiled_egg(smiles_tuple, labels_tuple):
            import pandas as _be_pd
            from rdkit import Chem as _be_Chem
            from rdkit.Chem import Descriptors as _be_Desc
            rows = []
            for i, smi in enumerate(smiles_tuple):
                mol = _be_Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                row = {
                    "SMILES": smi,
                    "WLOGP": _be_Desc.MolLogP(mol),
                    "TPSA": _be_Desc.TPSA(mol),
                }
                if labels_tuple and labels_tuple[i]:
                    row["Activity"] = labels_tuple[i]
                rows.append(row)
            return _be_pd.DataFrame(rows)

        if _be_has_pipeline:
            _be_result = st.session_state["pipeline_result"]
            _be_label_map = st.session_state.get("pipeline_label_map", {})
            _be_labels = [_be_label_map.get(s) for s in _be_result["kept_smiles"]]
            _be_has_labels = _be_label_map and any(l is not None for l in _be_labels)
            _be_df = _cached_boiled_egg(
                tuple(_be_result["kept_smiles"]),
                tuple(_be_labels) if _be_has_labels else None,
            )
            _be_chart, _be_n_gi, _be_n_bbb, _be_result_df, _be_sampled = plot_boiled_egg(
                _be_df, label_col="Activity" if _be_has_labels else None,
            )
        else:
            _be_fda_smiles, _ = get_fda_approved_drugs()
            _be_df = _cached_boiled_egg(tuple(_be_fda_smiles), None)
            _be_chart, _be_n_gi, _be_n_bbb, _be_result_df, _be_sampled = plot_boiled_egg(_be_df)

        # Plot all user molecules: star for selected, circle for others
        _be_sel_idx = st.session_state.get("explorer_mol_idx", 0) if len(_be_mols) > 1 else 0
        _be_other_smi, _be_other_name, _be_other_tpsa, _be_other_wlogp = [], [], [], []
        _be_sel_smi, _be_sel_name, _be_sel_tpsa, _be_sel_wlogp = None, None, None, None

        for _bi, (_b_smi, _b_name, _b_mol) in enumerate(_be_mols):
            _b_tpsa = Descriptors.TPSA(_b_mol)
            _b_wlogp = Descriptors.MolLogP(_b_mol)
            if _bi == _be_sel_idx:
                _be_sel_smi, _be_sel_name = _b_smi, _b_name
                _be_sel_tpsa, _be_sel_wlogp = _b_tpsa, _b_wlogp
            else:
                _be_other_smi.append(_b_smi)
                _be_other_name.append(_b_name)
                _be_other_tpsa.append(_b_tpsa)
                _be_other_wlogp.append(_b_wlogp)

        # Other user molecules as labeled circles
        if _be_other_smi:
            _be_chart.add_trace(go.Scatter(
                x=_be_other_tpsa, y=_be_other_wlogp,
                mode="markers+text",
                marker=dict(size=12, color="#FF6347", opacity=0.9,
                            line=dict(width=1.5, color="black")),
                name="Your molecules",
                text=_be_other_name,
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate="<b>%{text}</b><br>TPSA=%{x:.1f}<br>WLOGP=%{y:.2f}<extra></extra>",
            ))

        # Selected molecule as star
        if _be_sel_smi is not None:
            _be_chart.add_trace(go.Scatter(
                x=[_be_sel_tpsa], y=[_be_sel_wlogp],
                mode="markers+text",
                marker=dict(size=18, color="#1E90FF", opacity=1.0,
                            symbol="star",
                            line=dict(width=2, color="black")),
                name=f"Selected: {_be_sel_name}",
                text=[_be_sel_name],
                textposition="top center",
                textfont=dict(size=11, color="#1E90FF"),
                hovertemplate="<b>%{text}</b><br>TPSA=%{x:.1f}<br>WLOGP=%{y:.2f}<extra></extra>",
            ))

        if _be_has_pipeline and _be_sampled:
            st.info(f"Showing 300 of {len(_be_df)} pipeline molecules")
        st.plotly_chart(_be_chart, use_container_width=True, config={"displayModeBar": False}, key="explorer_boiled_egg")

        st.caption(
            "Molecules inside the white ellipse are predicted to be passively absorbed by the GI tract. "
            "Molecules inside the yellow ellipse are predicted to be brain-penetrant (BBB+)."
        )

        if _be_has_pipeline:
            st.write(f"**{_be_n_gi}** molecules in GI absorption zone, **{_be_n_bbb}** molecules in BBB zone")
            with st.expander("Molecules by zone"):
                _gi_mols = _be_result_df[_be_result_df["in_GI"]][["SMILES", "WLOGP", "TPSA"]].reset_index(drop=True)
                _bbb_mols = _be_result_df[_be_result_df["in_BBB"]][["SMILES", "WLOGP", "TPSA"]].reset_index(drop=True)
                _be_z1, _be_z2 = st.columns(2)
                with _be_z1:
                    st.markdown("**GI absorption zone**")
                    if len(_gi_mols):
                        st.dataframe(_gi_mols, use_container_width=True, hide_index=True)
                    else:
                        st.info("No molecules in GI absorption zone.")
                with _be_z2:
                    st.markdown("**BBB permeability zone**")
                    if len(_bbb_mols):
                        st.dataframe(_bbb_mols, use_container_width=True, hide_index=True)
                    else:
                        st.info("No molecules in BBB permeability zone.")
        else:
            st.caption("Your molecules are shown as colored markers. Grey dots show FDA-approved drugs for reference.")

        st.info(
            "The BOILED-Egg model (Daina & Zoete, 2016) predicts passive GI absorption and BBB permeability "
            "from two simple descriptors: WLOGP (lipophilicity) and TPSA (polar surface area). "
            "It is a simple but widely used heuristic \u2014 not a substitute for experimental ADMET data.",
            icon="\u2139\uFE0F",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Filter Comparison
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "comparison":
    st.header("Filter Comparison")
    st.write(
        "Run the preprocessing pipeline with two different filter configurations on the same "
        "molecule set and compare results side by side. PAINS filtering is always applied."
    )

    st.subheader("Input molecules")

    if "comparison_smiles" not in st.session_state:
        st.session_state["comparison_smiles"] = ""

    _CMP_EXAMPLES = {
        "FDA-approved drugs (20 molecules)": get_fda_approved_drugs,
        "PAINS-rich demo set (15 molecules)": get_pains_demo_set,
    }
    with st.expander("Try with example data", expanded=False):
        _cmp_choice = st.selectbox(
            "Select a dataset", list(_CMP_EXAMPLES.keys()),
            label_visibility="collapsed", key="cmp_example_choice",
        )
        _cmp_smi_ex, _cmp_desc_ex = _CMP_EXAMPLES[_cmp_choice]()
        st.caption(_cmp_desc_ex)
        if st.button("Load example data", key="cmp_load_example"):
            st.session_state["comparison_smiles"] = "\n".join(_cmp_smi_ex)
            st.rerun()

    _cmp_uploaded_file = st.file_uploader(
        "Upload a .txt, .csv, or .xlsx file with SMILES",
        type=["txt", "csv", "xlsx"],
        key="cmp_file_uploader",
    )
    _cmp_uploaded_df = None
    _cmp_uploaded_smiles_col = None
    if _cmp_uploaded_file is not None:
        _cmp_fname = _cmp_uploaded_file.name.lower()
        if _cmp_fname.endswith(".csv"):
            _cmp_uploaded_df = pd.read_csv(_cmp_uploaded_file)
        elif _cmp_fname.endswith(".xlsx"):
            _cmp_uploaded_df = pd.read_excel(_cmp_uploaded_file)
        if _cmp_uploaded_df is not None:
            for _c in ["canonical_smiles", "Smiles", "SMILES", "smiles"]:
                if _c in _cmp_uploaded_df.columns:
                    _cmp_uploaded_smiles_col = _c
                    break
            if _cmp_uploaded_smiles_col is None:
                st.warning("Could not auto-detect a SMILES column. Please select it manually.")
                _cmp_uploaded_smiles_col = st.selectbox(
                    "Which column contains the SMILES strings?",
                    _cmp_uploaded_df.columns.tolist(),
                    key="cmp_smiles_col",
                )

    _cmp_input = st.text_area(
        "Or paste SMILES here (one per line)", key="comparison_smiles", height=150,
    )

    st.subheader("Filter configurations")
    _cmp_ca, _cmp_cb = st.columns(2)
    with _cmp_ca:
        st.markdown("**Configuration A**")
        _cmp_max_a = st.number_input("Max Lipinski violations", min_value=0, max_value=4,
                                      value=1, step=1, key="cmp_max_viol_a")
        st.caption("Additional filters:")
        _cmp_brenk_a = st.checkbox("Brenk", key="cmp_brenk_a")
        _cmp_veber_a = st.checkbox("Veber", key="cmp_veber_a")
        _cmp_ghose_a = st.checkbox("Ghose", key="cmp_ghose_a")
        _cmp_egan_a = st.checkbox("Egan", key="cmp_egan_a")
        _cmp_muegge_a = st.checkbox("Muegge", key="cmp_muegge_a")
    with _cmp_cb:
        st.markdown("**Configuration B**")
        _cmp_max_b = st.number_input("Max Lipinski violations", min_value=0, max_value=4,
                                      value=0, step=1, key="cmp_max_viol_b")
        st.caption("Additional filters:")
        _cmp_brenk_b = st.checkbox("Brenk", key="cmp_brenk_b", value=True)
        _cmp_veber_b = st.checkbox("Veber", key="cmp_veber_b", value=True)
        _cmp_ghose_b = st.checkbox("Ghose", key="cmp_ghose_b")
        _cmp_egan_b = st.checkbox("Egan", key="cmp_egan_b")
        _cmp_muegge_b = st.checkbox("Muegge", key="cmp_muegge_b")

    if st.button("Compare", type="primary", key="cmp_run"):
        _cmp_list = []
        if _cmp_uploaded_df is not None and _cmp_uploaded_smiles_col:
            _cmp_list = _cmp_uploaded_df[_cmp_uploaded_smiles_col].dropna().astype(str).tolist()
        elif _cmp_uploaded_file is not None:
            _cmp_file_content = _cmp_uploaded_file.read().decode("utf-8")
            _cmp_list = [l.strip() for l in _cmp_file_content.splitlines() if l.strip()]
        elif _cmp_input.strip():
            _cmp_list = [l.strip() for l in _cmp_input.splitlines() if l.strip()]
        if not _cmp_list:
            st.warning("Please upload a file, paste SMILES, or load an example dataset.")
        else:
            with st.spinner("Running both pipelines..."):
                _cmp_ra = run_preprocessing_pipeline(
                    _cmp_list, lipinski_max_violations=_cmp_max_a,
                    enable_brenk=_cmp_brenk_a, enable_veber=_cmp_veber_a,
                    enable_ghose=_cmp_ghose_a, enable_egan=_cmp_egan_a,
                    enable_muegge=_cmp_muegge_a,
                )
                _cmp_rb = run_preprocessing_pipeline(
                    _cmp_list, lipinski_max_violations=_cmp_max_b,
                    enable_brenk=_cmp_brenk_b, enable_veber=_cmp_veber_b,
                    enable_ghose=_cmp_ghose_b, enable_egan=_cmp_egan_b,
                    enable_muegge=_cmp_muegge_b,
                )
            st.session_state["cmp_result"] = {
                "result_a": _cmp_ra, "result_b": _cmp_rb,
                "input_count": len(_cmp_list),
            }

    if "cmp_result" in st.session_state:
        _cmp_r = st.session_state["cmp_result"]
        _cmp_ra = _cmp_r["result_a"]
        _cmp_rb = _cmp_r["result_b"]
        _cmp_n = _cmp_r["input_count"]

        st.subheader("Results")
        _rc1, _rc2 = st.columns(2)
        with _rc1:
            st.markdown("**Configuration A**")
            _ka = len(_cmp_ra["kept_smiles"])
            st.metric("Kept", _ka)
            st.metric("Removed", _cmp_n - _ka)
            st.caption("Audit trail")
            st.dataframe(pd.DataFrame(_cmp_ra["audit_trail"]), use_container_width=True)
        with _rc2:
            st.markdown("**Configuration B**")
            _kb = len(_cmp_rb["kept_smiles"])
            st.metric("Kept", _kb)
            st.metric("Removed", _cmp_n - _kb)
            st.caption("Audit trail")
            st.dataframe(pd.DataFrame(_cmp_rb["audit_trail"]), use_container_width=True)

        st.subheader("Molecules that differ between configurations")
        _set_a = set(_cmp_ra["kept_smiles"])
        _set_b = set(_cmp_rb["kept_smiles"])
        _both = _set_a & _set_b
        _only_a = sorted(_set_a - _set_b)
        _only_b = sorted(_set_b - _set_a)
        st.write(
            f"**{len(_both)}** molecules kept by both | "
            f"**{len(_only_a)}** kept by A only | "
            f"**{len(_only_b)}** kept by B only"
        )
        _rl_a = {e["smiles"]: e for e in _cmp_ra["removed_log"]}
        _rl_b = {e["smiles"]: e for e in _cmp_rb["removed_log"]}
        _dc1, _dc2 = st.columns(2)
        with _dc1:
            st.markdown("**Kept by A, removed by B**")
            if _only_a:
                _rows_oa = [{"SMILES": s, "Removed at step": _rl_b.get(s, {}).get("step", "-"),
                             "Reason": _rl_b.get(s, {}).get("reason", "-")} for s in _only_a]
                st.dataframe(pd.DataFrame(_rows_oa), use_container_width=True)
            else:
                st.info("No molecules in this category.")
        with _dc2:
            st.markdown("**Kept by B, removed by A**")
            if _only_b:
                _rows_ob = [{"SMILES": s, "Removed at step": _rl_a.get(s, {}).get("step", "-"),
                             "Reason": _rl_a.get(s, {}).get("reason", "-")} for s in _only_b]
                st.dataframe(pd.DataFrame(_rows_ob), use_container_width=True)
            else:
                st.info("No molecules in this category.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Molecule Converter
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "converter":
    st.header("Molecule Converter")
    st.write(
        "Convert between SMILES, InChI, InChIKey, and molecular formula. "
        "Paste a single molecule for a full conversion card, or switch to Batch mode "
        "to process many molecules at once."
    )

    with st.expander("About these representations"):
        st.markdown(
            "| Representation | Description | Best for |\n"
            "|---|---|---|\n"
            "| **SMILES** | Line notation encoding atoms and bonds as a string | Input to ML models, database storage, copy-paste |\n"
            "| **Canonical SMILES** | Unique, standardised SMILES generated by RDKit | Deduplication, exact-match searching |\n"
            "| **Isomeric SMILES** | Canonical SMILES that also encodes stereochemistry | Chiral molecules, 3D-aware workflows |\n"
            "| **InChI** | IUPAC standard identifier; layered, verbose, unambiguous | Cross-database lookup, archiving |\n"
            "| **InChIKey** | 27-character hash of the InChI | Database keys, web search, hashing |\n"
            "| **Molecular Formula** | Atom count string (e.g. C9H8O4) | Quick composition check |\n"
        )

    _conv_mode = st.radio("Mode", ["Single molecule", "Batch conversion"],
                          horizontal=True, key="converter_mode")

    def _conv_detect_and_parse(text):
        text = text.strip()
        if not text:
            return None, None, "Empty input."
        if text.startswith("InChI="):
            mol = MolFromInchi(text)
            if mol is None:
                return None, "InChI", f"Could not parse InChI: {text}"
            return mol, "InChI", None
        else:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                return None, "SMILES", f"Could not parse SMILES: {text}"
            return mol, "SMILES", None

    def _conv_mol_to_reps(mol):
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        non_iso = Chem.MolToSmiles(mol, isomericSmiles=False)
        inchi = MolToInchi(mol) or "N/A"
        inchikey = InchiToInchiKey(inchi) if inchi != "N/A" else "N/A"
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return {
            "canonical_smiles": canonical,
            "isomeric_smiles": non_iso if non_iso != canonical else None,
            "inchi": inchi, "inchikey": inchikey, "formula": formula,
        }

    if _conv_mode == "Single molecule":
        st.subheader("Single Molecule Conversion")
        st.caption("Paste a SMILES or InChI string. The input type is detected automatically.")
        _conv_ic, _conv_bc = st.columns([5, 1])
        with _conv_ic:
            _conv_user = st.text_input(
                "SMILES or InChI", placeholder="CC(=O)Oc1ccccc1C(=O)O  or  InChI=1S/...",
                label_visibility="collapsed", key="converter_single_input",
            )
        with _conv_bc:
            _conv_go = st.button("Convert", type="primary", use_container_width=True,
                                 key="converter_single_btn")

        if (_conv_go or _conv_user) and _conv_user and _conv_user.strip():
            _conv_mol, _conv_itype, _conv_err = _conv_detect_and_parse(_conv_user)
            if _conv_err:
                st.error(_conv_err)
            else:
                st.success(f"Detected input type: **{_conv_itype}**")
                _conv_reps = _conv_mol_to_reps(_conv_mol)
                st.divider()
                _conv_img_c, _conv_id_c = st.columns([1, 2])
                with _conv_img_c:
                    st.image(mol_to_image(_conv_mol, size=(300, 300)), caption="2D Structure")
                with _conv_id_c:
                    st.markdown("**Canonical SMILES**")
                    st.code(_conv_reps["canonical_smiles"], language=None)
                    if _conv_reps["isomeric_smiles"] is not None:
                        st.markdown("**Non-isomeric SMILES** *(molecule has stereocentres)*")
                        st.code(_conv_reps["isomeric_smiles"], language=None)
                    st.markdown("**Molecular Formula**")
                    st.code(_conv_reps["formula"], language=None)
                    st.markdown("**InChI**")
                    st.code(_conv_reps["inchi"], language=None)
                    st.markdown("**InChIKey**")
                    st.code(_conv_reps["inchikey"], language=None)
    else:
        st.subheader("Batch Conversion")
        st.caption("Enter one SMILES per line, or upload a .txt / .csv / .xlsx file.")
        _conv_file = st.file_uploader(
            "Upload file", type=["txt", "csv", "xlsx"],
            label_visibility="collapsed", key="converter_batch_file",
        )
        _conv_pasted = st.text_area("Or paste SMILES here (one per line)", height=150,
                                     key="converter_batch_text")
        if st.button("Convert All", type="primary", key="converter_batch_btn"):
            _conv_slist = []
            if _conv_file is not None:
                _conv_fname = _conv_file.name.lower()
                if _conv_fname.endswith(".csv") or _conv_fname.endswith(".xlsx"):
                    _conv_df_up = (pd.read_csv(_conv_file) if _conv_fname.endswith(".csv")
                                   else pd.read_excel(_conv_file))
                    _conv_scol = None
                    for _cand in ["canonical_smiles", "SMILES", "Smiles", "smiles"]:
                        if _cand in _conv_df_up.columns:
                            _conv_scol = _cand
                            break
                    if _conv_scol is None:
                        _conv_scol = st.selectbox("Select the SMILES column",
                                                   _conv_df_up.columns.tolist(),
                                                   key="converter_batch_col")
                    _conv_slist = _conv_df_up[_conv_scol].dropna().astype(str).tolist()
                else:
                    _conv_content = _conv_file.read().decode("utf-8")
                    _conv_slist = [l.strip() for l in _conv_content.splitlines() if l.strip()]
            elif _conv_pasted.strip():
                _conv_slist = [l.strip() for l in _conv_pasted.splitlines() if l.strip()]
            if not _conv_slist:
                st.warning("Please upload a file or paste at least one SMILES string.")
            else:
                _conv_rows = []
                _conv_nfail = 0
                for _cs in _conv_slist:
                    _cm = Chem.MolFromSmiles(_cs)
                    if _cm is None:
                        _conv_rows.append({"Input_SMILES": _cs, "Canonical_SMILES": "PARSE ERROR",
                                           "InChI": "", "InChIKey": "", "Molecular_Formula": ""})
                        _conv_nfail += 1
                    else:
                        _cr = _conv_mol_to_reps(_cm)
                        _conv_rows.append({"Input_SMILES": _cs, "Canonical_SMILES": _cr["canonical_smiles"],
                                           "InChI": _cr["inchi"], "InChIKey": _cr["inchikey"],
                                           "Molecular_Formula": _cr["formula"]})
                _conv_result_df = pd.DataFrame(_conv_rows)
                _conv_nok = len(_conv_slist) - _conv_nfail
                st.write(f"**{_conv_nok}** converted successfully, **{_conv_nfail}** failed to parse.")
                st.dataframe(_conv_result_df, use_container_width=True)
                _conv_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "Download conversion table as CSV",
                    data=_conv_result_df.to_csv(index=False),
                    file_name=f"molecule_conversions_{_conv_ts}.csv",
                    mime="text/csv", key="converter_batch_dl",
                )
