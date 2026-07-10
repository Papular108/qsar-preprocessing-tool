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
from pipeline.featurization import featurize_dataset, compute_descriptors, compute_fingerprint, compute_esol, find_similar_molecules, batch_similarity_search, multi_reference_similarity
from pipeline.splitting import scaffold_split, random_split, compute_split_pca
from pipeline.clustering import cluster_butina, cluster_hierarchical, compute_cluster_pca
import altair as alt
from pipeline.visualization import mol_to_base64_png, mol_to_image, plot_boiled_egg, plot_radar_chart, plot_mini_radar
from pipeline.methodology import generate_methods_text
from pipeline.pains_catalog import get_pains_explanation, get_brenk_explanation
from pipeline.example_data import get_fda_approved_drugs, get_pains_demo_set, get_dpp4_inhibitors, get_common_analgesics
from pipeline.ui_components import (
    VERSION, timestamp_filename, render_dataset_status_bar, render_empty_state,
    render_provenance_caption, render_next_step_hint, render_footer,
    migrate_legacy_session_state,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def section_banner(title, color="#c8102e"):
    """Render a red section banner like SwissADME."""
    st.markdown(
        f'<div style="background:{color};color:white;padding:6px 12px;'
        f'border-radius:6px 6px 0 0;font-weight:600;font-size:1rem;'
        f'margin-top:20px;margin-bottom:0;">{title}</div>',
        unsafe_allow_html=True,
    )


def compact_table(rows):
    """Render a compact property table (SwissADME style)."""
    html_parts = ['<table style="width:100%;border-collapse:collapse;font-size:0.9rem;margin-bottom:16px;">']
    for label, value in rows:
        html_parts.append(
            f'<tr style="border-bottom:1px solid #eee;">'
            f'<td style="padding:4px 8px;color:#666;width:50%;">{label}</td>'
            f'<td style="padding:4px 8px;font-weight:500;">{value}</td></tr>'
        )
    html_parts.append('</table>')
    st.markdown(''.join(html_parts), unsafe_allow_html=True)


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
    /* ── Category buttons (level 1) ── */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_categories"])
        [data-testid="stButton"] > button {
        min-height: 56px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: 2px solid #e0e3e8 !important;
        background: #f8f9fa !important;
        color: #1a1a2e !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_categories"])
        [data-testid="stButton"] > button:hover {
        background: #eef0f4 !important;
        border-color: #ff4b4b !important;
    }
    .cat-active [data-testid="stButton"] > button {
        background: white !important;
        border-color: #ff4b4b !important;
        border-bottom: 3px solid #ff4b4b !important;
        color: #ff4b4b !important;
    }
    /* ── Tab buttons (level 2) ── */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_cards"])
        [data-testid="stButton"] > button {
        min-height: 52px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: 1.5px solid #e0e3e8 !important;
        background: white !important;
        color: #1a1a2e !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="nav_cards"])
        [data-testid="stButton"] > button:hover {
        box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
        border-color: #ff4b4b !important;
    }
    .nav-active [data-testid="stButton"] > button {
        background: #fafbff !important;
        border-bottom: 3px solid #ff4b4b !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("QSAR Preprocessing Tool")
st.write("Welcome! This tool helps preprocess and featurize molecules for QSAR/virtual screening workflows.")

# ── Two-level navigation: categories → tabs ──
_NAV_STRUCTURE = {
    "data": {
        "label": "Data",
        "icon": "📥",
        "subtitle": "Load & prepare",
        "tabs": [
            ("preprocessing", "🧪", "Preprocessing", "Clean, filter & featurize"),
            ("converter", "🔄", "Molecule Converter", "SMILES, InChI & more"),
        ],
    },
    "analyze": {
        "label": "Analyze",
        "icon": "📊",
        "subtitle": "Explore & compare",
        "tabs": [
            ("explorer", "🧬", "Molecule Explorer", "Analyze a single molecule"),
            ("comparison", "⚖️", "Filter Comparison", "Compare settings side by side"),
            ("screening", "🔍", "Similarity Screening", "Multi-reference virtual screening"),
            ("clustering", "🔬", "Cluster Analysis", "Group molecules by similarity"),
        ],
    },
    "model": {
        "label": "Model",
        "icon": "🧠",
        "subtitle": "Split, train & predict",
        "tabs": [
            ("splitting", "✂️", "Train/Test Split", "Scaffold & random splitting"),
        ],
    },
}

# Reverse lookup: tab_key → category_key (built from structure, stays in sync)
_TAB_TO_CATEGORY = {}
for _cat_key, _cat_info in _NAV_STRUCTURE.items():
    for (_tab_key, *_) in _cat_info["tabs"]:
        _TAB_TO_CATEGORY[_tab_key] = _cat_key

# Session state initialization (backwards-compatible: infer category from existing tab)
if "active_category" not in st.session_state:
    _existing_tab = st.session_state.get("active_tab", "preprocessing")
    st.session_state["active_category"] = _TAB_TO_CATEGORY.get(_existing_tab, "data")
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "preprocessing"

# Migrate legacy session state → active_dataset (one-time, idempotent)
migrate_legacy_session_state()

# Defensive: ensure active_tab belongs to active_category
if _TAB_TO_CATEGORY.get(st.session_state["active_tab"]) != st.session_state["active_category"]:
    st.session_state["active_tab"] = _NAV_STRUCTURE[st.session_state["active_category"]]["tabs"][0][0]

# Level 1: Category row
with st.container(key="nav_categories"):
    _cat_cols = st.columns(3)
    for _col, (_ck, _ci) in zip(_cat_cols, _NAV_STRUCTURE.items()):
        _is_active_cat = st.session_state["active_category"] == _ck
        with _col:
            if _is_active_cat:
                st.markdown('<div class="cat-active">', unsafe_allow_html=True)
            if st.button(f"{_ci['icon']} {_ci['label']}", key=f"cat_{_ck}", use_container_width=True):
                if not _is_active_cat:
                    st.session_state["active_category"] = _ck
                    st.session_state["active_tab"] = _ci["tabs"][0][0]
                    st.rerun()  # needed: tab row must re-render for new category
            st.caption(_ci["subtitle"])
            if _is_active_cat:
                st.markdown('</div>', unsafe_allow_html=True)

# Level 2: Tab row (only active category's tabs)
_active_tabs = _NAV_STRUCTURE[st.session_state["active_category"]]["tabs"]
with st.container(key="nav_cards"):
    # Pad single-tab categories so the button isn't full-width
    _n_tab_cols = 3 if len(_active_tabs) == 1 else len(_active_tabs)
    _tab_cols = st.columns(_n_tab_cols)
    for _col, (_tk, _ti, _tt, _ts) in zip(_tab_cols, _active_tabs):
        _is_active_tab = st.session_state["active_tab"] == _tk
        with _col:
            if _is_active_tab:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            if st.button(f"{_ti} {_tt}", key=f"nav_{_tk}", use_container_width=True):
                if not _is_active_tab:
                    st.session_state["active_tab"] = _tk
                    st.rerun()  # needed: content below must re-render for new tab
            st.caption(_ts)
            if _is_active_tab:
                st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "preprocessing":

    st.header("Batch preprocessing")
    st.write("Upload a file with one SMILES string per line, or paste them below.")

    # ── Landing card for first-time visitors ──────────────────────────────────
    if "active_dataset" not in st.session_state and "pipeline_result" not in st.session_state:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);border-radius:12px;'
            'padding:32px;margin-bottom:24px;border:1px solid #dee2e6;">'
            '<h3 style="margin-top:0;color:#1a1a2e;">Welcome to the QSAR Preprocessing Tool</h3>'
            '<p style="color:#495057;font-size:1rem;margin-bottom:20px;">'
            'Clean, filter, and featurize molecular datasets for QSAR modeling and virtual screening. '
            'Upload your own data or start with an example dataset.</p>'
            '<p style="color:#6c757d;font-size:0.95rem;margin-bottom:4px;"><strong>Quick start:</strong></p>'
            '<ol style="color:#6c757d;font-size:0.95rem;margin-top:0;">'
            '<li><strong>Upload</strong> a CSV/TXT with SMILES, or load an example dataset below</li>'
            '<li><strong>Preprocess</strong> — standardize, filter (Lipinski, PAINS, etc.), deduplicate</li>'
            '<li><strong>Analyze</strong> — explore molecules, cluster, split for modeling</li></ol>'
            '</div>',
            unsafe_allow_html=True,
        )

    if "pasted_smiles" not in st.session_state:
        st.session_state["pasted_smiles"] = ""

    with st.expander("Try with example data", expanded="active_dataset" not in st.session_state and "pipeline_result" not in st.session_state):
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
            st.session_state["_example_source"] = example_choice
            st.rerun()

    uploaded_file = st.file_uploader("Upload a .txt, .csv, or .xlsx file with SMILES", type=["txt", "csv", "xlsx"])
    if uploaded_file is not None:
        st.session_state.pop("_example_source", None)
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
            section_banner("Column Overview")
            compact_table([
                ("Rows", f"{_n_rows:,}"),
                ("Columns", f"{_n_cols:,}"),
                ("SMILES column", _uploaded_smiles_col or "\u2014"),
            ])
            if _numeric_cols_all:
                st.write(
                    "**Numeric columns:** "
                    + ", ".join(f"`{c}`" for c in _numeric_cols_all)
                )
            else:
                st.write("No numeric columns detected.")

            # ── Missing data table (before cleaning) ──
            section_banner("Missing Data")
            _miss_html = _build_missing_data_html(_uploaded_df, _uploaded_smiles_col)
            if _miss_html:
                st.markdown(_miss_html, unsafe_allow_html=True)
            else:
                st.info("No key columns found to display missing-data statistics.")

            # ── Clean Dataset ──
            section_banner("Clean Dataset")
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
                compact_table([
                    ("Rows before", f"{_report['rows_before']:,}"),
                    ("Rows removed", f"{_report['missing_smiles_removed'] + _report['invalid_smiles_removed']:,}"),
                    ("Rows after", f"{_report['rows_after']:,}"),
                ])

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
                section_banner("Missing Data (after cleaning)")
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

            # ── Populate active_dataset ───────────────────────────────────
            if uploaded_file is not None:
                _source_filename = uploaded_file.name
            elif st.session_state.get("_example_source"):
                _source_filename = st.session_state.pop("_example_source") + " (example)"
            elif pasted_smiles.strip():
                _source_filename = "Pasted SMILES"
            else:
                _source_filename = "Unknown"
            st.session_state["active_dataset"] = {
                "source_filename": _source_filename,
                "loaded_at": datetime.now().isoformat(),
                "n_original": len(smiles_list),
                "n_molecules": len(result["kept_smiles"]),
                "mols": result["kept_mols"],
                "smiles": result["kept_smiles"],
                "preprocessing_config": st.session_state["pipeline_settings"],
                "pipeline_result": st.session_state["pipeline_result"],
                "label_map": st.session_state["pipeline_label_map"],
                "pval_map": st.session_state["pipeline_pval_map"],
                "pchembl_map": st.session_state["pipeline_pchembl_map"],
            }
            # Clear stale downstream results (dataset changed)
            # NOTE: if these keys are renamed in a future cleanup, update this list too.
            for _stale_key in ("split_results", "split_data_hash", "split_pca_current",
                               "split_pca_random", "cluster_results", "cluster_data_hash",
                               "cluster_pca", "scr_results", "featurization_result"):
                st.session_state.pop(_stale_key, None)
            # Clear next-step hint dismissals so they re-appear for new data
            for _hk in list(st.session_state):
                if _hk.startswith("hint_dismissed_"):
                    del st.session_state[_hk]

    # ── Display pipeline results (persistent) ────────────────────────────────────
    if "pipeline_result" in st.session_state:
        result = st.session_state["pipeline_result"]
        _ts = st.session_state["pipeline_ts"]
        _pl_label_map = st.session_state.get("pipeline_label_map", {})
        _pl_pval_map = st.session_state.get("pipeline_pval_map", {})
        _pl_pchembl_map = st.session_state.get("pipeline_pchembl_map", {})

        section_banner("Pipeline Results")
        _removed_count = result['input_count'] - len(result['kept_smiles'])
        _result_rows = [
            ("Input molecules", str(result['input_count'])),
            ("Kept after preprocessing", str(len(result['kept_smiles']))),
            ("Removed", str(_removed_count)),
        ]
        if _pl_label_map:
            _kept_labels_summary = [_pl_label_map.get(s) for s in result["kept_smiles"]]
            for _cls in ["Active", "Intermediate", "Inactive"]:
                _cnt = _kept_labels_summary.count(_cls)
                if _cnt:
                    _result_rows.append((f"{_cls} kept", str(_cnt)))
        compact_table(_result_rows)

        with st.expander("Audit Trail"):
            audit_df = pd.DataFrame(result["audit_trail"])
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

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
        _dl_left, _dl_right = st.columns(2)
        with _dl_left:
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

        with _dl_right:
            if result["removed_log"]:
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
                st.download_button(
                    "Download removed molecules as CSV",
                    data=removed_df.drop(columns=["explanation"]).to_csv(index=False),
                    file_name=f"removed_molecules_{_ts}.csv",
                    mime="text/csv",
                )

        with st.expander(f"View removed molecules ({len(result['removed_log'])})"):
            if result["removed_log"]:
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
                else:
                    st.dataframe(removed_df, use_container_width=True, hide_index=True)
            else:
                st.info("No molecules were removed.")

        section_banner("Descriptor Distributions")
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
            section_banner("Descriptor Correlations")
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
            section_banner("Chemical Space Visualization")
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
        section_banner("Scaffold Analysis")
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
        _top_smi, _top_cnt = list(_scaf["scaffold_counts"].items())[0]
        compact_table([
            ("Unique Scaffolds", str(_scaf["unique_scaffold_count"])),
            ("Most Common Scaffold", f"{_top_cnt}x"),
            ("Singleton Scaffolds", str(_scaf["singleton_count"])),
        ])

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
        compact_table([("Diversity Score", f"{_diversity:.2f}")])
        st.progress(min(_diversity, 1.0))
        st.caption(
            "High scaffold diversity (>0.5) indicates a structurally diverse dataset. "
            "Low diversity suggests the dataset is dominated by a few chemical series."
        )

    section_banner("Featurization")
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

        section_banner("Featurized Data")
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

    section_banner("Generate Methods Section")
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

    # ── Next-step hint (after preprocessing) ──────────────────────────────────
    if "active_dataset" in st.session_state:
        render_next_step_hint(
            "preprocessing_complete",
            "Preprocessing complete. Explore your dataset or prepare it for modeling:",
            [
                ("Molecule Explorer", "explorer", "analyze"),
                ("Cluster Analysis", "clustering", "analyze"),
                ("Train/Test Split", "splitting", "model"),
            ],
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Molecule Explorer
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "explorer":
    render_dataset_status_bar()

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
            # ── Structure & Radar (top row) ──
            _exp_canonical = Chem.MolToSmiles(_exp_mol)
            _exp_insatu = 1.0 - _exp_fcsp3
            _exp_insolu_val = (_exp_logp + 2.0) / 7.0
            _exp_formula = rdMolDescriptors.CalcMolFormula(_exp_mol)
            _exp_inchi = MolToInchi(_exp_mol)
            _exp_inchikey = InchiToInchiKey(_exp_inchi) if _exp_inchi else "N/A"

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
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Compute all values needed for tables ──
            _exp_mr = Descriptors.MolMR(_exp_mol)
            _exp_arom_heavy = sum(1 for a in _exp_mol.GetAtoms() if a.GetIsAromatic())
            _exp_qed, _exp_qed_err = compute_qed(_exp_mol)
            try:
                _exp_sa = sascorer.calculateScore(_exp_mol)
                _exp_sa_err = None
            except Exception as e:
                _exp_sa = None
                _exp_sa_err = str(e)
            _exp_is_pains, _exp_pains_name = check_pains(_exp_mol)
            _exp_is_brenk, _exp_brenk_name = check_brenk(_exp_mol)
            _esol_logs, _esol_mg, _esol_mol_sol, _esol_class, _esol_err = compute_esol(_exp_mol)

            _lip_p, _, _lip_r = check_lipinski(_exp_mol, max_violations=1)
            _veb_p, _, _veb_r = check_veber(_exp_mol)
            _gho_p, _, _gho_r = check_ghose(_exp_mol)
            _ega_p, _, _ega_r = check_egan(_exp_mol)
            _mue_p, _, _mue_r = check_muegge(_exp_mol)

            def _pass_fail(passed, detail=None):
                icon = '<span style="color:green;font-weight:700">Yes</span>' if passed else '<span style="color:red;font-weight:700">No</span>'
                if detail:
                    return f'{icon}; {detail}'
                return icon

            # Leadlikeness check (250 < MW < 350, LogP <= 3.5, RotBonds <= 7)
            _lead_violations = []
            if _exp_mw < 250 or _exp_mw > 350:
                _lead_violations.append(f"MW={_exp_mw:.0f}")
            if _exp_logp > 3.5:
                _lead_violations.append(f"LogP={_exp_logp:.2f}")
            if _exp_rotb > 7:
                _lead_violations.append(f"RotBonds={_exp_rotb}")
            _lead_pass = len(_lead_violations) == 0

            # ── Two-column property layout ──
            _exp_left, _exp_right = st.columns(2)

            with _exp_left:
                section_banner("Physicochemical Properties")
                compact_table([
                    ("Formula", _exp_formula),
                    ("Molecular weight", f"{_exp_mw:.2f} g/mol"),
                    ("Num. heavy atoms", str(_exp_heavy)),
                    ("Num. arom. heavy atoms", str(_exp_arom_heavy)),
                    ("Fraction Csp3", f"{_exp_fcsp3:.3f}"),
                    ("Num. rotatable bonds", str(_exp_rotb)),
                    ("Num. H-bond acceptors", str(_exp_hba)),
                    ("Num. H-bond donors", str(_exp_hbd)),
                    ("Molar Refractivity", f"{_exp_mr:.2f}"),
                    ("TPSA", f"{_exp_tpsa:.2f} \u00c5\u00b2"),
                ])

                section_banner("Lipophilicity")
                compact_table([
                    ("Log P (Crippen/WLOGP)", f"{_exp_logp:.2f}"),
                    ("Consensus Log P", f"{_exp_logp:.2f}"),
                ])

                section_banner("Water Solubility")
                if _esol_err:
                    st.error(_esol_err)
                else:
                    _sol_mg_str = f"{_esol_mg:.4g} mg/ml ; {_esol_mol_sol:.4g} mol/l"
                    compact_table([
                        ("Log S (ESOL)", f"{_esol_logs:.2f}"),
                        ("Solubility", _sol_mg_str),
                        ("Class", _esol_class),
                    ])

            with _exp_right:
                section_banner("Druglikeness")
                compact_table([
                    ("Lipinski", _pass_fail(_lip_p, _lip_r)),
                    ("Veber", _pass_fail(_veb_p, _veb_r)),
                    ("Ghose", _pass_fail(_gho_p, _gho_r)),
                    ("Egan", _pass_fail(_ega_p, _ega_r)),
                    ("Muegge", _pass_fail(_mue_p, _mue_r)),
                    ("Bioavailability Score", f"{_exp_qed:.2f}" if not _exp_qed_err else "N/A"),
                ])

                section_banner("Medicinal Chemistry")
                _pains_str = f'<span style="color:red">1 alert: {_exp_pains_name}</span>' if _exp_is_pains else '<span style="color:green">0 alerts</span>'
                _brenk_str = f'<span style="color:red">1 alert: {_exp_brenk_name}</span>' if _exp_is_brenk else '<span style="color:green">0 alerts</span>'
                compact_table([
                    ("PAINS", _pains_str),
                    ("Brenk", _brenk_str),
                    ("Leadlikeness", _pass_fail(_lead_pass, "; ".join(_lead_violations) if _lead_violations else None)),
                    ("Synthetic accessibility", f"{_exp_sa:.2f}" if not _exp_sa_err else "N/A"),
                ])

            # ── PAINS / Brenk detail expanders ──
            if _exp_is_pains or _exp_is_brenk:
                _exp_al, _exp_ar = st.columns(2)
                if _exp_is_pains:
                    with _exp_al:
                        _pains_info = get_pains_explanation(_exp_pains_name)
                        with st.expander(f"PAINS: {_exp_pains_name}", expanded=True):
                            st.markdown(_pains_info["description"])
                            st.markdown(f"**Mechanism:** {_pains_info['mechanism']}")
                            st.markdown(f"**Recommendation:** {_pains_info['recommendation']}")
                if _exp_is_brenk:
                    with _exp_ar:
                        _brenk_info = get_brenk_explanation(_exp_brenk_name)
                        with st.expander(f"Brenk: {_exp_brenk_name}", expanded=True):
                            st.markdown(_brenk_info["description"])
                            st.markdown(f"**Mechanism:** {_brenk_info.get('mechanism', 'Various')}")
                            st.markdown(f"**Recommendation:** {_brenk_info['recommendation']}")

            # ── Identifiers ──
            section_banner("Identifiers")
            compact_table([
                ("Canonical SMILES", f"<code>{_exp_canonical}</code>"),
                ("InChI", f"<code style='word-break:break-all'>{_exp_inchi or 'N/A'}</code>"),
                ("InChIKey", f"<code>{_exp_inchikey}</code>"),
                ("Molecular Formula", _exp_formula),
            ])

            # Fingerprint Preview
            st.divider()
            _prev_fp_type = st.selectbox(
                "Fingerprint type (preview)",
                ["morgan", "fcfp", "maccs", "topological", "atom_pair", "torsion", "avalon", "pattern", "layered"],
                key="explorer_fp_type",
                help="Morgan: circular, counts atom environments (most common for QSAR). "
                     "FCFP: like Morgan but uses pharmacophoric atom types. "
                     "MACCS: 166 predefined structural keys. "
                     "Topological: path-based. Atom pair / Torsion: encode atom-pair or dihedral relationships. "
                     "Avalon: combines path and feature fingerprints.",
            )
            if _prev_fp_type == "maccs":
                _prev_n_bits = 167
            else:
                _prev_n_bits = st.number_input(
                    "Number of bits (preview)", min_value=128, max_value=4096, value=2048, step=128,
                    key="explorer_fp_nbits",
                    help="Length of the bit vector. More bits = fewer hash collisions = more discriminating, "
                         "but larger feature vectors. 2048 is standard for most QSAR applications.",
                )
            if _prev_fp_type in ("morgan", "fcfp"):
                _prev_radius = st.number_input(
                    "Radius (preview)", min_value=1, max_value=4, value=2, step=1,
                    key="explorer_fp_radius",
                    help="Number of bonds from each atom to include in the circular environment. "
                         "Radius 2 (≈ECFP4) is standard. Radius 3 captures larger substructures but may overfit.",
                )
            else:
                _prev_radius = 2

            _prev_label = f"{_prev_fp_type}, {_prev_n_bits} bits"
            section_banner(f"Fingerprint Preview ({_prev_label})")
            _exp_fp, _exp_fp_err = compute_fingerprint(_exp_mol, fp_type=_prev_fp_type, radius=_prev_radius, n_bits=_prev_n_bits)
            if _exp_fp_err:
                st.error(_exp_fp_err)
            else:
                _exp_bits_on = int(_exp_fp.sum())
                _exp_total_bits = len(_exp_fp)
                _exp_density = _exp_bits_on / _exp_total_bits
                compact_table([
                    ("Bits ON", str(_exp_bits_on)),
                    ("Bits OFF", str(_exp_total_bits - _exp_bits_on)),
                    ("Bit Density", f"{_exp_density:.1%}"),
                ])
                st.progress(_exp_density)
                st.caption(
                    "Bit density reflects how structurally rich the fingerprint is. "
                    "Values around 3-10% are typical for drug-like molecules with Morgan (r=2, 2048 bits)."
                )

    # ── Boiled-Egg Diagram (only shown when molecules are being analyzed) ────
    _be_mols = st.session_state.get("explorer_molecules", [])

    if _be_mols:
        import plotly.graph_objects as go

        section_banner("Boiled-Egg Diagram")
        _be_ds = st.session_state.get("active_dataset")
        _be_has_pipeline = (
            _be_ds is not None and len(_be_ds.get("smiles", [])) >= 2
        )

        # ── Display options ──
        if _be_has_pipeline:
            _be_show_bg = st.checkbox(
                "Show preprocessed dataset as background context",
                value=False, key="be_show_bg",
                help="Overlay your preprocessed molecules as faint background dots for reference.",
            )
        else:
            _be_show_bg = False

        with st.expander("Display options"):
            _be_show_names = st.checkbox("Show molecule names on plot", value=False, key="be_show_names",
                                         help="Label each query molecule with its name. Background dataset dots are never labeled.")

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

        # ── Build query molecule data (always needed for classification + plotting) ──
        _be_query_smiles = [smi for smi, _name, _mol in _be_mols]
        _be_query_df = _cached_boiled_egg(tuple(_be_query_smiles), None)
        # Classify query molecules for zone counts
        _be_chart, _be_q_n_gi, _be_q_n_bbb, _be_q_result_df, _ = plot_boiled_egg(
            _be_query_df, draw_molecules=False,
        )

        # ── Background dataset (only when toggle is on) ──
        _be_bg_n_gi, _be_bg_n_bbb, _be_bg_result_df, _be_bg_sampled = 0, 0, None, False
        if _be_show_bg and _be_has_pipeline:
            _be_bg_df = _cached_boiled_egg(tuple(_be_ds["smiles"]), None)
            _, _be_bg_n_gi, _be_bg_n_bbb, _be_bg_result_df, _be_bg_sampled = plot_boiled_egg(
                _be_bg_df, draw_molecules=False,
            )
            # Add faint background dots
            _be_bg_plot = _be_bg_df if len(_be_bg_df) <= 300 else _be_bg_df.sample(n=300, random_state=42)
            _be_chart.add_trace(go.Scatter(
                x=_be_bg_plot["TPSA"], y=_be_bg_plot["WLOGP"],
                mode="markers",
                marker=dict(size=5, color="#aaaaaa", opacity=0.25),
                name="Dataset",
                text=_be_bg_plot["SMILES"],
                hovertemplate="<b>Dataset</b><br>%{text}<br>TPSA=%{x:.1f}<br>WLOGP=%{y:.2f}<extra></extra>",
            ))

        # ── Plot query molecules (always) ──
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

        def _be_label(name, smi):
            """Label text for query molecules: name if available, else truncated SMILES."""
            if name and not name.startswith("Molecule "):
                return name
            return smi[:15] + ("..." if len(smi) > 15 else "")

        _be_query_mode = "markers+text" if _be_show_names else "markers"

        # Non-selected query molecules as blue dots
        if _be_other_smi:
            _be_other_labels = [_be_label(n, s) for n, s in zip(_be_other_name, _be_other_smi)]
            _be_chart.add_trace(go.Scatter(
                x=_be_other_tpsa, y=_be_other_wlogp,
                mode=_be_query_mode,
                marker=dict(size=10, color="#1E90FF", opacity=0.9,
                            line=dict(width=1.5, color="white")),
                name="Your molecules",
                text=_be_other_labels,
                textposition="top center",
                textfont=dict(size=10, color="#1E90FF"),
                hovertemplate="<b>%{text}</b><br>TPSA=%{x:.1f}<br>WLOGP=%{y:.2f}<extra></extra>",
            ))

        # Selected molecule as blue star
        if _be_sel_smi is not None:
            _be_sel_label = _be_label(_be_sel_name, _be_sel_smi)
            _be_chart.add_trace(go.Scatter(
                x=[_be_sel_tpsa], y=[_be_sel_wlogp],
                mode="markers+text",  # always show name for selected molecule
                marker=dict(size=18, color="#1E90FF", opacity=1.0,
                            symbol="star",
                            line=dict(width=2, color="white")),
                name=f"Selected: {_be_sel_name}",
                text=[_be_sel_label],
                textposition="top center",
                textfont=dict(size=11, color="#1E90FF"),
                hovertemplate="<b>%{text}</b><br>TPSA=%{x:.1f}<br>WLOGP=%{y:.2f}<extra></extra>",
            ))

        # Bump figure size
        _be_chart.update_layout(width=800, height=600)

        if _be_show_bg and _be_bg_sampled:
            st.info(f"Showing 300 of {len(_be_bg_df)} dataset molecules as background")
        st.plotly_chart(_be_chart, use_container_width=True, config={"displayModeBar": False}, key="explorer_boiled_egg")

        st.caption(
            "Molecules inside the white ellipse are predicted to be passively absorbed by the GI tract. "
            "Molecules inside the yellow ellipse are predicted to be brain-penetrant (BBB+)."
        )

        # Zone summary — describe what's shown
        if _be_show_bg and _be_has_pipeline:
            _be_bg_total = len(_be_bg_result_df) if _be_bg_result_df is not None else 0
            st.write(f"**{_be_bg_n_gi}** in GI zone, **{_be_bg_n_bbb}** in BBB zone")
            st.caption(f"Zone distribution across your {_be_bg_total:,}-molecule dataset.")
            render_provenance_caption(_be_ds)
            with st.expander("Molecules by zone"):
                _gi_mols = _be_bg_result_df[_be_bg_result_df["in_GI"]][["SMILES", "WLOGP", "TPSA"]].reset_index(drop=True)
                _bbb_mols = _be_bg_result_df[_be_bg_result_df["in_BBB"]][["SMILES", "WLOGP", "TPSA"]].reset_index(drop=True)
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
            st.write(f"**{_be_q_n_gi}** in GI zone, **{_be_q_n_bbb}** in BBB zone")
            st.caption(f"Zone distribution for your {len(_be_query_df)} query molecule(s).")

        st.info(
            "The BOILED-Egg model (Daina & Zoete, 2016) predicts passive GI absorption and BBB permeability "
            "from two simple descriptors: WLOGP (lipophilicity) and TPSA (polar surface area). "
            "It is a simple but widely used heuristic \u2014 not a substitute for experimental ADMET data.",
            icon="\u2139\uFE0F",
        )
        st.caption(
            "Note: WLOGP is computed using RDKit's Crippen implementation. Values may differ by ~0.1\u20130.3 units "
            "from SwissADME's WLOGP, which can cause borderline molecules to appear in slightly different zones. "
            "The scientific interpretation remains the same."
        )

    # ── Similarity Search ─────────────────────────────────────────────────────
    section_banner("Similarity Search")
    st.caption("Compare reference molecules against a target dataset using Tanimoto similarity on molecular fingerprints.")

    # --- Input panels: Reference (left) | Target (right) ---
    _sim_ref_col, _sim_tgt_col = st.columns(2)

    with _sim_ref_col:
        st.markdown("**Reference set**")
        # Quick-load buttons
        _SIM_REF_PRESETS = {
            "FDA-approved drugs": get_fda_approved_drugs,
            "DPP-4 inhibitors": get_dpp4_inhibitors,
        }
        _sim_preset_cols = st.columns(len(_SIM_REF_PRESETS))
        for _pc, (_pname, _pfunc) in zip(_sim_preset_cols, _SIM_REF_PRESETS.items()):
            if _pc.button(_pname, key=f"sim_preset_{_pname}", use_container_width=True):
                _p_smiles, _ = _pfunc()
                st.session_state["sim_ref_input"] = "\n".join(_p_smiles)
                st.rerun()

        if "sim_ref_input" not in st.session_state:
            st.session_state["sim_ref_input"] = ""
        _sim_ref_text = st.text_area(
            "Paste reference SMILES (one per line, optional name after space)",
            placeholder="CC(=O)Oc1ccccc1C(=O)O Aspirin\nCC(C)Cc1ccc(cc1)C(C)C(=O)O Ibuprofen",
            key="sim_ref_input",
            height=150,
        )

    with _sim_tgt_col:
        st.markdown("**Target dataset**")
        _sim_tgt_source = st.radio(
            "Source",
            ["Pipeline kept molecules", "Upload file"],
            key="sim_tgt_source",
            horizontal=True,
        )
        _sim_target_smiles = []
        _sim_target_labels = {}
        if _sim_tgt_source == "Pipeline kept molecules":
            _sim_ds = st.session_state.get("active_dataset")
            if _sim_ds and _sim_ds["smiles"]:
                _sim_target_smiles = _sim_ds["smiles"]
                _sim_target_labels = _sim_ds.get("label_map") or {}
                st.success(f"{len(_sim_target_smiles)} molecules from active dataset")
            else:
                st.info("No active dataset. Run the Preprocessing tab first.")
        else:
            _sim_tgt_file = st.file_uploader(
                "Upload .txt / .csv / .xlsx",
                type=["txt", "csv", "xlsx"],
                key="sim_tgt_file",
            )
            if _sim_tgt_file is not None:
                _sim_fname = _sim_tgt_file.name.lower()
                if _sim_fname.endswith(".csv"):
                    _sim_tgt_df = pd.read_csv(_sim_tgt_file)
                elif _sim_fname.endswith(".xlsx"):
                    _sim_tgt_df = pd.read_excel(_sim_tgt_file)
                else:
                    _sim_tgt_df = None
                    _sim_content = _sim_tgt_file.read().decode("utf-8")
                    _sim_target_smiles = [l.strip() for l in _sim_content.splitlines() if l.strip()]
                if _sim_tgt_df is not None:
                    _sim_scol = None
                    for _c in ["canonical_smiles", "Smiles", "SMILES", "smiles"]:
                        if _c in _sim_tgt_df.columns:
                            _sim_scol = _c
                            break
                    if _sim_scol is None and len(_sim_tgt_df.columns) > 0:
                        _sim_scol = st.selectbox("SMILES column", _sim_tgt_df.columns.tolist(), key="sim_tgt_scol")
                    if _sim_scol:
                        _sim_target_smiles = _sim_tgt_df[_sim_scol].dropna().astype(str).tolist()
                if _sim_target_smiles:
                    st.success(f"{len(_sim_target_smiles)} molecules loaded")

    # --- Settings ---
    _sim_s1, _sim_s2, _sim_s3 = st.columns(3)
    with _sim_s1:
        _sim_threshold = st.number_input("Similarity threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="sim_threshold",
                                          help="Minimum Tanimoto similarity to include a hit. 0.4 is a permissive cutoff for analog discovery; 0.7+ finds close analogs.")
    with _sim_s2:
        _sim_top_n = st.number_input("Top N per reference", min_value=1, max_value=50, value=5, key="sim_top_n",
                                     help="Maximum number of hits to return per reference molecule, ranked by similarity.")
    with _sim_s3:
        _sim_fp_type = st.selectbox(
            "Fingerprint type",
            ["morgan", "fcfp", "maccs", "topological", "atom_pair", "torsion", "avalon"],
            key="sim_fp_type",
            help="Fingerprint used for Tanimoto comparison. Morgan (ECFP-like) is the most common choice for similarity searching.",
        )

    # --- Run ---
    if st.button("Run Similarity Search", type="primary", key="sim_search_btn"):
        # Parse reference molecules
        _sim_refs = []
        if _sim_ref_text and _sim_ref_text.strip():
            for _li, _line in enumerate(_sim_ref_text.strip().splitlines()):
                _line = _line.strip()
                if not _line:
                    continue
                _parts = _line.split(None, 1)
                _smi = _parts[0]
                _name = _parts[1] if len(_parts) > 1 else f"Ref {_li + 1}"
                _mol = Chem.MolFromSmiles(_smi)
                if _mol:
                    _sim_refs.append((_name, _smi, _mol))
                else:
                    st.error(f"Could not parse: `{_smi}`")

        if not _sim_refs:
            st.warning("Paste at least one valid reference SMILES.")
        elif not _sim_target_smiles:
            st.warning("No target molecules available. Load pipeline results or upload a file.")
        else:
            with st.spinner("Computing fingerprints and similarities..."):
                @st.cache_data
                def _cached_batch_sim(ref_tuples, target_smi_tuple, top_n, threshold, fp_type):
                    q_mols = [(n, Chem.MolFromSmiles(s)) for n, s in ref_tuples]
                    t_mols = [Chem.MolFromSmiles(s) for s in target_smi_tuple]
                    return batch_similarity_search(
                        q_mols, t_mols, top_n=top_n, threshold=threshold, fp_type=fp_type,
                    )

                _ref_tuples = tuple((n, s) for n, s, _m in _sim_refs)
                _batch_results = _cached_batch_sim(
                    _ref_tuples, tuple(_sim_target_smiles),
                    _sim_top_n, _sim_threshold, _sim_fp_type,
                )
            st.session_state["batch_sim_results"] = _batch_results
            st.session_state["batch_sim_refs"] = [(n, s) for n, s, _m in _sim_refs]
            st.session_state["batch_sim_targets"] = list(_sim_target_smiles)

    # --- Display results ---
    if "batch_sim_results" in st.session_state:
        _bsr = st.session_state["batch_sim_results"]
        _bsr_refs = st.session_state["batch_sim_refs"]
        _bsr_targets = st.session_state["batch_sim_targets"]
        _bsr_ds = st.session_state.get("active_dataset")
        _bsr_label_map = (_bsr_ds.get("label_map") or {}) if _bsr_ds else {}

        _total_hits = sum(len(v) for v in _bsr.values())
        st.write(f"**{_total_hits}** hits across **{len(_bsr_refs)}** reference molecules (threshold \u2265 {_sim_threshold})")

        # Per-reference results
        for _ref_name, _ref_smi in _bsr_refs:
            _ref_hits = _bsr.get(_ref_name, [])
            _ref_mol = Chem.MolFromSmiles(_ref_smi)
            with st.expander(f"{_ref_name} \u2014 {len(_ref_hits)} hits", expanded=len(_bsr_refs) <= 5):
                # Reference molecule header
                _rh_img, _rh_info = st.columns([1, 3])
                with _rh_img:
                    if _ref_mol:
                        st.image(mol_to_image(_ref_mol, size=(120, 120)), width=120)
                with _rh_info:
                    st.code(_ref_smi[:60] + ("..." if len(_ref_smi) > 60 else ""), language=None)

                if not _ref_hits:
                    st.info("No targets above similarity threshold.")
                    continue

                # Hit cards in 2-column grid
                for _hi in range(0, len(_ref_hits), 2):
                    _hcols = st.columns(2)
                    for _ci, _hcol in enumerate(_hcols):
                        _hidx = _hi + _ci
                        if _hidx >= len(_ref_hits):
                            break
                        _t_idx, _t_score = _ref_hits[_hidx]
                        _t_smi = _bsr_targets[_t_idx]
                        _t_mol = Chem.MolFromSmiles(_t_smi)
                        _t_label = _bsr_label_map.get(_t_smi)

                        if _t_score > 0.7:
                            _sc = "#2ca02c"
                        elif _t_score >= 0.4:
                            _sc = "#ff7f0e"
                        else:
                            _sc = "#999999"

                        with _hcol:
                            st.markdown(
                                f'<div style="border-left:4px solid {_sc};padding:4px 8px;margin-bottom:4px;">'
                                f'<span style="font-weight:700;color:{_sc};">{_t_score:.1%}</span></div>',
                                unsafe_allow_html=True,
                            )
                            if _t_mol:
                                st.image(mol_to_image(_t_mol, size=(130, 130)), width=130)
                            _trunc = _t_smi[:45] + ("..." if len(_t_smi) > 45 else "")
                            st.code(_trunc, language=None)
                            if _t_label:
                                _bc = {"Active": "green", "Intermediate": "orange", "Inactive": "red"}.get(_t_label, "gray")
                                st.markdown(
                                    f'<span style="background:{_bc};color:white;padding:2px 8px;'
                                    f'border-radius:4px;font-size:12px">{_t_label}</span>',
                                    unsafe_allow_html=True,
                                )

        # Heatmap
        if len(_bsr_refs) > 1 and _total_hits > 0:
            st.subheader("Similarity Heatmap")
            # Collect all unique target indices that appear in any result
            _all_hit_idxs = sorted({idx for hits in _bsr.values() for idx, _ in hits})
            if _all_hit_idxs:
                _heat_rows = []
                for _ref_name, _ref_smi in _bsr_refs:
                    _hit_map = {idx: score for idx, score in _bsr.get(_ref_name, [])}
                    for _tidx in _all_hit_idxs:
                        _t_smi = _bsr_targets[_tidx]
                        _t_short = _t_smi[:25] + ("..." if len(_t_smi) > 25 else "")
                        _heat_rows.append({
                            "Reference": _ref_name,
                            "Target": _t_short,
                            "Tanimoto": _hit_map.get(_tidx, 0.0),
                        })
                _heat_df = pd.DataFrame(_heat_rows)
                _heat_chart = (
                    alt.Chart(_heat_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("Target:N", title=None, sort=None),
                        y=alt.Y("Reference:N", title=None, sort=None),
                        color=alt.Color(
                            "Tanimoto:Q",
                            scale=alt.Scale(domain=[0, 0.4, 0.7, 1.0], range=["#f0f0f0", "#fee08b", "#fdae61", "#2ca02c"]),
                            legend=alt.Legend(title="Tanimoto"),
                        ),
                        tooltip=[
                            alt.Tooltip("Reference:N"),
                            alt.Tooltip("Target:N"),
                            alt.Tooltip("Tanimoto:Q", format=".3f"),
                        ],
                    )
                )
                _heat_text = (
                    alt.Chart(_heat_df)
                    .mark_text(fontSize=10)
                    .encode(
                        x=alt.X("Target:N", sort=None),
                        y=alt.Y("Reference:N", sort=None),
                        text=alt.Text("Tanimoto:Q", format=".2f"),
                        color=alt.condition("datum.Tanimoto > 0.6", alt.value("white"), alt.value("#333")),
                    )
                )
                st.altair_chart(
                    (_heat_chart + _heat_text).properties(
                        height=max(len(_bsr_refs) * 35, 120),
                    ),
                    use_container_width=True,
                )

        # Download CSV
        _dl_rows = []
        for _ref_name, _ref_smi in _bsr_refs:
            for _rank, (_t_idx, _t_score) in enumerate(_bsr.get(_ref_name, []), 1):
                _t_smi = _bsr_targets[_t_idx]
                _row = {
                    "Reference": _ref_name,
                    "Reference_SMILES": _ref_smi,
                    "Rank": _rank,
                    "Target_SMILES": _t_smi,
                    "Tanimoto_Similarity": round(_t_score, 4),
                }
                _lbl = _bsr_label_map.get(_t_smi)
                if _lbl:
                    _row["Activity_Label"] = _lbl
                _dl_rows.append(_row)
        if _dl_rows:
            _dl_df = pd.DataFrame(_dl_rows)
            st.download_button(
                "Download similarity results as CSV",
                data=_dl_df.to_csv(index=False),
                file_name=timestamp_filename("similarity_results"),
                mime="text/csv",
                key="sim_download",
            )

        st.caption(
            "Tanimoto similarity ranges from 0 (no shared features) to 1 (identical). "
            "In drug discovery: Tanimoto > 0.85 usually indicates near-duplicates or same scaffold, "
            "0.4\u20130.7 indicates analogs, < 0.3 indicates structurally different molecules. "
            "The 'activity cliff' phenomenon is when similar structures have very different activities."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Filter Comparison
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "comparison":
    render_dataset_status_bar()

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
                                      value=1, step=1, key="cmp_max_viol_a",
                                      help="0 = strict (no violations allowed), 1 = standard threshold, higher = more permissive.")
        st.caption("Additional filters:")
        _cmp_brenk_a = st.checkbox("Brenk", key="cmp_brenk_a", help="Flags structural alerts beyond PAINS.")
        _cmp_veber_a = st.checkbox("Veber", key="cmp_veber_a", help="Rotatable bonds ≤10 and TPSA ≤140.")
        _cmp_ghose_a = st.checkbox("Ghose", key="cmp_ghose_a", help="Drug-like ranges for MW, LogP, refractivity, atom count.")
        _cmp_egan_a = st.checkbox("Egan", key="cmp_egan_a", help="LogP and TPSA within the Egan egg boundary.")
        _cmp_muegge_a = st.checkbox("Muegge", key="cmp_muegge_a", help="Combined pharmacophore-like property rules.")
    with _cmp_cb:
        st.markdown("**Configuration B**")
        _cmp_max_b = st.number_input("Max Lipinski violations", min_value=0, max_value=4,
                                      value=0, step=1, key="cmp_max_viol_b",
                                      help="0 = strict (no violations allowed), 1 = standard threshold, higher = more permissive.")
        st.caption("Additional filters:")
        _cmp_brenk_b = st.checkbox("Brenk", key="cmp_brenk_b", value=True, help="Flags structural alerts beyond PAINS.")
        _cmp_veber_b = st.checkbox("Veber", key="cmp_veber_b", value=True, help="Rotatable bonds ≤10 and TPSA ≤140.")
        _cmp_ghose_b = st.checkbox("Ghose", key="cmp_ghose_b", help="Drug-like ranges for MW, LogP, refractivity, atom count.")
        _cmp_egan_b = st.checkbox("Egan", key="cmp_egan_b", help="LogP and TPSA within the Egan egg boundary.")
        _cmp_muegge_b = st.checkbox("Muegge", key="cmp_muegge_b", help="Combined pharmacophore-like property rules.")

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

        section_banner("Results")
        _rc1, _rc2 = st.columns(2)
        with _rc1:
            _ka = len(_cmp_ra["kept_smiles"])
            st.markdown("**Configuration A**")
            compact_table([("Kept", str(_ka)), ("Removed", str(_cmp_n - _ka))])
            with st.expander("Audit trail"):
                st.dataframe(pd.DataFrame(_cmp_ra["audit_trail"]), use_container_width=True, hide_index=True)
        with _rc2:
            _kb = len(_cmp_rb["kept_smiles"])
            st.markdown("**Configuration B**")
            compact_table([("Kept", str(_kb)), ("Removed", str(_cmp_n - _kb))])
            with st.expander("Audit trail"):
                st.dataframe(pd.DataFrame(_cmp_rb["audit_trail"]), use_container_width=True, hide_index=True)

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
    render_dataset_status_bar()

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
                st.download_button(
                    "Download conversion table as CSV",
                    data=_conv_result_df.to_csv(index=False),
                    file_name=timestamp_filename("converter_results"),
                    mime="text/csv", key="converter_batch_dl",
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Similarity Screening
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "screening":
    render_dataset_status_bar()

    st.header("Multi-Reference Similarity Screening")
    st.write("Compare a set of reference molecules against a target dataset to find similar compounds using Tanimoto similarity.")

    # ── Two-panel input ──
    _scr_ref_col, _scr_tgt_col = st.columns(2)

    with _scr_ref_col:
        section_banner("Reference Molecules")
        st.write("Enter reference molecules (one per line). Optionally add a name after the SMILES separated by a space.")

        _SCR_REF_PRESETS = {
            "Common analgesics": get_common_analgesics,
        }
        _scr_ref_preset = st.selectbox("Load a preset", ["(none)"] + list(_SCR_REF_PRESETS.keys()),
                                        key="scr_ref_preset")
        if "scr_ref_text" not in st.session_state:
            st.session_state["scr_ref_text"] = ""
        if _scr_ref_preset != "(none)" and st.button("Load preset", key="scr_load_ref_preset"):
            preset_data, _ = _SCR_REF_PRESETS[_scr_ref_preset]()
            st.session_state["scr_ref_text"] = "\n".join(preset_data)
            st.rerun()

        _scr_ref_input = st.text_area("Reference SMILES", value=st.session_state["scr_ref_text"],
                                       height=180, key="scr_ref_area",
                                       placeholder="CC(=O)Oc1ccccc1C(=O)O Aspirin\nCC(C)Cc1ccc(cc1)C(C)C(=O)O Ibuprofen")
        st.session_state["scr_ref_text"] = _scr_ref_input

    with _scr_tgt_col:
        section_banner("Target Dataset")
        st.write("Upload a file or paste SMILES. This is the library you want to screen against.")

        _scr_tgt_source = st.radio("Target source", ["Use preprocessed results", "Upload file", "Paste SMILES"],
                                    key="scr_tgt_source", horizontal=True)

        _scr_tgt_smiles_list = []
        if _scr_tgt_source == "Use preprocessed results":
            _scr_ds = st.session_state.get("active_dataset")
            if _scr_ds and _scr_ds["smiles"]:
                _scr_tgt_smiles_list = _scr_ds["smiles"]
                st.success(f"{len(_scr_tgt_smiles_list)} molecules from active dataset")
            else:
                st.warning("No active dataset. Upload or paste targets below, or go to Preprocessing first.")
        elif _scr_tgt_source == "Upload file":
            _scr_tgt_file = st.file_uploader("Upload target file", type=["txt", "csv", "xlsx"], key="scr_tgt_file")
            if _scr_tgt_file is not None:
                _scr_fname = _scr_tgt_file.name
                if _scr_fname.endswith(".csv") or _scr_fname.endswith(".xlsx"):
                    _scr_df = pd.read_csv(_scr_tgt_file) if _scr_fname.endswith(".csv") else pd.read_excel(_scr_tgt_file)
                    _scr_col = None
                    for c in ["canonical_smiles", "SMILES", "Smiles", "smiles"]:
                        if c in _scr_df.columns:
                            _scr_col = c
                            break
                    if _scr_col is None:
                        _scr_col = st.selectbox("Select SMILES column", _scr_df.columns.tolist(), key="scr_tgt_col")
                    _scr_tgt_smiles_list = _scr_df[_scr_col].dropna().astype(str).tolist()
                else:
                    _scr_content = _scr_tgt_file.read().decode("utf-8")
                    _scr_tgt_smiles_list = [l.strip() for l in _scr_content.splitlines() if l.strip()]
                st.success(f"{len(_scr_tgt_smiles_list)} target molecules loaded")
        else:
            _scr_tgt_pasted = st.text_area("Paste target SMILES (one per line)", height=150, key="scr_tgt_paste")
            if _scr_tgt_pasted.strip():
                _scr_tgt_smiles_list = [l.strip() for l in _scr_tgt_pasted.splitlines() if l.strip()]

        # Example target dataset
        if not _scr_tgt_smiles_list:
            if st.button("Load FDA-approved drugs as targets", key="scr_load_fda"):
                _fda_smiles, _ = get_fda_approved_drugs()
                st.session_state["_scr_fda_loaded"] = _fda_smiles
                st.rerun()
            if "_scr_fda_loaded" in st.session_state:
                _scr_tgt_smiles_list = st.session_state["_scr_fda_loaded"]
                st.success(f"{len(_scr_tgt_smiles_list)} FDA-approved drugs loaded as targets")

    # ── Settings ──
    with st.expander("Screening settings", expanded=False):
        _scr_s1, _scr_s2, _scr_s3, _scr_s4 = st.columns(4)
        with _scr_s1:
            _scr_fp_type = st.selectbox("Fingerprint", ["morgan", "fcfp", "maccs", "topological",
                                        "atom_pair", "torsion", "avalon"], key="scr_fp_type",
                                        help="Fingerprint used for Tanimoto comparison. Morgan (ECFP-like) is the most common choice.")
        with _scr_s2:
            _scr_radius = st.number_input("Radius", min_value=1, max_value=4, value=2, key="scr_radius",
                                          help="Circular fingerprint radius. 2 (≈ECFP4) is standard. Only applies to Morgan/FCFP.")
        with _scr_s3:
            _scr_nbits = st.selectbox("Bits", [512, 1024, 2048], index=2, key="scr_nbits",
                                      help="Fingerprint length. More bits = fewer hash collisions. 2048 is standard.")
        with _scr_s4:
            _scr_threshold = st.slider("Min. similarity", 0.0, 1.0, 0.4, 0.05, key="scr_threshold",
                                       help="Minimum Tanimoto similarity to include a hit. 0.4 is permissive; 0.7+ finds close analogs only.")
        _scr_topn = st.number_input("Max hits per reference", min_value=1, max_value=100, value=10, key="scr_topn",
                                    help="Maximum number of hits to return per reference molecule, ranked by similarity.")

    # ── Run screening ──
    if st.button("Run Similarity Screening", type="primary", key="scr_run"):
        # Parse reference molecules
        _scr_ref_lines = [l.strip() for l in _scr_ref_input.splitlines() if l.strip()]
        if not _scr_ref_lines:
            st.error("Please enter at least one reference molecule.")
        elif not _scr_tgt_smiles_list:
            st.error("Please provide a target dataset.")
        else:
            _scr_ref_mols = []
            _scr_ref_errors = []
            for line in _scr_ref_lines:
                parts = line.split(None, 1)
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else smi
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    _scr_ref_errors.append(name)
                else:
                    _scr_ref_mols.append((name, mol))

            if _scr_ref_errors:
                st.warning(f"Could not parse {len(_scr_ref_errors)} reference molecule(s): {', '.join(_scr_ref_errors)}")

            # Parse target molecules
            _scr_tgt_mols = []
            _scr_tgt_valid_smiles = []
            _scr_tgt_fail = 0
            for smi in _scr_tgt_smiles_list:
                s = smi.split()[0] if smi.split() else smi
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    _scr_tgt_mols.append(mol)
                    _scr_tgt_valid_smiles.append(Chem.MolToSmiles(mol))
                else:
                    _scr_tgt_fail += 1

            if _scr_tgt_fail > 0:
                st.warning(f"{_scr_tgt_fail} target molecule(s) failed to parse and were skipped.")

            if _scr_ref_mols and _scr_tgt_mols:
                with st.spinner("Computing fingerprints and similarities..."):
                    _scr_results = multi_reference_similarity(
                        ref_mols=_scr_ref_mols,
                        target_mols=_scr_tgt_mols,
                        target_smiles=_scr_tgt_valid_smiles,
                        top_n=_scr_topn,
                        threshold=_scr_threshold,
                        fp_type=_scr_fp_type,
                        radius=_scr_radius,
                        n_bits=_scr_nbits,
                    )

                st.session_state["scr_results"] = _scr_results
                st.session_state["scr_ref_mols_display"] = _scr_ref_mols

    # ── Display results ──
    if "scr_results" in st.session_state:
        _scr_res = st.session_state["scr_results"]
        _scr_stats = _scr_res["stats"]

        render_provenance_caption()
        section_banner("Screening Results")
        _stat1, _stat2, _stat3 = st.columns(3)
        with _stat1:
            st.metric("Total targets", _scr_stats["total_targets"])
        with _stat2:
            st.metric("Hits above threshold", _scr_stats["targets_above_threshold"])
        with _stat3:
            st.metric("Avg. best similarity", f"{_scr_stats['avg_similarity']:.3f}")

        # Per-reference results
        section_banner("Hits per Reference")
        for ref_data in _scr_res["per_reference"]:
            ref_name = ref_data["ref_name"]
            hits = ref_data["hits"]
            with st.expander(f"{ref_name} — {len(hits)} hit(s)", expanded=len(hits) > 0):
                if ref_data["ref_smiles"]:
                    ref_mol = Chem.MolFromSmiles(ref_data["ref_smiles"])
                    if ref_mol:
                        st.image(mol_to_image(ref_mol, size=(200, 200)), caption=f"Reference: {ref_name}", width=200)
                if hits:
                    hit_rows = []
                    for h in hits:
                        hit_rows.append({
                            "Target SMILES": h["target_smiles"],
                            "Similarity": h["similarity"],
                        })
                    st.dataframe(pd.DataFrame(hit_rows), use_container_width=True, hide_index=True)

                    # Show top 3 hit structures
                    _hit_cols = st.columns(min(3, len(hits)))
                    for ci, h in enumerate(hits[:3]):
                        hmol = Chem.MolFromSmiles(h["target_smiles"])
                        if hmol:
                            with _hit_cols[ci]:
                                st.image(mol_to_image(hmol, size=(180, 180)),
                                         caption=f"Sim: {h['similarity']:.3f}", width=180)
                else:
                    st.write("No hits above the similarity threshold.")

        # Heatmap
        _sim_matrix = _scr_res["similarity_matrix"]
        if not _sim_matrix.empty and len(_sim_matrix.columns) <= 50:
            section_banner("Similarity Heatmap")
            import plotly.graph_objects as go
            # Truncate column labels for readability
            col_labels = [s[:20] + "..." if len(s) > 20 else s for s in _sim_matrix.columns.tolist()]
            _hm_fig = go.Figure(data=go.Heatmap(
                z=_sim_matrix.values,
                x=col_labels,
                y=_sim_matrix.index.tolist(),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                colorbar=dict(title="Tanimoto"),
                hovertemplate="Reference: %{y}<br>Target: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
            ))
            _hm_fig.update_layout(
                width=min(900, 200 + len(col_labels) * 60),
                height=200 + len(_sim_matrix) * 40,
                margin=dict(l=120, r=30, t=30, b=120),
                xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(_hm_fig, use_container_width=True)
        elif not _sim_matrix.empty:
            st.info("Heatmap is available for up to 50 target hits. Use a higher similarity threshold to reduce the number of hits.")

        # All hits table + download
        _all_hits = _scr_res["all_hits"]
        if len(_all_hits) > 0:
            section_banner("All Hits (Deduplicated)")
            st.write(f"**{len(_all_hits)}** unique target molecules matched at least one reference above the threshold.")
            st.dataframe(_all_hits, use_container_width=True, hide_index=True)

            st.download_button(
                "Download all hits as CSV",
                data=_all_hits.to_csv(index=False),
                file_name=timestamp_filename("screening_hits"),
                mime="text/csv",
                key="scr_dl_hits",
            )

        # Educational caption
        st.caption(
            "**How to interpret:** Tanimoto similarity measures fingerprint overlap (0 = no shared features, "
            "1 = identical). Scores > 0.85 indicate near-duplicates; 0.7-0.85 close analogs; 0.4-0.7 moderate "
            "similarity. The fingerprint type affects scores — Morgan (ECFP) gives lower values than MACCS for "
            "the same pair because it captures finer structural detail."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "splitting":
    render_dataset_status_bar()

    st.header("Train/Test Split")
    st.write("Generate realistic train/test splits for downstream QSAR modeling. "
             "Scaffold splitting ensures no Murcko scaffold appears in both sets, "
             "forcing the model to generalize to unseen chemotypes.")

    # ── Staleness check ──
    def _split_data_hash(smiles_list):
        if not smiles_list:
            return None
        return (smiles_list[0], smiles_list[-1], len(smiles_list))

    # ── Check for active dataset ──
    _spl_ds = st.session_state.get("active_dataset")
    if not _spl_ds:
        render_empty_state(
            "✂️",
            "No active dataset. Preprocess molecules first to generate train/test splits.",
        )
        st.stop()

    _spl_mols = _spl_ds["mols"]
    _spl_smiles = _spl_ds["smiles"]
    _spl_current_hash = _split_data_hash(_spl_smiles)

    # Staleness: clear old results if the dataset changed
    if "split_data_hash" in st.session_state and st.session_state["split_data_hash"] != _spl_current_hash:
        for k in ["split_results", "split_pca_current", "split_pca_random", "split_method"]:
            st.session_state.pop(k, None)
        st.warning("The preprocessed dataset has changed since the last split. Please re-run the split.")
    st.session_state["split_data_hash"] = _spl_current_hash

    # ── Settings ──
    _spl_c1, _spl_c2, _spl_c3, _spl_c4 = st.columns(4)
    with _spl_c1:
        _spl_method = st.radio("Split method", ["Scaffold", "Random", "Stratified random"],
                                key="spl_method_radio", horizontal=False,
                                help="Scaffold: splits by Murcko scaffold so no chemotype appears in both sets (recommended for QSAR). "
                                     "Random: standard shuffle. Stratified: preserves class balance across sets.")
    with _spl_c2:
        _spl_test_size = st.slider("Test fraction", 0.1, 0.5, 0.2, 0.05, key="spl_test_size",
                                   help="Proportion of molecules to place in the test set. "
                                        "0.2 (20%) is standard. Scaffold splits may deviate from this target.")
    with _spl_c3:
        _spl_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, key="spl_seed",
                                    help="Controls reproducibility. Same seed + same data = same split.")
    with _spl_c4:
        _spl_labels = None
        _spl_label_col = None
        if _spl_method == "Stratified random":
            # Check for activity labels in session state
            _spl_available_labels = {}
            if "activity_labels" in st.session_state:
                _spl_available_labels["Activity (from preprocessing)"] = st.session_state["activity_labels"]
            if _spl_available_labels:
                _spl_label_col = st.selectbox("Label column", list(_spl_available_labels.keys()), key="spl_label_col")
                _spl_labels = _spl_available_labels[_spl_label_col]
            else:
                st.warning("No activity labels found. Run preprocessing with activity labeling, or use a different split method.")

    # ── Run split ──
    if st.button("Run Split", type="primary", key="spl_run"):
        if _spl_method == "Stratified random" and _spl_labels is None:
            st.error("Stratified split requires activity labels. Please select a label column or use a different method.")
        else:
            with st.spinner("Computing split..."):
                if _spl_method == "Scaffold":
                    _spl_result = scaffold_split(_spl_mols, _spl_smiles,
                                                  test_size=_spl_test_size, random_state=_spl_seed)
                elif _spl_method == "Random":
                    _spl_result = random_split(_spl_mols, _spl_smiles,
                                               test_size=_spl_test_size, random_state=_spl_seed)
                else:
                    _spl_result = random_split(_spl_mols, _spl_smiles,
                                               test_size=_spl_test_size, random_state=_spl_seed,
                                               labels=_spl_labels, stratify=True)

                st.session_state["split_results"] = _spl_result
                st.session_state["split_method"] = _spl_method

                # Compute PCA for current split
                if _spl_result["test_indices"]:
                    _spl_pca = compute_split_pca(_spl_mols, _spl_result["train_indices"],
                                                  _spl_result["test_indices"])
                    st.session_state["split_pca_current"] = _spl_pca

                    # Also compute random baseline PCA for comparison (if not already random)
                    if _spl_method != "Random":
                        _spl_rand = random_split(_spl_mols, _spl_smiles,
                                                  test_size=_spl_test_size, random_state=_spl_seed)
                        _spl_pca_rand = compute_split_pca(_spl_mols, _spl_rand["train_indices"],
                                                           _spl_rand["test_indices"])
                        st.session_state["split_pca_random"] = _spl_pca_rand
                    else:
                        st.session_state.pop("split_pca_random", None)
                else:
                    st.session_state.pop("split_pca_current", None)
                    st.session_state.pop("split_pca_random", None)

            st.rerun()

    # ── Display results ──
    if "split_results" in st.session_state:
        _spl_res = st.session_state["split_results"]
        _spl_meth = st.session_state.get("split_method", "")

        # Fallback warning
        if _spl_res.get("fallback") == "single_scaffold":
            st.error("All molecules share the same Murcko scaffold — scaffold splitting is not possible. "
                     "All molecules were placed in the training set. Consider using a random split instead.")
        elif _spl_res.get("fallback") == "too_small":
            st.warning("Dataset too small to split (< 2 molecules). All molecules placed in training set.")

        # Summary table
        render_provenance_caption()
        section_banner("Split Summary")
        _n_train = len(_spl_res["train_indices"])
        _n_test = len(_spl_res["test_indices"])
        _n_total = _n_train + _n_test
        _achieved = _spl_res["achieved_test_fraction"]
        _deviation = abs(_achieved - _spl_test_size) * 100

        _summary_rows = [
            ("Method", _spl_meth),
            ("Total molecules", str(_n_total)),
            ("Train set", f"{_n_train} ({_n_train/_n_total*100:.1f}%)" if _n_total > 0 else "0"),
            ("Test set", f"{_n_test} ({_n_test/_n_total*100:.1f}%)" if _n_total > 0 else "0"),
            ("Target test fraction", f"{_spl_test_size:.0%}"),
            ("Achieved test fraction", f"{_achieved:.1%}"),
        ]
        if _spl_meth == "Scaffold":
            _summary_rows.append(("Scaffolds", str(_spl_res.get("n_scaffolds", "-"))))
            _summary_rows.append(("Singleton scaffolds", str(_spl_res.get("n_singletons", "-"))))

        compact_table(_summary_rows)

        if _deviation > 10:
            st.warning(f"The achieved test fraction ({_achieved:.1%}) deviates from the target "
                       f"({_spl_test_size:.0%}) by {_deviation:.1f} percentage points. This typically happens "
                       f"when one or more scaffold groups are large relative to the dataset. "
                       f"Consider adjusting the test fraction or using a random split.")

        # Scaffold distribution (scaffold split only)
        if _spl_meth == "Scaffold" and "scaffold_counts" in _spl_res:
            section_banner("Scaffold Distribution")
            import plotly.graph_objects as go

            _sc_df = _spl_res["scaffold_counts"]
            _sc_top = _sc_df.head(20).copy()
            _sc_other_count = _sc_df.iloc[20:]["Count"].sum() if len(_sc_df) > 20 else 0

            _sc_labels = []
            for i, row in _sc_top.iterrows():
                scaf = row["Scaffold"]
                if scaf == "":
                    _sc_labels.append("(acyclic)")
                elif len(scaf) > 25:
                    _sc_labels.append(scaf[:22] + "...")
                else:
                    _sc_labels.append(scaf)

            _sc_counts = _sc_top["Count"].tolist()
            if _sc_other_count > 0:
                _sc_labels.append(f"Other ({len(_sc_df) - 20})")
                _sc_counts.append(_sc_other_count)

            _sc_fig = go.Figure(go.Bar(
                y=_sc_labels[::-1],
                x=_sc_counts[::-1],
                orientation="h",
                marker_color="#1f77b4",
            ))
            _sc_fig.update_layout(
                height=max(300, len(_sc_labels) * 25 + 80),
                margin=dict(l=180, r=20, t=30, b=40),
                xaxis=dict(title=dict(text="Number of molecules", font=dict(size=13, color="#333")),
                           tickfont=dict(color="#333")),
                yaxis=dict(tickfont=dict(size=10, color="#333")),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(_sc_fig, use_container_width=True)
            st.caption("Datasets dominated by a few large scaffolds produce more challenging scaffold splits, "
                       "because entire clusters of similar molecules move together. Many singleton scaffolds "
                       "(scaffolds with only one molecule) make the split behave more like a random split.")

        # Chemical space PCA
        if "split_pca_current" in st.session_state:
            section_banner("Chemical Space (PCA)")
            import plotly.graph_objects as go

            _pca_cur = st.session_state["split_pca_current"]
            _has_random_comparison = "split_pca_random" in st.session_state

            if _has_random_comparison:
                _pca_col1, _pca_col2 = st.columns(2)
            else:
                _pca_col1 = st.container()
                _pca_col2 = None

            def _pca_scatter(pca_data, title):
                df = pca_data["pca_df"]
                vr = pca_data["var_ratio"]
                fig = go.Figure()
                for split_val, color in [("Train", "#1f77b4"), ("Test", "#ff7f0e")]:
                    subset = df[df["Split"] == split_val]
                    fig.add_trace(go.Scatter(
                        x=subset["PC1"], y=subset["PC2"],
                        mode="markers",
                        marker=dict(size=8, color=color, opacity=0.75),
                        name=split_val,
                        text=subset["SMILES"],
                        hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                    ))
                fig.update_layout(
                    title=dict(text=title, font=dict(size=13, color="#333")),
                    xaxis=dict(title=dict(text=f"PC1 ({vr[0]*100:.1f}% var.)", font=dict(size=13, color="#333")),
                               tickfont=dict(color="#333"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
                    yaxis=dict(title=dict(text=f"PC2 ({vr[1]*100:.1f}% var.)", font=dict(size=13, color="#333")),
                               tickfont=dict(color="#333"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
                    width=450, height=400,
                    margin=dict(l=50, r=20, t=40, b=50),
                    paper_bgcolor="white", plot_bgcolor="white",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                                font=dict(color="#333")),
                )
                return fig

            with _pca_col1:
                st.plotly_chart(_pca_scatter(_pca_cur, f"{_spl_meth} Split"),
                               use_container_width=True)

            if _has_random_comparison and _pca_col2 is not None:
                _pca_rand = st.session_state["split_pca_random"]
                with _pca_col2:
                    st.plotly_chart(_pca_scatter(_pca_rand, "Random Split (baseline)"),
                                   use_container_width=True)

            st.caption("Each point is a molecule projected into 2D chemical space via PCA on Morgan fingerprints. "
                       "In a scaffold split, train (blue) and test (orange) points tend to occupy different regions, "
                       "meaning the model must extrapolate to new chemical space. In a random split, train and test "
                       "are interleaved — easier for the model but less realistic.")

        # Split details + download
        section_banner("Split Details")
        _dl_col1, _dl_col2 = st.columns(2)

        with _dl_col1:
            with st.expander(f"Train set ({_n_train} molecules)", expanded=False):
                st.dataframe(_spl_res["train_df"], use_container_width=True, hide_index=True, height=300)

        with _dl_col2:
            with st.expander(f"Test set ({_n_test} molecules)", expanded=False):
                if _n_test > 0:
                    st.dataframe(_spl_res["test_df"], use_container_width=True, hide_index=True, height=300)
                else:
                    st.write("No test molecules (see warning above).")

        # Download buttons
        _dl_b1, _dl_b2, _dl_b3 = st.columns(3)
        with _dl_b1:
            st.download_button(
                "Download train set CSV",
                data=_spl_res["train_df"].to_csv(index=False),
                file_name=timestamp_filename("split_train"),
                mime="text/csv", key="spl_dl_train",
            )
        with _dl_b2:
            if _n_test > 0:
                st.download_button(
                    "Download test set CSV",
                    data=_spl_res["test_df"].to_csv(index=False),
                    file_name=timestamp_filename("split_test"),
                    mime="text/csv", key="spl_dl_test",
                )
        with _dl_b3:
            _combined = pd.concat([_spl_res["train_df"], _spl_res["test_df"]], ignore_index=True)
            st.download_button(
                "Download combined CSV",
                data=_combined.to_csv(index=False),
                file_name=timestamp_filename("split_combined"),
                mime="text/csv", key="spl_dl_combined",
            )

        st.caption(
            "**Why scaffold split?** Random splits allow train and test sets to share the same molecular "
            "scaffolds, inflating apparent model performance. Scaffold splits force generalization to "
            "new chemotypes, giving a more realistic estimate of how the model will perform on truly "
            "novel compounds. This is the recommended splitting strategy for QSAR benchmarking "
            "(Wu et al., MoleculeNet, 2018)."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: Cluster Analysis
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["active_tab"] == "clustering":
    render_dataset_status_bar()

    st.header("Cluster Analysis")
    st.write("Group molecules by fingerprint similarity to understand dataset structure, "
             "identify overrepresented chemical series, and select diverse subsets.")

    # ── Staleness check ──
    def _clust_data_hash(smiles_list):
        if not smiles_list:
            return None
        return (smiles_list[0], smiles_list[-1], len(smiles_list))

    # ── Check for active dataset ──
    _cl_ds = st.session_state.get("active_dataset")
    if not _cl_ds:
        render_empty_state(
            "🔬",
            "No active dataset. Preprocess molecules first to run cluster analysis.",
        )
        st.stop()

    _cl_mols = _cl_ds["mols"]
    _cl_smiles = _cl_ds["smiles"]
    _cl_current_hash = _clust_data_hash(_cl_smiles)

    # Staleness: clear old results if the dataset changed
    if "cluster_data_hash" in st.session_state and st.session_state["cluster_data_hash"] != _cl_current_hash:
        for k in ["cluster_results", "cluster_method", "cluster_pca"]:
            st.session_state.pop(k, None)
        st.warning("The preprocessed dataset has changed since the last clustering. Please re-run.")
    st.session_state["cluster_data_hash"] = _cl_current_hash

    # ── Settings ──
    _cl_c1, _cl_c2, _cl_c3, _cl_c4 = st.columns(4)
    with _cl_c1:
        _cl_method = st.radio("Clustering method", ["Butina", "Hierarchical"],
                               key="cl_method_radio",
                               help="Butina: distance-based, number of clusters emerges from the cutoff. "
                                    "Hierarchical: specify the number of clusters (or auto-select via silhouette score).")
    with _cl_c2:
        _cl_fp_type = st.selectbox("Fingerprint", ["morgan", "fcfp", "maccs", "topological",
                                    "atom_pair", "torsion", "avalon"], key="cl_fp_type",
                                    help="Fingerprint used to compute pairwise Tanimoto distances. Morgan is the most common choice.")
    with _cl_c3:
        _cl_radius = st.number_input("Radius", min_value=1, max_value=4, value=2, key="cl_radius",
                                     help="Circular fingerprint radius. Only applies to Morgan/FCFP. 2 (≈ECFP4) is standard.")
    with _cl_c4:
        _cl_nbits = st.selectbox("Bits", [512, 1024, 2048], index=2, key="cl_nbits",
                                 help="Fingerprint length. More bits = finer discrimination. 2048 is standard.")

    if _cl_method == "Butina":
        _cl_cutoff = st.slider("Distance cutoff", 0.1, 0.9, 0.4, 0.05, key="cl_cutoff",
                                help="Lower cutoff = tighter clusters (higher similarity required). "
                                     "A cutoff of 0.4 means molecules need Tanimoto similarity >= 0.6 to cluster together.")
    else:
        _cl_auto = st.checkbox("Auto-select number of clusters (silhouette score)", value=True, key="cl_auto_k",
                               help="When checked, the tool tries k=2..10 and picks the k with the highest silhouette score.")
        _cl_n_clusters = None
        if not _cl_auto:
            _cl_n_clusters = st.number_input("Number of clusters", min_value=2,
                                              max_value=min(50, len(_cl_mols)),
                                              value=min(5, len(_cl_mols)), key="cl_n_clusters",
                                              help="Fixed number of clusters to produce.")

    # Size warning
    if len(_cl_mols) > 5000:
        st.warning(f"Large dataset ({len(_cl_mols)} molecules) — distance matrix computation may take a minute. "
                   f"Consider using a higher cutoff or subsampling for faster results.")

    # ── Run clustering ──
    if st.button("Run Clustering", type="primary", key="cl_run"):
        with st.spinner("Computing distance matrix and clustering..."):
            if _cl_method == "Butina":
                _cl_result = cluster_butina(
                    _cl_mols, _cl_smiles,
                    fp_type=_cl_fp_type, radius=_cl_radius, n_bits=_cl_nbits,
                    cutoff=_cl_cutoff,
                )
            else:
                _cl_result = cluster_hierarchical(
                    _cl_mols, _cl_smiles,
                    fp_type=_cl_fp_type, radius=_cl_radius, n_bits=_cl_nbits,
                    n_clusters=_cl_n_clusters if not _cl_auto else None,
                )

            # Store only serializable data — no fingerprint objects
            st.session_state["cluster_results"] = {
                "assignments": _cl_result["assignments"],
                "cluster_sizes": _cl_result["cluster_sizes"],
                "representatives": _cl_result["representatives"],
                "n_clusters": _cl_result["n_clusters"],
                "n_singletons": _cl_result["n_singletons"],
                "silhouette_score": _cl_result.get("silhouette_score"),
                "method": _cl_result["method"],
                "params": _cl_result["params"],
            }
            st.session_state["cluster_method"] = _cl_method

            # Compute PCA
            _cl_ids = _cl_result["assignments"]["Cluster_ID"].tolist()
            if len(_cl_mols) >= 3:
                _cl_pca = compute_cluster_pca(_cl_mols, _cl_ids,
                                               fp_type=_cl_fp_type, radius=_cl_radius, n_bits=1024)
                st.session_state["cluster_pca"] = _cl_pca
            else:
                st.session_state.pop("cluster_pca", None)

        st.rerun()

    # ── Display results ──
    if "cluster_results" in st.session_state:
        _cl_res = st.session_state["cluster_results"]

        # ── Cluster Summary ──
        render_provenance_caption()
        section_banner("Cluster Summary")
        _cl_n = len(_cl_res["assignments"])
        _cl_nc = _cl_res["n_clusters"]
        _cl_largest = int(_cl_res["cluster_sizes"]["Size"].iloc[0]) if _cl_nc > 0 else 0
        _cl_sil = _cl_res.get("silhouette_score")

        _cl_summary_rows = [
            ("Method", st.session_state.get("cluster_method", "")),
            ("Total molecules", str(_cl_n)),
            ("Number of clusters", str(_cl_nc)),
            ("Largest cluster", str(_cl_largest)),
            ("Singleton clusters", str(_cl_res["n_singletons"])),
        ]
        if _cl_sil is not None:
            _cl_summary_rows.append(("Silhouette score", f"{_cl_sil:.3f}"))
        if "params" in _cl_res:
            _p = _cl_res["params"]
            _cl_summary_rows.append(("Fingerprint", f"{_p.get('fp_type', '')} (r={_p.get('radius', '')}, {_p.get('n_bits', '')} bits)"))
            if "cutoff" in _p:
                _cl_summary_rows.append(("Distance cutoff", f"{_p['cutoff']} (similarity >= {1 - _p['cutoff']:.1f})"))

        compact_table(_cl_summary_rows)

        # Warnings for degenerate cases
        if _cl_nc == _cl_n and _cl_nc > 1:
            st.warning("Every molecule is its own cluster (all singletons). Try increasing the distance cutoff "
                       "to allow more molecules to group together.")
        elif _cl_nc == 1 and _cl_n > 1:
            st.warning("All molecules fall into a single cluster. Try decreasing the distance cutoff "
                       "for finer grouping, or use hierarchical clustering with more clusters.")

        # ── Cluster Size Distribution ──
        if _cl_nc > 1:
            section_banner("Cluster Size Distribution")
            import plotly.graph_objects as go

            _cs_df = _cl_res["cluster_sizes"]
            _cs_labels = [f"Cluster {row['Cluster_ID']}" for _, row in _cs_df.iterrows()]
            _cs_sizes = _cs_df["Size"].tolist()

            _cs_fig = go.Figure(go.Bar(
                x=_cs_labels, y=_cs_sizes,
                marker_color="#1f77b4",
                text=_cs_sizes, textposition="outside",
            ))
            _cs_fig.update_layout(
                height=350,
                margin=dict(l=50, r=20, t=30, b=60),
                xaxis=dict(title=dict(text="Cluster", font=dict(size=13, color="#333")),
                           tickangle=45 if _cl_nc > 10 else 0,
                           tickfont=dict(size=10 if _cl_nc > 15 else 12, color="#333")),
                yaxis=dict(title=dict(text="Number of molecules", font=dict(size=13, color="#333")),
                           tickfont=dict(color="#333")),
                paper_bgcolor="white", plot_bgcolor="white",
            )
            st.plotly_chart(_cs_fig, use_container_width=True)
            st.caption("A skewed distribution (one or two dominant clusters) indicates the dataset is "
                       "dominated by a few chemical series. A uniform distribution suggests higher "
                       "chemical diversity. Many singletons may indicate the cutoff is too tight or "
                       "the dataset is structurally diverse.")

        # ── Cluster Representatives ──
        section_banner("Cluster Representatives")
        _cl_reps = _cl_res["representatives"]

        # Show top 5 with structure images
        _n_show = min(5, len(_cl_reps))
        for _ri in range(_n_show):
            _rrow = _cl_reps.iloc[_ri]
            _rcid = _rrow["Cluster_ID"]
            _rsize = _rrow["Size"]
            _rsmi = _rrow["SMILES"]
            with st.expander(f"Cluster {_rcid} — {_rsize} molecule(s)", expanded=(_ri < 3)):
                _r_col1, _r_col2 = st.columns([1, 3])
                with _r_col1:
                    _rmol = Chem.MolFromSmiles(_rsmi)
                    if _rmol:
                        st.image(mol_to_image(_rmol, size=(200, 200)),
                                 caption="Representative", width=200)
                with _r_col2:
                    st.code(_rsmi, language=None)
                    # Show other members of this cluster
                    _cmembers = _cl_res["assignments"][
                        _cl_res["assignments"]["Cluster_ID"] == _rcid
                    ]["SMILES"].tolist()
                    if len(_cmembers) > 1:
                        _others = [s for s in _cmembers if s != _rsmi]
                        st.write(f"Other members ({len(_others)}):")
                        st.dataframe(pd.DataFrame({"SMILES": _others}),
                                     use_container_width=True, hide_index=True, height=150)

        if len(_cl_reps) > _n_show:
            with st.expander(f"View all {len(_cl_reps)} cluster representatives"):
                st.dataframe(_cl_reps, use_container_width=True, hide_index=True)

        # ── Chemical Space PCA ──
        if "cluster_pca" in st.session_state:
            section_banner("Chemical Space (PCA)")
            import plotly.graph_objects as go
            import plotly.express as px

            _cpca = st.session_state["cluster_pca"]
            _cpca_df = _cpca["pca_df"]
            _cvr = _cpca["var_ratio"]
            _n_unique_clusters = _cpca_df["Cluster_ID"].nunique()

            _cfig = go.Figure()

            if _n_unique_clusters <= 10:
                _ccolors = px.colors.qualitative.Plotly
                for ci, cid in enumerate(sorted(_cpca_df["Cluster_ID"].unique())):
                    _csub = _cpca_df[_cpca_df["Cluster_ID"] == cid]
                    _cfig.add_trace(go.Scatter(
                        x=_csub["PC1"], y=_csub["PC2"],
                        mode="markers",
                        marker=dict(size=8, color=_ccolors[ci % len(_ccolors)], opacity=0.8),
                        name=f"Cluster {cid}",
                        text=_csub["SMILES"],
                        hovertemplate="<b>Cluster %{meta}</b><br>%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                        meta=[cid] * len(_csub),
                    ))
            else:
                _cfig.add_trace(go.Scatter(
                    x=_cpca_df["PC1"], y=_cpca_df["PC2"],
                    mode="markers",
                    marker=dict(size=8, color=_cpca_df["Cluster_ID"], colorscale="Turbo",
                                opacity=0.8, showscale=True,
                                colorbar=dict(title="Cluster")),
                    text=_cpca_df["SMILES"],
                    customdata=_cpca_df["Cluster_ID"],
                    hovertemplate="<b>Cluster %{customdata}</b><br>%{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
                ))

            _cfig.update_layout(
                xaxis=dict(title=dict(text=f"PC1 ({_cvr[0]*100:.1f}% var.)", font=dict(size=13, color="#333")),
                           tickfont=dict(color="#333"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
                yaxis=dict(title=dict(text=f"PC2 ({_cvr[1]*100:.1f}% var.)", font=dict(size=13, color="#333")),
                           tickfont=dict(color="#333"), gridcolor="rgba(0,0,0,0.05)", zeroline=False),
                height=500, width=700,
                margin=dict(l=60, r=30, t=30, b=60),
                paper_bgcolor="white", plot_bgcolor="white",
                legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.8)",
                            font=dict(size=10, color="#333")),
            )
            st.plotly_chart(_cfig, use_container_width=True)
            st.caption("Each point is a molecule projected into 2D chemical space via PCA on Morgan fingerprints. "
                       "Points colored by cluster assignment. Well-separated clusters in PCA space suggest "
                       "the clustering captures genuine chemical diversity. Overlapping clusters may indicate "
                       "the boundary between groups is gradual rather than sharp.")

        # ── Diverse Subset Selection ──
        section_banner("Diverse Subset Selection")
        _cl_max_div = min(_cl_nc, 20)
        _cl_div_n = st.number_input(
            "Number of diverse representatives to select",
            min_value=1, max_value=_cl_nc, value=_cl_max_div,
            key="cl_div_n",
            help="Selects one medoid (most central molecule) from each of the N largest clusters.",
        )
        _cl_div_df = _cl_reps.head(_cl_div_n)[["Cluster_ID", "Size", "SMILES"]].copy()
        st.write(f"**{len(_cl_div_df)}** representative molecules selected (one medoid per cluster, "
                 f"from the {len(_cl_div_df)} largest clusters).")
        st.dataframe(_cl_div_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download diverse subset CSV",
            data=_cl_div_df.to_csv(index=False),
            file_name=timestamp_filename("cluster_diverse_subset"),
            mime="text/csv", key="cl_dl_diverse",
        )
        st.caption("A diverse subset contains one representative molecule per cluster, maximizing "
                   "chemical coverage with minimal redundancy. Useful for screening library design, "
                   "selecting training set compounds, or prioritizing molecules for experimental testing.")

        # ── Full Cluster Assignments ──
        section_banner("Full Cluster Assignments")
        with st.expander(f"All {_cl_n} molecules with cluster assignments", expanded=False):
            st.dataframe(_cl_res["assignments"], use_container_width=True, hide_index=True, height=400)

        st.download_button(
            "Download full cluster assignments CSV",
            data=_cl_res["assignments"].to_csv(index=False),
            file_name=timestamp_filename("cluster_assignments"),
            mime="text/csv", key="cl_dl_full",
        )

        st.caption(
            "**Butina vs Hierarchical:** Butina clustering uses a distance cutoff — molecules within the "
            "cutoff are grouped together, producing a variable number of clusters that reflects natural "
            "groupings in the data. Hierarchical clustering produces a fixed number of clusters (user-specified "
            "or auto-selected via silhouette score), which is useful when you need a specific number of groups. "
            "Both use Tanimoto distance on molecular fingerprints."
        )
