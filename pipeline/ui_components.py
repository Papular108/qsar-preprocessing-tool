"""
Reusable UI components for the QSAR Preprocessing Tool.

All functions render Streamlit widgets directly. No computational logic lives here.
"""

import streamlit as st
from io import BytesIO
from datetime import datetime

# ── Version (single source of truth) ──────────────────────────────────────────
VERSION = "1.0.1"


def timestamp_filename(prefix, extension="csv"):
    """Generate a download filename: <prefix>_<yyyymmdd_hhmmss>.<ext>"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension}"


def _format_relative_time(loaded_at):
    """Format a loaded_at ISO timestamp as a human-readable relative time."""
    if not loaded_at or loaded_at == "Unknown (migrated)":
        return "unknown time"
    try:
        dt = datetime.fromisoformat(loaded_at)
        delta = datetime.now() - dt
        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() // 60)} min ago"
        elif delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() // 3600)}h ago"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(loaded_at)


def _format_filter_list(config):
    """Build a list of applied filter names from a preprocessing config dict."""
    filters = []
    if config.get("lipinski_max_violations") is not None:
        filters.append(f"Lipinski (max {config['lipinski_max_violations']} violations)")
    filters.append("PAINS")  # always on
    for f in ["Veber", "Ghose", "Egan", "Muegge", "Brenk"]:
        if config.get(f"enable_{f.lower()}"):
            filters.append(f)
    return filters


def render_sidebar_data_panel():
    """Persistent sidebar showing dataset state + data loading controls."""
    from pipeline.example_data import get_fda_approved_drugs, get_pains_demo_set

    ds = st.session_state.get("active_dataset")
    has_staged_file = "sidebar_staged_file" in st.session_state
    has_staged_smiles = bool(st.session_state.get("pasted_smiles", "").strip())

    # ── Dataset status ──
    if ds is not None:
        st.sidebar.markdown("### Active Dataset")
        filename = ds.get("source_filename", "Unknown")
        n_mols = ds.get("n_molecules", 0)
        n_orig = ds.get("n_original", 0)
        time_str = _format_relative_time(ds.get("loaded_at"))
        config = ds.get("preprocessing_config", {})
        filters = _format_filter_list(config)

        st.sidebar.markdown(
            f"**{filename}**\n\n"
            f"{n_mols:,} molecules (from {n_orig:,} input)\n\n"
            f"Preprocessed {time_str}\n\n"
            f"Filters: {', '.join(filters)}"
        )
        if st.sidebar.button("Clear dataset", key="sidebar_clear_ds"):
            for k in ["active_dataset", "kept_mols_for_featurization",
                       "kept_smiles_for_featurization", "pipeline_result",
                       "pipeline_settings", "pipeline_ts",
                       "split_results", "split_data_hash", "split_pca_current",
                       "split_pca_random", "cluster_results", "cluster_data_hash",
                       "cluster_pca", "scr_results", "featurization_result",
                       "sidebar_staged_file", "pasted_smiles"]:
                st.session_state.pop(k, None)
            st.rerun()
    elif has_staged_file or has_staged_smiles:
        st.sidebar.warning("Data staged — go to Preprocessing to configure filters and run pipeline.")
    else:
        st.sidebar.info("No dataset loaded yet.")

    # ── Load / change data expander ──
    with st.sidebar.expander("Load / change data"):
        # File uploader
        sidebar_file = st.file_uploader(
            "Upload file", type=["txt", "csv", "xlsx"],
            key="sidebar_file_uploader",
        )
        if sidebar_file is not None:
            file_bytes = sidebar_file.read()
            sidebar_file.seek(0)
            if (st.session_state.get("sidebar_staged_file", {}).get("name") != sidebar_file.name):
                st.session_state["sidebar_staged_file"] = {
                    "name": sidebar_file.name,
                    "bytes": file_bytes,
                }
                st.rerun()

        # Example dataset
        st.markdown("---")
        _SIDEBAR_EXAMPLES = {
            "FDA-approved drugs (20 molecules)": get_fda_approved_drugs,
            "PAINS-rich demo set (15 molecules)": get_pains_demo_set,
        }
        sidebar_example = st.selectbox(
            "Example dataset",
            list(_SIDEBAR_EXAMPLES.keys()),
            key="sidebar_example_select",
            label_visibility="collapsed",
        )
        if st.button("Load example", key="sidebar_load_example"):
            smiles_list, _ = _SIDEBAR_EXAMPLES[sidebar_example]()
            st.session_state["pasted_smiles"] = "\n".join(smiles_list)
            st.session_state["_example_source"] = sidebar_example
            st.rerun()

        # Paste SMILES
        st.markdown("---")
        sidebar_smiles = st.text_area(
            "Paste SMILES (one per line)",
            key="sidebar_paste_smiles",
            height=100,
        )
        if st.button("Stage SMILES", key="sidebar_stage_smiles"):
            if sidebar_smiles.strip():
                st.session_state["pasted_smiles"] = sidebar_smiles.strip()
                st.rerun()

    # ── Citation ──
    with st.sidebar.expander("Cite this tool"):
        st.code(
            f"""@software{{qsar_preprocessing_tool,
  author = {{Papular108}},
  title = {{QSAR Preprocessing Tool: An open-source cheminformatics platform}},
  url = {{https://github.com/Papular108/qsar-preprocessing-tool}},
  year = {{2026}},
  version = {{{VERSION}}}
}}""",
            language="bibtex",
        )

    # ── Resources ──
    st.sidebar.markdown("##### Resources")
    st.sidebar.page_link("pages/1_📖_About_Methodology.py", label="📖 Methodology")
    st.sidebar.page_link("pages/2_❓_FAQ_Limitations.py", label="❓ FAQ & Limitations")


def render_dataset_status_bar():
    """
    Persistent status bar showing the active dataset.
    Call once, just below the nav, on every tab except Preprocessing.
    """
    ds = st.session_state.get("active_dataset")
    if ds is None:
        st.markdown(
            '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;'
            'padding:10px 16px;margin-bottom:16px;font-size:0.9rem;color:#856404;">'
            'No active dataset. Go to <strong>Preprocessing</strong> to load and filter molecules.'
            '</div>',
            unsafe_allow_html=True,
        )
        return False  # caller can use this to short-circuit

    time_str = _format_relative_time(ds.get("loaded_at", "Unknown"))

    filename = ds.get("source_filename", "Unknown")
    n_mols = ds.get("n_molecules", 0)

    st.markdown(
        f'<div style="background:#e8f5e9;border:1px solid #4caf50;border-radius:6px;'
        f'padding:10px 16px;margin-bottom:16px;font-size:0.9rem;color:#2e7d32;">'
        f'Active dataset: <strong>{filename}</strong> &nbsp;|&nbsp; '
        f'<strong>{n_mols:,}</strong> molecules &nbsp;|&nbsp; '
        f'Preprocessed {time_str}'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("View provenance"):
        config = ds.get("preprocessing_config", {})
        n_orig = ds.get("n_original", 0)
        st.markdown(f"**Source:** {filename}")
        st.markdown(f"**Uploaded:** {loaded_at}")
        st.markdown(f"**Input molecules:** {n_orig:,}")
        st.markdown(f"**After preprocessing:** {n_mols:,}")
        if config:
            filters = _format_filter_list(config)
            st.markdown(f"**Filters applied:** {', '.join(filters)}")
            dedup = config.get("deduplication", "canonical_smiles")
            st.markdown(f"**Deduplication:** {dedup}")

    return True  # dataset is available


def render_empty_state(icon, message, action_label="Go to Preprocessing",
                       action_tab="preprocessing", action_category="data",
                       secondary_label=None, secondary_tab=None,
                       secondary_category=None):
    """
    Standardized empty state: large icon, message, action button.
    Returns True if the action button was clicked (caller should st.rerun()).
    """
    st.markdown(
        f'<div style="text-align:center;padding:60px 20px;color:#999;">'
        f'<div style="font-size:4rem;margin-bottom:16px;">{icon}</div>'
        f'<p style="font-size:1.1rem;color:#666;max-width:500px;margin:0 auto 24px;">'
        f'{message}</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1]) if secondary_label else st.columns([1, 1, 1])
    with cols[1]:
        if st.button(action_label, type="primary", use_container_width=True,
                      key=f"empty_{action_tab}_goto"):
            st.session_state["active_category"] = action_category
            st.session_state["active_tab"] = action_tab
            st.rerun()
    if secondary_label and secondary_tab:
        with cols[1]:
            if st.button(secondary_label, use_container_width=True,
                          key=f"empty_{secondary_tab}_goto"):
                st.session_state["active_category"] = secondary_category or "data"
                st.session_state["active_tab"] = secondary_tab
                st.rerun()


def render_provenance_caption(ds=None):
    """
    Small 'Based on:' caption from the active dataset.
    Call at the top of results sections.
    """
    if ds is None:
        ds = st.session_state.get("active_dataset")
    if ds is None:
        return

    filename = ds.get("source_filename", "Unknown")
    n_mols = ds.get("n_molecules", 0)
    config = ds.get("preprocessing_config", {})

    filters = []
    if config.get("lipinski_max_violations") is not None:
        filters.append(f"Lipinski (≤{config['lipinski_max_violations']} violations)")
    filters.append("PAINS")
    for f in ["Veber", "Ghose", "Egan", "Muegge", "Brenk"]:
        if config.get(f"enable_{f.lower()}"):
            filters.append(f)

    filter_str = ", ".join(filters) if filters else "default"
    st.caption(
        f"Based on {n_mols:,} molecules from {filename}, "
        f"preprocessed with {filter_str}."
    )


def render_next_step_hint(hint_key, message, suggestions):
    """
    Dismissable next-step hint after a major action.
    hint_key: unique key per hint type (e.g., 'preprocessing_complete')
    suggestions: list of (label, tab_key, category_key) tuples
    """
    dismiss_key = f"hint_dismissed_{hint_key}"
    if st.session_state.get(dismiss_key):
        return

    cols = st.columns([6, 1])
    with cols[0]:
        st.info(message)
        for label, tab_key, cat_key in suggestions:
            if st.button(label, key=f"hint_{hint_key}_{tab_key}"):
                st.session_state["active_category"] = cat_key
                st.session_state["active_tab"] = tab_key
                st.rerun()
    with cols[1]:
        if st.button("Dismiss", key=f"hint_{hint_key}_dismiss"):
            st.session_state[dismiss_key] = True
            st.rerun()


def render_footer():
    """Unobtrusive footer for every tab."""
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;color:#999;font-size:0.8rem;padding:8px 0;">'
        f'QSAR Preprocessing Tool v{VERSION} &nbsp;·&nbsp; '
        f'<a href="https://github.com/Papular108/qsar-preprocessing-tool/issues" '
        f'style="color:#999;">Report a bug</a> &nbsp;·&nbsp; '
        f'<a href="#" onclick="return false;" style="color:#999;" '
        f'title="Copy: QSAR Preprocessing Tool (2026). '
        f'https://github.com/Papular108/qsar-preprocessing-tool">Cite this tool</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


def migrate_legacy_session_state():
    """
    One-time migration: old session state keys → active_dataset dict.
    Safe to call on every app run — only acts when old keys exist
    and active_dataset does not.
    """
    if "active_dataset" in st.session_state:
        return  # already migrated or freshly set

    if "kept_mols_for_featurization" not in st.session_state:
        return  # nothing to migrate

    mols = st.session_state["kept_mols_for_featurization"]
    smiles = list(st.session_state.get("kept_smiles_for_featurization", []))
    pipeline_result = st.session_state.get("pipeline_result")
    settings = st.session_state.get("pipeline_settings", {})

    st.session_state["active_dataset"] = {
        "source_filename": "Unknown (migrated from previous session)",
        "loaded_at": "Unknown (migrated)",
        "n_original": pipeline_result.get("input_count", 0) if pipeline_result else 0,
        "n_molecules": len(mols),
        "mols": mols,
        "smiles": smiles,
        "preprocessing_config": settings,
        "pipeline_result": pipeline_result,
        "label_map": st.session_state.get("pipeline_label_map"),
        "pval_map": st.session_state.get("pipeline_pval_map"),
        "pchembl_map": st.session_state.get("pipeline_pchembl_map"),
    }
