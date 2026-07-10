"""
Reusable UI components for the QSAR Preprocessing Tool.

All functions render Streamlit widgets directly. No computational logic lives here.
"""

import streamlit as st
from datetime import datetime

# ── Version (single source of truth) ──────────────────────────────────────────
VERSION = "1.0.0"


def timestamp_filename(prefix, extension="csv"):
    """Generate a download filename: <prefix>_<yyyymmdd_hhmmss>.<ext>"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension}"


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

    loaded_at = ds.get("loaded_at", "Unknown")
    if loaded_at and loaded_at != "Unknown (migrated)":
        try:
            dt = datetime.fromisoformat(loaded_at)
            delta = datetime.now() - dt
            if delta.total_seconds() < 60:
                time_str = "just now"
            elif delta.total_seconds() < 3600:
                time_str = f"{int(delta.total_seconds() // 60)} min ago"
            elif delta.total_seconds() < 86400:
                time_str = f"{int(delta.total_seconds() // 3600)}h ago"
            else:
                time_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            time_str = str(loaded_at)
    else:
        time_str = "unknown time"

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
            filters = []
            if config.get("lipinski_max_violations") is not None:
                filters.append(f"Lipinski (max {config['lipinski_max_violations']} violations)")
            filters.append("PAINS")  # always on
            for f in ["Veber", "Ghose", "Egan", "Muegge", "Brenk"]:
                key = f"enable_{f.lower()}"
                if config.get(key):
                    filters.append(f)
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
