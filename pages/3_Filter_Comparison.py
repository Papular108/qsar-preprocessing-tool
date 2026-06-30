import streamlit as st
import pandas as pd
from pipeline.preprocessing import run_preprocessing_pipeline
from pipeline.example_data import get_fda_approved_drugs, get_pains_demo_set

st.title("Filter Comparison")
st.write(
    "Run the preprocessing pipeline with two different filter configurations on the same "
    "molecule set and compare results side by side. PAINS filtering is always applied."
)

# ---------------------------------------------------------------------------
# SMILES input
# ---------------------------------------------------------------------------
st.header("Input molecules")

if "comparison_smiles" not in st.session_state:
    st.session_state["comparison_smiles"] = ""

_EXAMPLE_OPTIONS = {
    "FDA-approved drugs (20 molecules)": get_fda_approved_drugs,
    "PAINS-rich demo set (15 molecules)": get_pains_demo_set,
}

with st.expander("Try with example data", expanded=False):
    example_choice = st.selectbox(
        "Select a dataset",
        list(_EXAMPLE_OPTIONS.keys()),
        label_visibility="collapsed",
        key="comparison_example_choice",
    )
    smiles_ex, desc_ex = _EXAMPLE_OPTIONS[example_choice]()
    st.caption(desc_ex)
    if st.button("Load example data", key="comparison_load_example"):
        st.session_state["comparison_smiles"] = "\n".join(smiles_ex)
        st.rerun()

smiles_input = st.text_area(
    "Paste SMILES here (one per line)",
    key="comparison_smiles",
    height=150,
)

# ---------------------------------------------------------------------------
# Configuration columns
# ---------------------------------------------------------------------------
st.header("Filter configurations")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Configuration A")
    max_viol_a = st.number_input(
        "Max Lipinski violations",
        min_value=0, max_value=4, value=1, step=1,
        key="max_viol_a",
        help="Lipinski's Rule of Five — max violations allowed.",
    )
    st.caption("Additional filters:")
    enable_brenk_a  = st.checkbox("Brenk",  key="brenk_a")
    enable_veber_a  = st.checkbox("Veber",  key="veber_a")
    enable_ghose_a  = st.checkbox("Ghose",  key="ghose_a")
    enable_egan_a   = st.checkbox("Egan",   key="egan_a")
    enable_muegge_a = st.checkbox("Muegge", key="muegge_a")

with col_b:
    st.subheader("Configuration B")
    max_viol_b = st.number_input(
        "Max Lipinski violations",
        min_value=0, max_value=4, value=0, step=1,
        key="max_viol_b",
        help="Lipinski's Rule of Five — max violations allowed.",
    )
    st.caption("Additional filters:")
    enable_brenk_b  = st.checkbox("Brenk",  key="brenk_b",  value=True)
    enable_veber_b  = st.checkbox("Veber",  key="veber_b",  value=True)
    enable_ghose_b  = st.checkbox("Ghose",  key="ghose_b")
    enable_egan_b   = st.checkbox("Egan",   key="egan_b")
    enable_muegge_b = st.checkbox("Muegge", key="muegge_b")

# ---------------------------------------------------------------------------
# Compare button
# ---------------------------------------------------------------------------
if st.button("Compare", type="primary"):
    smiles_list = [
        line.strip()
        for line in smiles_input.splitlines()
        if line.strip()
    ]

    if not smiles_list:
        st.warning("Please paste at least one SMILES string or load an example dataset.")
        st.stop()

    with st.spinner("Running both pipelines…"):
        result_a = run_preprocessing_pipeline(
            smiles_list,
            lipinski_max_violations=max_viol_a,
            enable_brenk=enable_brenk_a,
            enable_veber=enable_veber_a,
            enable_ghose=enable_ghose_a,
            enable_egan=enable_egan_a,
            enable_muegge=enable_muegge_a,
        )
        result_b = run_preprocessing_pipeline(
            smiles_list,
            lipinski_max_violations=max_viol_b,
            enable_brenk=enable_brenk_b,
            enable_veber=enable_veber_b,
            enable_ghose=enable_ghose_b,
            enable_egan=enable_egan_b,
            enable_muegge=enable_muegge_b,
        )

    # -----------------------------------------------------------------------
    # Side-by-side summary
    # -----------------------------------------------------------------------
    st.header("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration A")
        kept_a   = len(result_a["kept_smiles"])
        removed_a = len(smiles_list) - kept_a
        st.metric("Kept",    kept_a)
        st.metric("Removed", removed_a)
        st.caption("Audit trail")
        st.dataframe(pd.DataFrame(result_a["audit_trail"]), use_container_width=True)

    with col2:
        st.subheader("Configuration B")
        kept_b   = len(result_b["kept_smiles"])
        removed_b = len(smiles_list) - kept_b
        st.metric("Kept",    kept_b)
        st.metric("Removed", removed_b)
        st.caption("Audit trail")
        st.dataframe(pd.DataFrame(result_b["audit_trail"]), use_container_width=True)

    # -----------------------------------------------------------------------
    # Set-difference analysis
    # -----------------------------------------------------------------------
    st.header("Molecules that differ between configurations")

    set_a = set(result_a["kept_smiles"])
    set_b = set(result_b["kept_smiles"])
    both  = set_a & set_b
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    st.write(
        f"**{len(both)}** molecules kept by both | "
        f"**{len(only_a)}** kept by A only | "
        f"**{len(only_b)}** kept by B only"
    )

    # Build per-SMILES lookup for each pipeline's removed_log
    removed_lookup_a = {e["smiles"]: e for e in result_a["removed_log"]}
    removed_lookup_b = {e["smiles"]: e for e in result_b["removed_log"]}

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Kept by A, removed by B")
        if only_a:
            rows = []
            for smi in only_a:
                info = removed_lookup_b.get(smi, {})
                rows.append({
                    "SMILES": smi,
                    "Removed at step": info.get("step", "—"),
                    "Reason": info.get("reason", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No molecules in this category.")

    with col4:
        st.subheader("Kept by B, removed by A")
        if only_b:
            rows = []
            for smi in only_b:
                info = removed_lookup_a.get(smi, {})
                rows.append({
                    "SMILES": smi,
                    "Removed at step": info.get("step", "—"),
                    "Reason": info.get("reason", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No molecules in this category.")
