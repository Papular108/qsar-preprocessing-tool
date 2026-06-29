# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the app locally:**
```bash
streamlit run app.py
```

**Install dependencies** (requires conda with RDKit installed separately):
```bash
conda create -n qsar-tool python=3.11 -y
conda activate qsar-tool
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

There are no automated tests or linting configured in this project.

## Architecture

This is a single-page Streamlit app (`app.py`) with a two-stage workflow: **preprocessing** then **featurization**. The pipeline modules are pure Python with no Streamlit dependencies, so they can be called independently of the UI.

**Data flow:**
1. User uploads `.txt`/`.csv`/`.xlsx` or pastes SMILES strings in `app.py`
2. `app.py` calls `pipeline/preprocessing.py:run_preprocessing_pipeline()` which runs molecules through a fixed sequential pipeline: parse → standardize (MolVS) → strip salts → Lipinski filter → PAINS filter → optional filters (Brenk, Veber, Ghose, Egan, Muegge) → deduplicate
3. Surviving RDKit `Mol` objects are stored in `st.session_state["kept_mols_for_featurization"]`
4. User then triggers `pipeline/featurization.py:featurize_dataset()` which computes physicochemical descriptors + one fingerprint type per molecule, returning a combined DataFrame
5. Both outputs include a reproducibility metadata header (RDKit version, settings, timestamp) prepended to the downloaded CSV

**Key design points:**
- `run_preprocessing_pipeline()` returns an audit trail (per-step counts) and a removed log (per-molecule reason), both surfaced in the UI
- SA score is computed post-pipeline and is informational only — it does not filter molecules
- SMILES column auto-detection checks for `canonical_smiles`, `Smiles`, `SMILES`, `smiles` in that order; falls back to manual `st.selectbox`
- MACCS fingerprints are always 167 bits (fixed); all other fingerprint types accept 512/1024/2048 bits
- `pipeline/__init__.py` and `utils/__init__.py` are empty stubs
- `pages/` contains two static informational Streamlit pages (methodology and FAQ) with no logic
- `data/` directory is a placeholder (`.gitkeep` only)
