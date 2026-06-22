# QSAR Preprocessing Tool

A web-based tool for preprocessing and featurizing molecules for QSAR (Quantitative Structure-Activity Relationship) modeling and virtual screening workflows. Built for cheminformatics students and researchers who need transparent, auditable, reproducible molecule preprocessing without writing custom scripts for every project.

**Live demo:** https://qsar-preprocessing-tool.streamlit.app/

## What it does

- **Preprocessing**: SMILES parsing, MolVS standardization, salt stripping, deduplication
- **Druglikeness filters** (all toggleable): Lipinski's Rule of Five, PAINS, Brenk structural alerts, Veber, Ghose, Egan, Muegge
- **Synthetic Accessibility (SA) scoring**
- **Featurization**: physicochemical descriptors (MW, LogP, TPSA, etc.) and 6 fingerprint types (Morgan/ECFP, MACCS, topological, atom pair, torsion, Avalon)
- **Full audit trail**: every removed molecule is logged with the exact step and reason it was removed
- **Reproducibility metadata**: every exported CSV includes the settings used, RDKit version, and timestamp
- **Flexible input**: accepts `.txt`, `.csv`, and `.xlsx` files, with automatic SMILES column detection (and manual fallback for non-standard column names)

## Tech stack

- **Backend logic**: Python, RDKit, MolVS, pandas, numpy
- **Frontend/UI**: Streamlit
- **Deployment**: Streamlit Community Cloud

## Running locally

```bash
git clone git@github.com:Papular108/qsar-preprocessing-tool.git
cd qsar-preprocessing-tool
conda create -n qsar-tool python=3.11 -y
conda activate qsar-tool
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
streamlit run app.py
```

## Project structure


qsar-preprocessing-tool/
- app.py
- pages/
  - 1_About_Methodology.py
  - 2_FAQ_Limitations.py
- pipeline/
  - preprocessing.py
  - featurization.py
- requirements.txt

## Limitations

This tool performs 2D-based preprocessing and featurization only. It does not perform 3D conformer generation, docking, or ADMET prediction. See the in-app FAQ/Limitations page for full details.

## References

- Landrum, G. RDKit: Open-source cheminformatics.
- Lipinski, C. A., et al. (2001). Advanced Drug Delivery Reviews, 46(1-3), 3-26.
- Baell, J. B., & Holloway, G. A. (2010). Journal of Medicinal Chemistry, 53(7), 2719-2740.
- Rogers, D., & Hahn, M. (2010). Journal of Chemical Information and Modeling, 50(5), 742-754.
- Ertl, P., & Schuffenhauer, A. (2009). Journal of Cheminformatics, 1(1), 8.

## License

Open source, available for the cheminformatics community to use and learn from.
