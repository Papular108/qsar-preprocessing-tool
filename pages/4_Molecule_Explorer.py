"""
Molecule Explorer — interactive single-molecule profile card.
"""
import sys
from rdkit import Chem
from rdkit.Chem import RDConfig
sys.path.append(RDConfig.RDContribDir + "/SA_Score")
import sascorer

import streamlit as st
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

from pipeline.preprocessing import (
    check_lipinski, check_veber, check_ghose, check_egan, check_muegge,
    check_pains, check_brenk, compute_qed,
)
from pipeline.featurization import compute_fingerprint
from pipeline.visualization import mol_to_image

QUICK_PICKS = {
    "Aspirin":    "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine":   "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Ibuprofen":  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Metformin":  "CN(C)C(=N)NC(=N)N",
}

st.title("Molecule Explorer")
st.write("Enter a SMILES string to see a full physicochemical and druglikeness profile.")

# ── Quick-pick buttons ──────────────────────────────────────────────────────
cols = st.columns(len(QUICK_PICKS))
for col, (name, smi) in zip(cols, QUICK_PICKS.items()):
    if col.button(name, use_container_width=True):
        st.session_state["mol_explorer_smiles"] = smi
        st.rerun()

smiles_input = st.text_input(
    "SMILES",
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    key="mol_explorer_smiles",
)

analyze = st.button("Analyze", type="primary")

if not (analyze or smiles_input):
    st.stop()

if not smiles_input.strip():
    st.warning("Please enter a SMILES string.")
    st.stop()

mol = Chem.MolFromSmiles(smiles_input.strip())
if mol is None:
    st.error(f"Could not parse SMILES: `{smiles_input.strip()}`")
    st.stop()

# ── Section 1: Structure & Identifiers ──────────────────────────────────────
st.divider()
st.subheader("Structure & Identifiers")

img_col, id_col = st.columns([1, 2])

with img_col:
    png = mol_to_image(mol, size=(400, 400))
    st.image(png, use_container_width=True)

with id_col:
    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    inchi = MolToInchi(mol)
    inchikey = InchiToInchiKey(inchi) if inchi else "N/A"

    st.markdown("**Canonical SMILES**")
    st.code(canonical, language=None)
    st.markdown("**Molecular Formula**")
    st.code(formula, language=None)
    st.markdown("**InChI**")
    st.code(inchi or "N/A", language=None)
    st.markdown("**InChIKey**")
    st.code(inchikey, language=None)

# ── Section 2: Physicochemical Properties ───────────────────────────────────
st.divider()
st.subheader("Physicochemical Properties")

mw       = Descriptors.MolWt(mol)
logp     = Descriptors.MolLogP(mol)
tpsa     = Descriptors.TPSA(mol)
hbd      = Descriptors.NumHDonors(mol)
hba      = Descriptors.NumHAcceptors(mol)
rotb     = Descriptors.NumRotatableBonds(mol)
arom     = Descriptors.NumAromaticRings(mol)
fcsp3    = rdMolDescriptors.CalcFractionCSP3(mol)
heavy    = mol.GetNumHeavyAtoms()
rings    = mol.GetRingInfo().NumRings()

props = [
    ("MW (g/mol)",        f"{mw:.2f}"),
    ("LogP",              f"{logp:.2f}"),
    ("TPSA (Å²)",         f"{tpsa:.1f}"),
    ("HBD",               str(hbd)),
    ("HBA",               str(hba)),
    ("Rotatable Bonds",   str(rotb)),
    ("Aromatic Rings",    str(arom)),
    ("Fraction Csp3",     f"{fcsp3:.3f}"),
    ("Heavy Atom Count",  str(heavy)),
    ("Ring Count",        str(rings)),
]

row1, row2 = props[:5], props[5:]
cols_a = st.columns(5)
cols_b = st.columns(5)

for col, (label, value) in zip(cols_a, row1):
    col.metric(label, value)

for col, (label, value) in zip(cols_b, row2):
    col.metric(label, value)

# ── Section 3: Druglikeness Rules ───────────────────────────────────────────
st.divider()
st.subheader("Druglikeness Rules")

rules = []

lip_pass, _, lip_reason = check_lipinski(mol, max_violations=1)
rules.append(("Lipinski (≤1 violation)", lip_pass, lip_reason or ""))

veb_pass, _, veb_reason = check_veber(mol)
rules.append(("Veber", veb_pass, veb_reason or ""))

gho_pass, _, gho_reason = check_ghose(mol)
rules.append(("Ghose", gho_pass, gho_reason or ""))

ega_pass, _, ega_reason = check_egan(mol)
rules.append(("Egan", ega_pass, ega_reason or ""))

mue_pass, _, mue_reason = check_muegge(mol)
rules.append(("Muegge", mue_pass, mue_reason or ""))

rule_rows = []
for rule_name, passes, reason in rules:
    icon = "Pass" if passes else "Fail"
    rule_rows.append({"Rule": rule_name, "Result": icon, "Detail": reason})

import pandas as pd
rules_df = pd.DataFrame(rule_rows)

def _style_result(val):
    if val == "Pass":
        return "color: green; font-weight: bold"
    if val == "Fail":
        return "color: red; font-weight: bold"
    return ""

st.dataframe(
    rules_df.style.applymap(_style_result, subset=["Result"]),
    use_container_width=True,
    hide_index=True,
)

# ── Section 4: Scores & Alerts ───────────────────────────────────────────────
st.divider()
st.subheader("Scores & Alerts")

score_l, score_r = st.columns(2)

# QED
qed_val, qed_err = compute_qed(mol)
with score_l:
    st.markdown("**QED (Quantitative Estimate of Druglikeness)**")
    if qed_err:
        st.error(qed_err)
    else:
        st.write(f"Score: **{qed_val:.3f}** (0 = least drug-like, 1 = most)")
        st.progress(float(qed_val))
        if qed_val >= 0.67:
            st.caption("High druglikeness")
        elif qed_val >= 0.34:
            st.caption("Moderate druglikeness")
        else:
            st.caption("Low druglikeness")

# SA Score
try:
    sa_val = sascorer.calculateScore(mol)
    sa_err = None
except Exception as e:
    sa_val = None
    sa_err = str(e)

with score_r:
    st.markdown("**SA Score (Synthetic Accessibility)**")
    if sa_err:
        st.error(sa_err)
    else:
        st.write(f"Score: **{sa_val:.2f}** (1 = easy, 10 = very difficult)")
        # Invert so bar fills more for easy-to-make molecules
        st.progress(float(1.0 - (sa_val - 1) / 9.0))
        if sa_val <= 3:
            st.caption("Easy to synthesize")
        elif sa_val <= 6:
            st.caption("Moderate synthetic difficulty")
        else:
            st.caption("Difficult to synthesize")

# PAINS and Brenk
alert_l, alert_r = st.columns(2)

is_pains, pains_name = check_pains(mol)
with alert_l:
    st.markdown("**PAINS Filter**")
    if is_pains:
        st.error(f"PAINS hit: {pains_name}")
    else:
        st.success("No PAINS alerts detected")

is_brenk, brenk_name = check_brenk(mol)
with alert_r:
    st.markdown("**Brenk Filter**")
    if is_brenk:
        st.warning(f"Brenk alert: {brenk_name}")
    else:
        st.success("No Brenk alerts detected")

# ── Section 5: Fingerprint Preview ──────────────────────────────────────────
st.divider()
st.subheader("Fingerprint Preview (Morgan, radius=2, 2048 bits)")

fp_arr, fp_err = compute_fingerprint(mol, fp_type="morgan", radius=2, n_bits=2048)

if fp_err:
    st.error(fp_err)
else:
    bits_on = int(fp_arr.sum())
    density = bits_on / 2048

    c1, c2, c3 = st.columns(3)
    c1.metric("Bits ON", bits_on)
    c2.metric("Bits OFF", 2048 - bits_on)
    c3.metric("Bit Density", f"{density:.1%}")

    st.markdown("**Bit density**")
    st.progress(density)
    st.caption(
        "Bit density reflects how structurally rich the fingerprint is. "
        "Values around 3–10% are typical for drug-like molecules with Morgan (r=2, 2048 bits)."
    )
