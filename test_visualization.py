"""
Quick smoke test for pipeline/visualization.py.
Tests mol_to_image with 3 molecules and saves PNGs to /tmp for inspection.
"""
from rdkit import Chem
from pipeline.visualization import mol_to_image

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

test_cases = [
    ("Ethanol",    "CCO"),
    ("Aspirin",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("Rhodanine (PAINS hit)", "O=C1CSC(=S)N1"),
]

all_passed = True
for name, smiles in test_cases:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES for {name}"

    png_bytes = mol_to_image(mol, size=(300, 300))

    ok_magic  = png_bytes[:8] == PNG_MAGIC
    ok_size   = len(png_bytes) > 1000          # sanity: a real image has some content
    passed    = ok_magic and ok_size

    out_path = f"/tmp/test_viz_{name.split()[0].lower()}.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:30s}  bytes={len(png_bytes):6d}  magic_ok={ok_magic}  saved={out_path}")
    if not passed:
        all_passed = False

print()
print("All tests passed." if all_passed else "SOME TESTS FAILED.")
