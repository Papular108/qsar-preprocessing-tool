"""
Curated example datasets for first-time users to explore the pipeline
without uploading their own file.
"""


def get_fda_approved_drugs():
    """
    20 common FDA-approved small-molecule drugs spanning diverse therapeutic areas.
    All pass Lipinski's Rule of Five (≤1 violation). None are PAINS hits.

    Returns:
        tuple: (list of SMILES strings, description string)
    """
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",                                           # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                                      # Ibuprofen
        "CC(=O)Nc1ccc(O)cc1",                                               # Acetaminophen
        "CN(C)C(=N)NC(=N)N",                                                # Metformin
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                                      # Caffeine
        "CC(c1ccc2cccc(OC)c2c1)C(=O)O",                                    # Naproxen
        "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",                     # Ciprofloxacin
        "Cc1ncc([N+](=O)[O-])n1CCO",                                        # Metronidazole
        "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",                                  # Atenolol
        "COCCc1ccc(OCC(O)CNC(C)C)cc1",                                     # Metoprolol
        "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",               # Amlodipine
        "COc1ccc2[nH]c(CS(=O)c3ncc(C)c(OC)c3C)nc2c1",                     # Omeprazole
        "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",                        # Furosemide
        "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",                           # Warfarin
        "CC(C)NCC(O)COc1cccc2ccccc12",                                      # Propranolol
        "CC(CS)C(=O)N1CCCC1C(=O)O",                                         # Captopril
        "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",                             # Diazepam
        "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",                              # Fluoxetine
        "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3ccncc32)CC1",                    # Loratadine
        "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12", # Sildenafil
    ]
    description = "FDA-approved drugs (20 molecules) — common drugs spanning diverse therapeutic areas. All pass Lipinski's Rule of Five."
    return smiles, description


def get_pains_demo_set():
    """
    15 molecules: 8 clean drug-like compounds and 7 known PAINS scaffolds
    (rhodanines, catechol, quinones, azo, chalcone, anthraquinone).
    Designed to demonstrate the PAINS filter removing problematic structures.

    Returns:
        tuple: (list of SMILES strings, description string)
    """
    smiles = [
        # --- Clean drug-like molecules (expect to pass PAINS) ---
        "CC(=O)Nc1ccc(O)cc1",          # Acetaminophen
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine
        "CC(C)NCC(O)COc1cccc2ccccc12",  # Propranolol
        "CC(CS)C(=O)N1CCCC1C(=O)O",    # Captopril
        "OC(=O)c1ccc(N)cc1",           # 4-Aminobenzoic acid (PABA)
        "OC(=O)c1ccccc1",              # Benzoic acid
        "CC(C)(C)OC(=O)Nc1ccccc1",     # Boc-aniline
        # --- Known PAINS scaffolds (expect to be removed) ---
        "O=C1CSC(=S)N1",               # Rhodanine
        "O=C1/C(=C/c2ccccc2)SC(=S)N1", # Benzylidene rhodanine
        "Oc1ccccc1O",                   # Catechol
        "O=C1C=CC(=O)C=C1",            # para-Benzoquinone
        "O=C(/C=C/c1ccccc1)c1ccccc1",  # Chalcone (Michael acceptor)
        "O=C1c2ccccc2C(=O)c2ccccc21",  # Anthraquinone
        "Nc1ccc(/N=N/c2ccc(N)cc2)cc1", # 4,4'-Azodianiline (azo compound)
    ]
    description = "PAINS-rich demo set (15 molecules) — 8 clean molecules + 7 known PAINS scaffolds (rhodanines, catechol, quinone, chalcone, anthraquinone, azo). Run with default settings to see PAINS filtering in action."
    return smiles, description
