"""
Generate publication-ready Methods section text from pipeline settings.
"""

_FP_SENTENCES = {
    "morgan": (
        "Molecular fingerprints were generated using Morgan circular fingerprints "
        "(Rogers & Hahn, 2010) with radius {radius} and {n_bits} bits."
    ),
    "maccs": (
        "Molecular fingerprints were generated using MACCS structural keys "
        "(166-bit fixed-length key descriptors) as implemented in RDKit."
    ),
    "topological": (
        "Molecular fingerprints were generated using RDKit topological (Daylight-style) "
        "fingerprints with {n_bits} bits."
    ),
    "atom_pair": (
        "Molecular fingerprints were generated using atom pair fingerprints "
        "(Carhart et al., 1985) hashed to {n_bits} bits."
    ),
    "torsion": (
        "Molecular fingerprints were generated using topological torsion fingerprints "
        "(Nilakantan et al., 1987) hashed to {n_bits} bits."
    ),
    "avalon": (
        "Molecular fingerprints were generated using Avalon fingerprints "
        "(Gedeck et al., 2006) with {n_bits} bits."
    ),
}


def generate_methods_text(settings):
    """
    Generate a publication-ready Methods paragraph from pipeline settings.

    Parameters:
        settings (dict): keys expected:
            lipinski_max_violations (int)
            enable_veber (bool)
            enable_ghose (bool)
            enable_egan (bool)
            enable_muegge (bool)
            enable_brenk (bool)
            enable_sa_score (bool)
            fp_type (str): morgan | maccs | topological | atom_pair | torsion | avalon
            n_bits (int)
            radius (int)
            input_count (int, optional)
            kept_count (int, optional)

    Returns:
        str: publication-ready paragraph
    """
    sentences = []

    # Opening counts line (optional)
    input_count = settings.get("input_count")
    kept_count = settings.get("kept_count")
    if input_count is not None and kept_count is not None:
        sentences.append(
            f"A dataset of {input_count} compounds was subjected to the following "
            f"preprocessing workflow, yielding {kept_count} compounds for further analysis."
        )

    # Standardization and salt stripping (always applied)
    sentences.append(
        "Compounds were standardized using MolVS (https://github.com/mcs07/MolVS) "
        "and desalted using RDKit's SaltRemover module "
        "(Landrum, G., RDKit: Open-source cheminformatics)."
    )

    # Lipinski (always applied)
    max_viol = settings.get("lipinski_max_violations", 1)
    if max_viol == 0:
        viol_clause = "strict compliance (zero violations allowed)"
    elif max_viol == 1:
        viol_clause = "a maximum of 1 violation allowed"
    else:
        viol_clause = f"a maximum of {max_viol} violations allowed"
    sentences.append(
        f"Druglikeness was assessed using Lipinski's Rule of Five "
        f"(Lipinski et al., 2001), with {viol_clause}."
    )

    # PAINS (always applied)
    sentences.append(
        "Compounds matching Pan-Assay Interference Compound (PAINS) substructure "
        "patterns were removed using RDKit's FilterCatalog "
        "(Baell & Holloway, 2010)."
    )

    # Optional filters
    if settings.get("enable_brenk"):
        sentences.append(
            "Structural alerts for reactive or otherwise undesirable functional groups "
            "were flagged and removed using the Brenk filter "
            "(Brenk et al., 2008) via RDKit's FilterCatalog."
        )

    if settings.get("enable_veber"):
        sentences.append(
            "Oral bioavailability was further assessed using Veber's rules "
            "(Veber et al., 2002): compounds with more than 10 rotatable bonds "
            "or a topological polar surface area (TPSA) exceeding 140 \u00c5\u00b2 were removed."
        )

    if settings.get("enable_ghose"):
        sentences.append(
            "The Ghose filter (Ghose et al., 1999) was applied to retain only compounds "
            "within drug-like property ranges: MW 160\u2013480 Da, "
            "\u22120.4 \u2264 logP \u2264 5.6, molar refractivity 40\u2013130, "
            "and heavy atom count 20\u201370."
        )

    if settings.get("enable_egan"):
        sentences.append(
            "Passive intestinal absorption was estimated using Egan's egg model "
            "(Egan et al., 2000), removing compounds outside the boundaries of "
            "\u22121 \u2264 logP \u2264 5.88 and TPSA \u2264 131.6 \u00c5\u00b2."
        )

    if settings.get("enable_muegge"):
        sentences.append(
            "The Muegge filter (Muegge et al., 2001) was applied, retaining compounds "
            "with MW 200\u2013600 Da, \u22122 \u2264 logP \u2264 5, TPSA \u2264 150 \u00c5\u00b2, "
            "\u22647 rings, \u22645 H-bond donors, \u226410 H-bond acceptors, "
            "and \u226415 rotatable bonds."
        )

    # Deduplication (always applied)
    sentences.append(
        "Duplicate structures were identified by canonical SMILES comparison "
        "and removed, retaining the first occurrence."
    )

    # SA Score (informational, not a filter)
    if settings.get("enable_sa_score"):
        sentences.append(
            "The synthetic accessibility (SA) score (Ertl & Schuffenhauer, 2009) "
            "was computed for retained compounds as an informational metric "
            "(scale: 1 = easily synthesizable, 10 = very difficult); "
            "it was not used as a filter."
        )

    # Featurization
    fp_type = settings.get("fp_type")
    if fp_type and fp_type in _FP_SENTENCES:
        n_bits = settings.get("n_bits", 2048)
        radius = settings.get("radius", 2)
        sentences.append(
            _FP_SENTENCES[fp_type].format(radius=radius, n_bits=n_bits)
        )
        sentences.append(
            "Physicochemical descriptors (MW, LogP, TPSA, H-bond donors, "
            "H-bond acceptors, rotatable bonds, aromatic rings) were also computed "
            "using RDKit (Landrum, G., RDKit: Open-source cheminformatics)."
        )

    # Closing citation
    sentences.append(
        "All preprocessing and featurization steps were performed using the "
        "QSAR Preprocessing Tool (https://qsar-preprocessing-tool.streamlit.app/)."
    )

    return " ".join(sentences)
