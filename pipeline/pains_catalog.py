"""
Explanations for common PAINS and Brenk structural alert patterns.

Used by the UI to provide educational context when a molecule is flagged.
"""

import re

PAINS_EXPLANATIONS = {
    "catechol_A": {
        "name": "Catechol",
        "smarts": "[OH]c1ccccc1[OH]",
        "description": "Catechols (1,2-dihydroxybenzene) are one of the most common PAINS patterns. They chelate metal ions in assay buffers, undergo oxidation to reactive quinones, and generate reactive oxygen species (ROS) — all of which cause false positive activity readings.",
        "mechanism": "Metal chelation, redox cycling, quinone formation",
        "affected_assays": "Nearly all biochemical assays, especially those using metal-dependent enzymes",
        "recommendation": "Remove from screening hits unless activity is confirmed by orthogonal assays. True catechol drugs exist (e.g., dopamine, norepinephrine) but are rare and require extensive validation.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010, 53, 2719-2740",
    },
    "quinone_A": {
        "name": "Quinone",
        "smarts": "O=C1C=CC(=O)C=C1",
        "description": "Quinones are Michael acceptors — they react covalently with nucleophilic residues (cysteine thiols) in proteins, causing non-specific inhibition unrelated to the target of interest.",
        "mechanism": "Covalent modification via Michael addition",
        "affected_assays": "All protein-based assays, especially cysteine-dependent enzymes",
        "recommendation": "Generally remove. Some approved drugs contain quinone-like motifs (e.g., doxorubicin) but their mechanism involves deliberate DNA intercalation, not accidental reactivity.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010, 53, 2719-2740",
    },
    "rhod_sat_A": {
        "name": "Rhodanine (saturated)",
        "smarts": "O=C1CSC(=S)N1",
        "description": "Saturated rhodanines are frequent hitters across diverse assay types. They aggregate in solution, chelate metals, and undergo thiol exchange reactions, all producing false positives.",
        "mechanism": "Aggregation, metal chelation, thiol exchange",
        "affected_assays": "HTS campaigns across all target classes",
        "recommendation": "One of the most notorious PAINS families. Remove unless extensive counter-screening confirms genuine activity.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010; Tomasic & Masic, Curr. Med. Chem. 2009",
    },
    "rhodanine_A": {
        "name": "Rhodanine",
        "smarts": "O=C1CSC(=S)N1",
        "description": "Rhodanines are frequent hitters across diverse assay types. They aggregate in solution, chelate metals, and undergo thiol exchange reactions, all producing false positives.",
        "mechanism": "Aggregation, metal chelation, thiol exchange",
        "affected_assays": "HTS campaigns across all target classes",
        "recommendation": "One of the most notorious PAINS families. Remove unless extensive counter-screening confirms genuine activity.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010; Tomasic & Masic, Curr. Med. Chem. 2009",
    },
    "azo_A": {
        "name": "Azo compound",
        "smarts": "N=N",
        "description": "Azo compounds (R-N=N-R) interfere with optical assay readouts due to their intense color, and can be metabolically reduced to release potentially mutagenic aromatic amines.",
        "mechanism": "Spectroscopic interference, metabolic instability",
        "affected_assays": "Absorbance-based and fluorescence-based HTS assays",
        "recommendation": "Remove from primary screening results. Color interference is especially problematic in AlphaScreen and absorbance assays.",
        "reference": "Baell, J. Med. Chem. 2010",
    },
    "ene_rhodanine_A": {
        "name": "Ene-rhodanine",
        "smarts": "",
        "description": "Benzylidene rhodanines combine rhodanine PAINS liability with an exocyclic alkene that acts as a Michael acceptor — a double liability for false positives.",
        "mechanism": "Aggregation + Michael addition",
        "affected_assays": "All HTS assay types",
        "recommendation": "Remove. Among the most unreliable chemotypes in drug discovery.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010",
    },
    "ene_rhod_A": {
        "name": "Ene-rhodanine",
        "smarts": "",
        "description": "Benzylidene rhodanines combine rhodanine PAINS liability with an exocyclic alkene that acts as a Michael acceptor — a double liability for false positives.",
        "mechanism": "Aggregation + Michael addition",
        "affected_assays": "All HTS assay types",
        "recommendation": "Remove. Among the most unreliable chemotypes in drug discovery.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010",
    },
    "hydroxyphenyl_thiazolinedione_A": {
        "name": "Hydroxyphenyl thiazolidinedione",
        "smarts": "",
        "description": "Thiazolidinediones with hydroxyphenyl groups combine multiple PAINS liabilities: the thiazolidinedione ring aggregates and chelates metals, while the phenol undergoes oxidation.",
        "mechanism": "Aggregation, metal chelation, redox cycling",
        "affected_assays": "Broad HTS interference",
        "recommendation": "Remove unless the thiazolidinedione is the intended pharmacophore (e.g., PPARγ agonists like pioglitazone).",
        "reference": "Baell & Holloway, J. Med. Chem. 2010",
    },
    "mannich_A": {
        "name": "Mannich base",
        "smarts": "",
        "description": "Mannich bases (aminomethyl compounds) can decompose to release formaldehyde and an amine, both of which react non-specifically with proteins and assay components.",
        "mechanism": "Decomposition to reactive aldehydes",
        "affected_assays": "Protein-based biochemical assays",
        "recommendation": "Treat with caution. Confirm activity is not due to formaldehyde release.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010",
    },
    "imine_one_A": {
        "name": "Imine",
        "smarts": "",
        "description": "Imines (Schiff bases) are hydrolytically unstable and can react reversibly with amine and thiol groups on proteins, leading to non-specific apparent inhibition.",
        "mechanism": "Reversible covalent modification, hydrolytic instability",
        "affected_assays": "Biochemical assays with long incubation times",
        "recommendation": "Check for stability in assay buffer. Activity may disappear with pre-incubation or in cell-based assays.",
        "reference": "Baell & Holloway, J. Med. Chem. 2010",
    },
}

GENERIC_PAINS_EXPLANATION = {
    "name": "PAINS pattern",
    "description": "This compound matches a Pan-Assay Interference pattern — a structural motif known to produce false positive results in biological assays through non-specific mechanisms rather than genuine target binding.",
    "mechanism": "Various (aggregation, reactivity, fluorescence interference, redox activity, or metal chelation)",
    "affected_assays": "Multiple assay types in high-throughput screening",
    "recommendation": "Treat screening results with caution. Confirm activity with orthogonal assays before pursuing as a genuine hit.",
    "reference": "Baell & Holloway, J. Med. Chem. 2010, 53, 2719-2740",
}

BRENK_EXPLANATIONS = {
    "phenol_ester": {
        "name": "Phenol ester",
        "description": "Phenol esters are metabolically labile — they are rapidly hydrolyzed by esterases in vivo, releasing the parent phenol and acid. This makes them unsuitable as drug candidates due to poor metabolic stability.",
        "mechanism": "Enzymatic hydrolysis by esterases",
        "recommendation": "Remove unless the ester is a deliberate prodrug strategy. Consider replacing with an amide or other metabolically stable bioisostere.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "alkyl_halide": {
        "name": "Alkyl halide",
        "description": "Alkyl halides are reactive electrophiles that alkylate DNA and proteins non-specifically. They pose mutagenicity and toxicity risks and are flagged as structural alerts by regulatory agencies.",
        "mechanism": "SN2 alkylation of nucleophilic biomolecules (DNA bases, cysteine thiols)",
        "recommendation": "Remove. Alkyl halides are well-established genotoxic alerts. Rare exceptions exist for targeted covalent inhibitors (e.g., ibrutinib-like warheads) but require deliberate design.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "aldehyde": {
        "name": "Aldehyde",
        "description": "Aldehydes are reactive electrophiles that form Schiff bases (imines) with lysine residues and other nucleophilic sites on proteins. They can cause non-specific protein modification and assay interference.",
        "mechanism": "Schiff base formation with protein amines, reversible covalent modification",
        "recommendation": "Generally avoid. Some aldehyde-containing drugs exist (e.g., proteasome inhibitors) but these are carefully optimized for selectivity.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "michael_acceptor": {
        "name": "Michael acceptor",
        "description": "Michael acceptors (activated alkenes conjugated with electron-withdrawing groups) react covalently with nucleophilic residues — particularly cysteine thiols — in an irreversible or slowly reversible manner.",
        "mechanism": "1,4-conjugate addition to nucleophilic protein residues",
        "recommendation": "Remove from non-covalent drug discovery programs. In targeted covalent inhibitor programs, Michael acceptors are used deliberately but require careful selectivity profiling.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "epoxide": {
        "name": "Epoxide",
        "description": "Epoxides are highly strained three-membered ring ethers that react readily with nucleophiles. They alkylate DNA and proteins and are well-known mutagens and toxicophores.",
        "mechanism": "Ring-opening alkylation of DNA and protein nucleophiles",
        "recommendation": "Remove. Epoxides are established genotoxic alerts. Carfilzomib is a rare approved drug with an epoxide warhead, but it was specifically designed as a covalent proteasome inhibitor.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "acyl_halide": {
        "name": "Acyl halide",
        "description": "Acyl halides are extremely reactive — they hydrolyze instantly in aqueous media and acylate any available nucleophile. They are not viable as drug candidates.",
        "mechanism": "Rapid hydrolysis and non-specific acylation",
        "recommendation": "Remove immediately. No drug-like properties.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "sulfonyl_halide": {
        "name": "Sulfonyl halide",
        "description": "Sulfonyl halides are reactive electrophiles that form sulfonamide bonds with amines non-specifically. They are too reactive for use as drug candidates.",
        "mechanism": "Non-specific sulfonylation of nucleophilic residues",
        "recommendation": "Remove. Sometimes used as chemical biology probes but not suitable for drug development.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "polycyclic_aromatic_hydrocarbon": {
        "name": "Polycyclic aromatic hydrocarbon",
        "description": "Polycyclic aromatic hydrocarbons (PAHs) are flat, highly lipophilic molecules that intercalate into DNA and are metabolically activated to carcinogenic diol-epoxides.",
        "mechanism": "DNA intercalation, metabolic activation to carcinogens",
        "recommendation": "Remove. PAHs are well-established carcinogens and are not drug-like.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "hydroquinone": {
        "name": "Hydroquinone",
        "description": "Hydroquinones (1,4-dihydroxybenzene) are easily oxidized to quinones, generating reactive oxygen species and reactive Michael acceptor intermediates. They cause redox cycling in biological assays.",
        "mechanism": "Oxidation to quinone, redox cycling, ROS generation",
        "recommendation": "Remove. Hydroquinones are both PAINS-like interference compounds and metabolic liabilities.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "catechol": {
        "name": "Catechol",
        "description": "Catechols (1,2-dihydroxybenzene) chelate metal ions, undergo oxidation to reactive quinones, and generate reactive oxygen species — causing false positives in biochemical assays and poor metabolic stability in vivo.",
        "mechanism": "Metal chelation, oxidation to quinone, ROS generation",
        "recommendation": "Remove unless activity is confirmed by orthogonal assays. True catechol drugs are rare.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "chinone_1": {
        "name": "Quinone (chinone)",
        "description": "Quinones are Michael acceptors that react covalently with nucleophilic protein residues. They also undergo redox cycling, generating reactive oxygen species that interfere with assay readouts.",
        "mechanism": "Michael addition, redox cycling",
        "recommendation": "Remove. Quinones are among the most common causes of false positives in HTS.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "aniline": {
        "name": "Aniline",
        "description": "Anilines (aromatic amines) are metabolically activated by cytochrome P450 enzymes to reactive nitrene and hydroxylamine intermediates that can form DNA adducts, posing mutagenicity and carcinogenicity risks.",
        "mechanism": "Metabolic activation to reactive intermediates, DNA adduct formation",
        "recommendation": "Treat with caution. Some aniline-containing drugs exist but require careful toxicity assessment. Substituted anilines with electron-withdrawing groups are generally safer.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "Thiocarbonyl_group": {
        "name": "Thiocarbonyl group",
        "description": "Thiocarbonyl groups (C=S) are metabolically labile — they are oxidized to reactive sulfoxides and sulfones by cytochrome P450 enzymes. They can also chelate metal ions and undergo thiol exchange.",
        "mechanism": "Metabolic oxidation, metal chelation, thiol exchange",
        "recommendation": "Remove or replace with a carbonyl (C=O) bioisostere if possible.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "Michael_acceptor_1": {
        "name": "Michael acceptor",
        "description": "Michael acceptors (activated alkenes conjugated with electron-withdrawing groups) react covalently with nucleophilic residues — particularly cysteine thiols — in an irreversible or slowly reversible manner.",
        "mechanism": "1,4-conjugate addition to nucleophilic protein residues",
        "recommendation": "Remove from non-covalent drug discovery programs. In targeted covalent inhibitor programs, Michael acceptors are used deliberately but require careful selectivity profiling.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "thiol_2": {
        "name": "Thiol",
        "description": "Free thiols are problematic in drug discovery: they form disulfide bonds with assay proteins, oxidize in air, and complicate formulation. They frequently cause false positives in HTS.",
        "mechanism": "Disulfide exchange, oxidation, metal chelation",
        "recommendation": "Remove from HTS hits. Captopril is a rare approved thiol-containing drug, but thiol drugs require careful handling.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
    "thiol": {
        "name": "Thiol",
        "description": "Free thiols are problematic in drug discovery: they form disulfide bonds with assay proteins, oxidize in air, and complicate formulation. They frequently cause false positives in HTS.",
        "mechanism": "Disulfide exchange, oxidation, metal chelation",
        "recommendation": "Remove from HTS hits. Captopril is a rare approved thiol-containing drug, but thiol drugs require careful handling.",
        "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
    },
}

GENERIC_BRENK_EXPLANATION = {
    "name": "Brenk structural alert",
    "description": "This compound contains a structural motif flagged by the Brenk filter — a set of patterns associated with poor pharmacokinetics, metabolic instability, toxicity, or reactivity that make compounds unsuitable for drug development.",
    "mechanism": "Various (reactivity, metabolic instability, toxicity, or poor drug-likeness)",
    "recommendation": "Consider removing or modifying the flagged substructure before progressing the compound.",
    "reference": "Brenk et al., ChemMedChem 2008, 3, 435-444",
}


def _extract_base_name(pattern_string):
    """Extract base pattern name from strings like 'catechol_A(92)' → 'catechol_A'."""
    if not pattern_string:
        return ""
    match = re.match(r"^(.+?)(?:\(\d+\))?$", pattern_string.strip())
    return match.group(1) if match else pattern_string.strip()


def get_pains_explanation(pattern_name):
    """Look up a PAINS pattern explanation by name (e.g. 'catechol_A(92)')."""
    base = _extract_base_name(pattern_name)
    return PAINS_EXPLANATIONS.get(base, GENERIC_PAINS_EXPLANATION)


def get_brenk_explanation(pattern_name):
    """Look up a Brenk pattern explanation by name."""
    base = _extract_base_name(pattern_name)
    return BRENK_EXPLANATIONS.get(base, GENERIC_BRENK_EXPLANATION)
