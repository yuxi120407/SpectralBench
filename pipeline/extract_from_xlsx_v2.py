"""
Extract spectral analysis data from papers listed in xanes_literature_examples.xlsx.

V2 changes vs extract_from_xlsx.py: adds CRITICAL RULES to the extraction
prompt to prevent downstream conflicts in stage 1:
  1. Per-edge splitting for multi-element papers (Ti K + Zn K → 2 conditions)
  2. Canonical, consistent phase names across all conditions in one paper
  3. Deduplicate fitting-method variants (LCF vs shell-fit → 1 condition)
  4. Consolidate pure reference samples (8 Pt-foil measurements → 1 condition)

Usage:
    python extract_from_xlsx_v2.py
    python extract_from_xlsx_v2.py --max-papers 5
    python extract_from_xlsx_v2.py --doi 10.1103/PhysRevMaterials.2.125403
"""

import os
import sys
import json
import time
import glob
import argparse
import subprocess
import re
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(BASE_DIR)
XLSX_FILE = os.path.join(BASE_DIR, "xanes_literature_examples.xlsx")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

GEMINI_PYTHON = (
    "/home/idies/workspace/Storage/xyu1/persistent"
    "/pytorch_env/sam3_gcloud/bin/python3"
)
GEMINI_SCRIPT = os.path.join(BENCHMARK_DIR, "..", "gemini_call_v2.py")

API_KEYS_FILE = os.path.join(
    os.path.dirname(BENCHMARK_DIR), "..",
    "scientific-literature-mining", "corpus_web", "tdm",
    "api_key_and_archive_info.txt"
)

# Vertex AI config (same as gemini_call.py)
PROJECT_ID = "geminienterpriseprod-485218"
LOCATION = "global"
GEMINI_MODEL = "gemini-3.1-pro-preview"

AVAILABLE_REFERENCES = """
Available reference spectra we have for fitting:

Fe K-edge: metallic_iron (Fe), wustite (FeO, Fe2+), hematite (Fe2O3, Fe3+),
           siderite (FeCO3), austenite (Fe, fcc)
Cu K-edge: metallic_copper (Cu), cuprite (Cu2O, Cu+), tenorite (CuO, Cu2+)
Cu L3-edge: metallic_copper_L3 (Cu0), cuprite_L3 (Cu2O, Cu+), tenorite_L3 (CuO, Cu2+)
Ni K-edge: metallic_nickel (Ni), nickel_oxide (NiO, Ni2+), lithium_nickelate (LiNiO2, Ni3+)
Ni L3-edge: metallic_nickel_L3 (Ni0), nickel_oxide_L3 (NiO, Ni2+)
Mn K-edge: manganese_oxide_II (MnO, Mn2+), manganese_dioxide (MnO2, Mn4+),
           manganese_sesquioxide (Mn2O3, Mn3+)
Ti K-edge: metallic_titanium (Ti), rutile (TiO2, Ti4+), titanium_sesquioxide (Ti2O3, Ti3+),
           titanium_disulfide (TiS2), strontium_titanate (SrTiO3, Ti4+)
Zn K-edge: metallic_zinc (Zn), zincite (ZnO, Zn2+), sphalerite (ZnS),
           zinc_chloride (ZnCl2)
S K-edge:  zinc_sulfide_S (ZnS), pyrite_S (FeS2), lithium_thiophosphate_S (Li3PS4),
           titanium_disulfide_S (TiS2)
C K-edge:  graphite (C, sp2)
P K-edge:  lithium_thiophosphate_P (Li3PS4)
O K-edge:  rutile_O (TiO2), strontium_titanate_O (SrTiO3)
Li K-edge: lithium_titanate_Li (Li4Ti5O12)
Pt L3-edge: metallic_platinum (Pt), platinum_dioxide (PtO2, Pt4+)
Pd K-edge: metallic_palladium (Pd), palladium_oxide (PdO, Pd2+)
"""

# =============================================================================
# Step 1: Classify paper category
# =============================================================================

CLASSIFY_PROMPT = """You are an expert in XANES/XAS spectroscopy. Read this paper and classify
what type of experiment it describes.

Title: {title}
DOI: {doi}
Element: {element}
Edge: {edge}
Material: {material}

Classify into ONE primary category:

1. "synthesis" — Phase fractions depend on synthesis conditions (temperature, precursors,
   atmosphere, duration). Example: calcining iron oxide at different temperatures.
2. "electrochemistry" — Phase fractions depend on electrochemical conditions (charge/discharge,
   C-rate, voltage, SOC, cycle number). Example: operando battery cathode study.
3. "pure_phase" — Known pure-phase reference compounds (standards). Example: CuO powder standard.
4. "thin_film" — Phases depend on deposition or spatial composition gradients.
5. "catalyst" — Phases depend on catalyst preparation, support, particle size, reaction conditions.
6. "environmental" — Phases from natural/environmental samples (soil, mineral weathering).

Return ONLY a JSON object:
{{
  "category": "synthesis|electrochemistry|pure_phase|thin_film|catalyst|environmental",
  "confidence": "high|medium|low",
  "reasoning": "one-sentence explanation",
  "key_variables": ["list", "of", "variables", "that", "control", "phase", "fractions"]
}}"""

# =============================================================================
# Step 2: Category-specific extraction prompts
# =============================================================================

EXTRACT_COMMON_HEADER = """You are an expert in XANES/XAS spectroscopy. Carefully analyze this paper
and extract ALL quantitative spectral analysis results — from BOTH the text AND figures/tables.

Title: {title}
DOI: {doi}
Element: {element}
Edge: {edge}
Material: {material}

PAPER CATEGORY: {category}

Look for ANY of these types of quantitative results:
1. **Linear combination fitting (LCF)** — phase fractions from fitting
2. **Phase identification** — which phase/compound a spectrum matches
3. **Oxidation state analysis** — ratios of oxidation states
4. **Spectral component decomposition** — weights from PCA, MCR, etc.
5. **Peak fitting** — relative areas of fitted peaks
6. **Simulation fitting** — FEFF/FDMNES parameters matched to experiment
7. **Pre-edge analysis** — centroid position or area ratios

IMPORTANT: Check figures carefully! Results are often in figure captions,
tables, stacked spectra plots, or bar charts.

Available reference spectra:
{references}
"""

EXTRACT_COMMON_FOOTER = """
IMPORTANT:
- Include results from BOTH text AND figures — examine every figure carefully
- For phase identification (single phase), use fraction=1.0
- Fractions should sum to ~1.0 (normalize percentages to decimals)
- Note which figure/table in data_source (e.g., "figure_3", "table_2")
- Include ALL conditions/samples, even if phases are outside our reference set
- If NO quantitative results exist at all, set relevant=false
- If preparation details come from a CITED reference:
  - Put the DOI in `conditions_from_ref` (if you're confident) OR leave null if uncertain
  - ALWAYS fill `conditions_ref_citation` with title, first_author family name, year, and journal
  - DO NOT invent DOIs — if you're not sure of the DOI, leave it null and rely on the title/author lookup
  - The pipeline will resolve the DOI from Semantic Scholar / Crossref using the citation info
- For fields not reported in the paper, use null — do NOT invent values

CRITICAL RULES to prevent downstream conflicts:

1. **PER-EDGE SPLITTING for multi-element papers**:
   If XANES was measured at multiple absorption edges (e.g. Ti K AND Zn K),
   create SEPARATE conditions per edge. Each must have its own phase_fractions
   from the LCF/fit at THAT specific edge (they will usually differ).
   Tag the sample_id with the edge: e.g. "Combinatorial film fTi=0.15 (Ti K-edge)"
   and "Combinatorial film fTi=0.15 (Zn K-edge)". DO NOT copy the same fractions
   to both edges — extract each edge's fit independently.

2. **CANONICAL, CONSISTENT PHASE NAMES**:
   Use the SAME phase name for the same phase across ALL conditions in the paper.
   Prefer chemical formula with structure tag: "TiO2 (anatase)", "TiO2 (rutile)",
   "LiFePO4", "Pd in staple site". Never mix "Pd staple site" and "Pd in staple site"
   for the same thing. Do NOT include metadata like "(outside_references: true)" in
   the phase name string.

3. **DEDUPLICATE FITTING-METHOD VARIANTS**:
   If the SAME sample is reported with multiple fitting methods (e.g. LCF fit vs
   shell-fit vs amorphous model), output only ONE condition using the paper's
   PRIMARY fit. Add a "fitting_method" field noting the method used. Do NOT
   create separate conditions that differ only by analysis method.

4. **CONSOLIDATE PURE REFERENCE SAMPLES**:
   If multiple measurements yield the same pure reference (e.g. 8 different Pt
   samples all giving metallic_platinum=1.0), output only ONE representative
   condition with is_reference=true. Do not create 8 near-identical scenarios."""

EXTRACT_PURE_PHASE_FOOTER = """
IMPORTANT:
- Include results from BOTH text AND figures — examine every figure carefully
- For each pure-phase compound, set phase_fractions to [{{"phase": "compound_name", "fraction": 1.0}}]
- Note which figure/table in data_source (e.g., "figure_3", "table_2")
- Include ALL compounds measured, even if they are outside our reference set
- Set relevant=true if the paper measures, describes, or shows XANES spectra of known compounds — even without LCF or quantitative fitting
- For fields not reported in the paper, use null — do NOT invent values

CRITICAL RULES to prevent downstream conflicts:
1. Use CONSISTENT phase/compound names across all conditions in this paper
   (e.g. always "TiO2 (anatase)", never mix with "anatase TiO2")
2. For multi-edge papers, create SEPARATE conditions per edge with edge tag
   in sample_id: "sample_X (Ti K-edge)" and "sample_X (Zn K-edge)"
3. Deduplicate reference measurements — if 8 measurements all give the same
   pure compound, output only ONE with is_reference=true"""

EXTRACT_SYNTHESIS_PROMPT = EXTRACT_COMMON_HEADER + """
This paper studies how SYNTHESIS CONDITIONS affect phase composition.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "synthesis",
  "uses_our_phases": true/false,
  "analysis_type": "LCF/phase_id/PCA/peak_fitting/simulation/other",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "brief description",
      "material": "material description",
      "preparation": "detailed synthesis from paper",
      "method": "synthesis method (e.g., solid-state, sol-gel)",
      "precursors": "precursor chemicals with ratios, or null",
      "temperature_C": number or null,
      "duration_hours": number or null,
      "heating_rate_C_per_min": number or null,
      "atmosphere": "gas environment, or null",
      "cooling_method": "cooling method, or null",
      "post_treatment": "post-treatment steps, or null",
      "key_variable": "what distinguishes this sample",
      "key_variable_value": "specific value",
      "phase_fractions": [
        {{"phase": "chemical name", "fraction": 0.45}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": number or null,
      "conditions_from_ref": "DOI if conditions from another paper, else null",
      "conditions_ref_citation": {{"title": "cited paper title", "first_author": "first author family name", "year": 2021, "journal": "e.g. J. Phys. Chem. C"}} or null
    }}
  ],
  "causal_reasoning": "Why these synthesis conditions produce these phases",
  "condition_fraction_rules": ["rule1", "rule2"],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}""" + EXTRACT_COMMON_FOOTER

EXTRACT_ELECTROCHEMISTRY_PROMPT = EXTRACT_COMMON_HEADER + """
This paper studies how ELECTROCHEMICAL CONDITIONS affect phase composition.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "electrochemistry",
  "uses_our_phases": true/false,
  "analysis_type": "LCF/phase_id/PCA/peak_fitting/other",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "brief description of this electrochemical state",
      "material": "electrode material",
      "electrode_type": "cathode/anode/working electrode",
      "electrolyte": "electrolyte composition, or null",
      "cell_type": "coin cell/pouch cell/in-situ cell/etc., or null",
      "charge_discharge": "charging/discharging/charged/discharged/OCV",
      "c_rate": "C-rate (e.g., C/10, 1C, 8C), or null",
      "voltage_V": "voltage or voltage window, or null",
      "state_of_charge": "SOC or ion content x, or null",
      "cycle_number": "which cycle, or null",
      "capacity_mAh_g": "specific capacity, or null",
      "measurement_mode": "operando/in-situ/ex-situ",
      "temperature_C": "measurement temperature, or null",
      "key_variable": "what distinguishes this state",
      "key_variable_value": "specific value",
      "phase_fractions": [
        {{"phase": "chemical name", "fraction": 0.45}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": number or null,
      "conditions_from_ref": null
    }}
  ],
  "causal_reasoning": "Why these electrochemical conditions produce these phases",
  "condition_fraction_rules": ["rule1", "rule2"],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}""" + EXTRACT_COMMON_FOOTER

EXTRACT_PURE_PHASE_PROMPT = EXTRACT_COMMON_HEADER + """
This paper measures KNOWN PURE-PHASE reference compounds or standards.

For pure-phase papers, the key data is the DETAILED SPECTRAL FINGERPRINT of each
compound. We need research-paper-level detail — not just "has a white line" but
specific peak labels, energy positions in eV, orbital origins, and how the spectrum
compares to related compounds.

Think of it like slide 10 from a spectroscopy lecture: for each compound we want
the spectral shape, individually labeled peaks with energies, electronic structure
interpretation, and structural sensitivity.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "pure_phase",
  "uses_our_phases": true/false,
  "analysis_type": "reference measurement",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "what this reference compound is",
      "material": "chemical formula and name",
      "crystal_structure": "crystal structure, or null",
      "oxidation_state": "formal oxidation state",
      "coordination": "coordination geometry, or null",
      "purity": "purity or source info, or null",
      "measurement_mode": "transmission/fluorescence/electron yield/EELS",
      "sample_form": "powder/foil/pellet/thin film/solution",

      "edge_position_eV": "edge energy in eV reported in paper, or null",

      "labeled_peaks": [
        {{
          "label": "peak label used in paper (e.g., A1, A2, B, pre-edge, white line) or descriptive name",
          "energy_eV": "energy position in eV, or energy relative to edge, or null",
          "intensity": "relative intensity (weak/moderate/strong) or quantitative if given",
          "origin": "electronic transition and orbital character described in paper (e.g., 1s->3d t2g, 2p->eg, exciton), or null",
          "source": "paper_data or null"
        }}
      ],

      "spectral_shape_summary": "overall description of spectral shape — number of distinct features, their pattern, key visual characteristics as described in paper",

      "pre_edge_description": "pre-edge features: number of peaks, energies, intensities, and physical origin as described in paper, or null",

      "white_line_description": "white line: shape, intensity, splitting, energy, as described in paper, or null",

      "post_edge_description": "post-edge features: shoulders, oscillations, multiple scattering resonances as described in paper, or null",

      "electronic_structure_interpretation": "how the paper explains spectral features in terms of electronic structure — orbital assignments, band structure, selection rules, PDOS interpretation, or null",

      "structural_sensitivity": "how spectral features differ from other polymorphs or related compounds as discussed in paper (e.g., anatase vs rutile, octahedral vs tetrahedral), or null",

      "distinguishing_features": "what the paper says makes this spectrum distinguishable from other phases of the same element, or null",

      "simulation_method": "if the paper uses theory/simulation: method and code (e.g., BSE/exciting, FEFF, FDMNES, VASP), or null",

      "phase_fractions": [
        {{"phase": "chemical name", "fraction": 1.0}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": null
    }}
  ],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}

IMPORTANT for pure_phase:
- Set relevant=true if the paper measures or describes XANES spectra of known compounds,
  even without LCF fitting
- Extract spectral feature descriptions from the text AND figure captions AND supplementary info
- For labeled_peaks: extract EVERY individually identifiable peak or feature from the paper.
  Use the paper's own labels if it names peaks (A1, A2, B, C1, etc.). If the paper does
  not label peaks, use descriptive names (pre-edge, white line, post-edge shoulder, etc.).
  Include energy positions in eV whenever the paper reports them.
- For electronic_structure_interpretation: extract the paper's explanation of WHY each
  feature appears — orbital assignments, band character, selection rules, hybridization.
  This is critical for deep benchmarking.
- For structural_sensitivity: extract any discussion of how the spectrum compares to
  other polymorphs, coordination environments, or related compounds.
- For each compound, set phase_fractions to [{{"phase": "compound_name", "fraction": 1.0}}]
- If the paper only has simulated spectra (not experimental), still extract but note
  simulation_method
- Do NOT invent spectral details not in the paper — use null for missing fields""" + EXTRACT_PURE_PHASE_FOOTER

EXTRACT_THIN_FILM_PROMPT = EXTRACT_COMMON_HEADER + """
This paper studies THIN FILMS or COMBINATORIAL LIBRARIES.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "thin_film",
  "uses_our_phases": true/false,
  "analysis_type": "LCF/phase_id/PCA/MCR/other",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "what this sample/position is",
      "material": "film material and substrate",
      "deposition_method": "PLD/sputtering/ALD/CVD/etc.",
      "substrate": "substrate material, or null",
      "deposition_temperature_C": number or null,
      "film_thickness_nm": number or null,
      "composition_variable": "what varies (e.g., Ti fraction)",
      "composition_value": "specific value at this point",
      "annealing": "post-deposition treatment, or null",
      "key_variable": "what distinguishes this point",
      "key_variable_value": "specific value",
      "phase_fractions": [
        {{"phase": "chemical name or structural motif", "fraction": 0.45}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": number or null,
      "conditions_from_ref": null
    }}
  ],
  "causal_reasoning": "Why composition/deposition affects these phases",
  "condition_fraction_rules": ["rule1", "rule2"],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}""" + EXTRACT_COMMON_FOOTER

EXTRACT_CATALYST_PROMPT = EXTRACT_COMMON_HEADER + """
This paper studies CATALYSTS or NANOPARTICLES.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "catalyst",
  "uses_our_phases": true/false,
  "analysis_type": "LCF/phase_id/simulation/other",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "what this catalyst sample is",
      "material": "catalyst composition",
      "support": "support material, or null",
      "active_metal_loading": "metal loading, or null",
      "particle_size_nm": "particle size, or null",
      "preparation_method": "synthesis method",
      "treatment": "pre-treatment or activation conditions, or null",
      "reaction_conditions": "gas composition, temperature during measurement, or null",
      "measurement_mode": "operando/in-situ/ex-situ",
      "key_variable": "what distinguishes this sample",
      "key_variable_value": "specific value",
      "phase_fractions": [
        {{"phase": "chemical name or structural site", "fraction": 0.45}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": number or null,
      "conditions_from_ref": null
    }}
  ],
  "causal_reasoning": "Why these conditions produce these phases",
  "condition_fraction_rules": ["rule1", "rule2"],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}""" + EXTRACT_COMMON_FOOTER

EXTRACT_ENVIRONMENTAL_PROMPT = EXTRACT_COMMON_HEADER + """
This paper studies ENVIRONMENTAL or GEOLOGICAL samples.

Return ONLY a JSON object:
{{
  "relevant": true/false,
  "element": "{element}",
  "edge": "{edge}",
  "category": "environmental",
  "uses_our_phases": true/false,
  "analysis_type": "LCF/phase_id/other",
  "sample_conditions": [
    {{
      "sample_id": "descriptive name",
      "description": "what this sample is",
      "material": "sample type (soil, sediment, mineral, etc.)",
      "sample_origin": "where from",
      "environmental_conditions": "pH, redox, temperature, etc., or null",
      "weathering_history": "exposure conditions, or null",
      "spatial_location": "where on the sample, or null",
      "key_variable": "what distinguishes this sample",
      "key_variable_value": "specific value",
      "phase_fractions": [
        {{"phase": "chemical name or mineral", "fraction": 0.45}}
      ],
      "data_source": "text/table_N/figure_N",
      "r_factor": number or null,
      "conditions_from_ref": null
    }}
  ],
  "causal_reasoning": "Why these conditions produce this speciation",
  "condition_fraction_rules": ["rule1", "rule2"],
  "phase_names": ["phase1", "phase2"],
  "n_phases": number,
  "key_finding": "one-sentence summary"
}}""" + EXTRACT_COMMON_FOOTER

CATEGORY_EXTRACT_PROMPTS = {
    "synthesis": EXTRACT_SYNTHESIS_PROMPT,
    "electrochemistry": EXTRACT_ELECTROCHEMISTRY_PROMPT,
    "pure_phase": EXTRACT_PURE_PHASE_PROMPT,
    "thin_film": EXTRACT_THIN_FILM_PROMPT,
    "catalyst": EXTRACT_CATALYST_PROMPT,
    "environmental": EXTRACT_ENVIRONMENTAL_PROMPT,
}

CONDITION_LOOKUP_PROMPT = """You are an expert in XANES/XAS spectroscopy.

Paper A ({parent_doi}) has a sample described as:
  sample_id: {sample_id}
  description: {description}
  preparation: {preparation}
  category: {category}

Paper A says the experimental conditions for this sample come from YOUR paper (the one attached).

Extract the ACTUAL experimental conditions for this sample from this paper.
The parent paper's category is "{category}", so extract the relevant fields.

Return ONLY a JSON object:
{{
  "found": true/false,
  "preparation": "detailed preparation/condition description",
  "method": "synthesis or experimental method",
  "temperature_C": number or null,
  "heating_rate_C_per_min": number or null,
  "duration_hours": number or null,
  "environment": "atmosphere/gas/vacuum conditions, or null",
  "precursors": "starting materials, or null",
  "post_treatment": "any post-treatment, or null",
  "electrode_type": "cathode/anode if electrochemistry, or null",
  "electrolyte": "electrolyte composition if electrochemistry, or null",
  "c_rate": "C-rate if electrochemistry, or null",
  "voltage_V": "voltage if electrochemistry, or null",
  "state_of_charge": "SOC if electrochemistry, or null",
  "cycle_number": "cycle number if electrochemistry, or null",
  "deposition_method": "deposition method if thin film, or null",
  "substrate": "substrate if thin film, or null",
  "film_thickness_nm": "thickness if thin film, or null"
}}

Be specific — include actual numbers, not vague descriptions. Use null for irrelevant fields."""

REFS_PROMPT = """You are an expert in XANES/XAS spectroscopy. This paper references experimental
conditions or spectral decomposition results from OTHER papers.

Title: {title}
DOI: {doi}
Element: {element}
Edge: {edge}

Look for:
1. References to other papers that contain XANES decomposition/LCF/fitting results
   (e.g., "sample prepared as in [ref 23]", "LCF results from [15]", "spectra compared to standards in [8]")
2. Cited papers that provide reference spectra used in this paper
3. Any paper cited for quantitative phase fractions, oxidation state analysis, or spectral fitting

Return ONLY a JSON object:
{{
  "cited_papers": [
    {{
      "doi": "10.xxxx/yyyy",
      "reason": "why this paper is relevant (e.g., 'provides LCF results for Fe K-edge')",
      "element": "element studied",
      "edge": "K/L3/etc"
    }}
  ]
}}

IMPORTANT:
- Only include papers with DOIs you can clearly read from the references section
- Only include papers relevant to XANES spectral analysis (not general chemistry refs)
- If no relevant cited papers found, return {{"cited_papers": []}}"""


def load_api_keys():
    keys = {}
    if not os.path.exists(API_KEYS_FILE):
        return keys
    with open(API_KEYS_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("API_key/"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    name = parts[0].replace("API_key/", "").strip()
                    key = parts[1].strip()
                    if not key.startswith("YOUR_"):
                        keys[name] = key
    return keys


def read_xlsx(path=None):
    import openpyxl
    wb = openpyxl.load_workbook(path or XLSX_FILE)
    ws = wb.active

    papers = []
    for r in range(2, ws.max_row + 1):
        doi = ws.cell(r, 1).value
        if not doi:
            continue
        papers.append({
            "doi": doi.strip(),
            "year": ws.cell(r, 2).value,
            "spectroscopy": ws.cell(r, 3).value or "",
            "element": ws.cell(r, 4).value or "",
            "edge": ws.cell(r, 5).value or "",
            "has_exp": ws.cell(r, 6).value or "",
            "has_sim": ws.cell(r, 7).value or "",
            "material": ws.cell(r, 8).value or "",
            "insight": ws.cell(r, 9).value or "",
            "application": ws.cell(r, 10).value or "",
            "contributor": ws.cell(r, 11).value or "",
            "source": "xlsx_curated",
        })
    return papers


# ── Gemini call ────────────────────────────────────────────────

GEMINI_MODEL = None  # set from CLI in main()

def call_gemini(prompt, pdf_path=None, image_paths=None, timeout=180):
    """Call Gemini via gemini_call_v2.py subprocess. Supports PDF and/or images.
    pdf_path can be a str (single PDF) or list of strs (multi-PDF fusion)."""
    cmd = [GEMINI_PYTHON, GEMINI_SCRIPT]
    if GEMINI_MODEL:
        cmd += ["--model", GEMINI_MODEL]
    if pdf_path:
        pdf_list = pdf_path if isinstance(pdf_path, list) else [pdf_path]
        for p in pdf_list:
            cmd += ["--pdf", p]
    if image_paths:
        cmd += ["--images"] + image_paths
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        response = json.loads(result.stdout.strip())
        if not response.get("ok"):
            return None
        return response["text"]
    except Exception:
        return None


def parse_json_response(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


# ── DOI resolution (verify / find canonical DOI) ───────────────

def _title_similarity(a, b):
    """Simple word-overlap similarity between two titles (0-1)."""
    if not a or not b:
        return 0.0
    words_a = set(w.lower() for w in a.split() if len(w) > 3)
    words_b = set(w.lower() for w in b.split() if len(w) > 3)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def resolve_doi(title=None, first_author=None, year=None, doi_hint=None):
    """Resolve a citation to a canonical DOI using Semantic Scholar (primary)
    and Crossref (fallback). Verifies title match to avoid returning wrong papers."""
    if not title:
        return None

    if doi_hint and doi_hint.startswith("10."):
        try:
            r = requests.get(
                f"https://api.crossref.org/works/{doi_hint}",
                headers={"User-Agent": "XanesBench/1.0 (mailto:xanes@bnl.gov)"},
                timeout=10,
            )
            if r.status_code == 200:
                cr_title = " ".join(r.json().get("message", {}).get("title", [""]))
                if _title_similarity(title, cr_title) >= 0.5:
                    return doi_hint
        except Exception:
            pass

    try:
        q = title
        if first_author:
            q = f"{title} {first_author}"
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": q, "fields": "title,authors,externalIds,year", "limit": 5},
            timeout=15,
        )
        if r.status_code == 200:
            for paper in r.json().get("data", []):
                ss_title = paper.get("title", "")
                sim = _title_similarity(title, ss_title)
                if sim < 0.5:
                    continue
                if year and paper.get("year") and abs(int(paper["year"]) - int(year)) > 1:
                    continue
                doi = (paper.get("externalIds") or {}).get("DOI")
                if doi:
                    return doi
    except Exception:
        pass

    try:
        params = {"query.bibliographic": title, "rows": 5}
        if first_author:
            params["query.author"] = first_author
        r = requests.get(
            "https://api.crossref.org/works",
            params=params,
            headers={"User-Agent": "XanesBench/1.0 (mailto:xanes@bnl.gov)"},
            timeout=15,
        )
        if r.status_code == 200:
            for item in r.json().get("message", {}).get("items", []):
                cr_title = " ".join(item.get("title", [""]))
                sim = _title_similarity(title, cr_title)
                if sim < 0.5:
                    continue
                if year:
                    pub_year = (item.get("published", {}).get("date-parts", [[0]])[0][0])
                    if pub_year and abs(int(pub_year) - int(year)) > 1:
                        continue
                doi = item.get("DOI")
                if doi:
                    return doi
    except Exception:
        pass

    return None


# ── Paper download ─────────────────────────────────────────────

def doi_to_filename(doi):
    return doi.replace("/", "_").replace(":", "_")


def lookup_pmcid(doi):
    if not doi:
        return None
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"ids": doi, "format": "json", "tool": "spectralbench", "email": "xyu1@bnl.gov"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            records = r.json().get("records", [])
            if records and records[0].get("pmcid"):
                return records[0]["pmcid"]
    except Exception:
        pass
    return None


def fetch_pmc_fulltext(pmcid, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pmc", "id": pmcid.replace("PMC", ""), "rettype": "full", "retmode": "text"}
    if api_key:
        params["api_key"] = api_key
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200 and len(r.text) > 500:
            return r.text[:80000]
    except Exception:
        pass
    return None


def download_pmc_figures(pmcid, doi):
    """Download figure images from PMC CDN. Returns list of local file paths."""
    from xml.etree import ElementTree as ET

    fig_dir = os.path.join(PAPERS_DIR, f"{doi_to_filename(doi)}_figs")

    # Check cache
    cached = sorted(glob.glob(os.path.join(fig_dir, "*.webp")) +
                    glob.glob(os.path.join(fig_dir, "*.png")) +
                    glob.glob(os.path.join(fig_dir, "*.jpg")))
    if cached:
        return cached

    # Get figure filenames from PMC XML
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pmc", "id": pmcid.replace("PMC", ""), "rettype": "xml"},
            timeout=30,
        )
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
    except Exception:
        return []

    figs = root.findall(".//fig")
    if not figs:
        return []

    # Scrape CDN URLs from figure pages
    fig_paths = []
    os.makedirs(fig_dir, exist_ok=True)

    for i, fig in enumerate(figs):
        fig_id = fig.get("id", f"fig{i+1}")
        try:
            page_r = requests.get(
                f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/figure/{fig_id}/",
                timeout=15,
            )
            if page_r.status_code != 200:
                continue
            match = re.search(
                r'(https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[^"\s>]+\.(?:webp|png|jpg|jpeg))',
                page_r.text,
            )
            if not match:
                continue
            img_url = match.group(1)
            img_r = requests.get(img_url, timeout=30)
            if img_r.status_code != 200:
                continue
            ext = os.path.splitext(img_url)[1].split("?")[0]
            local_path = os.path.join(fig_dir, f"{fig_id}{ext}")
            with open(local_path, "wb") as f:
                f.write(img_r.content)
            fig_paths.append(local_path)
        except Exception:
            continue

    return fig_paths


def download_pdf_unpaywall(doi):
    """Try to download PDF via Unpaywall. Returns bytes or None."""
    if not doi:
        return None
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                         params={"email": "xyu1@bnl.gov"}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        # Try all OA locations, not just best
        locations = []
        if data.get("best_oa_location"):
            locations.append(data["best_oa_location"])
        for loc in data.get("oa_locations", []):
            if loc not in locations:
                locations.append(loc)

        for loc in locations:
            pdf_url = loc.get("url_for_pdf")

            # OSTI: url_for_pdf is often null, but PDF lives at servlets/purl/{ID}
            if not pdf_url:
                loc_url = loc.get("url", "")
                osti_match = re.search(r"osti\.gov/biblio/(\d+)", loc_url)
                if osti_match:
                    pdf_url = f"https://www.osti.gov/servlets/purl/{osti_match.group(1)}"

            if not pdf_url:
                continue
            try:
                r2 = requests.get(pdf_url, timeout=60,
                                  headers={"User-Agent": "SpectralBench/1.0 (mailto:xyu1@bnl.gov)"},
                                  allow_redirects=True)
                if r2.status_code == 200:
                    ct = r2.headers.get("Content-Type", "")
                    if "pdf" in ct or r2.content[:4] == b"%PDF":
                        return r2.content
            except Exception:
                continue
    except Exception:
        pass
    return None


def download_pdf_osti(doi):
    """Try to download PDF via OSTI API with BNL institutional access."""
    if not doi:
        return None
    try:
        r = requests.get("https://www.osti.gov/api/v1/records",
                         headers={"Accept": "application/json"},
                         params={"doi": doi}, timeout=15)
        if r.status_code != 200:
            return None
        records = r.json()
        for item in records:
            for link in item.get("links", []):
                if link.get("rel") == "fulltext":
                    fulltext_url = link["href"]
                    r2 = requests.get(fulltext_url,
                                      params={"directFulltextAccess": "BNL"},
                                      timeout=60, allow_redirects=True)
                    if r2.status_code == 200 and r2.content[:4] == b"%PDF":
                        return r2.content
    except Exception:
        pass
    return None


def download_pdf_pmc(pmcid):
    """Try PMC PDF download."""
    if not pmcid:
        return None
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    try:
        r = requests.get(pdf_url, timeout=60,
                         headers={"User-Agent": "SpectralBench/1.0"},
                         allow_redirects=True)
        if r.status_code == 200 and (r.content[:4] == b"%PDF" or "pdf" in r.headers.get("Content-Type", "")):
            return r.content
    except Exception:
        pass
    return None


def fetch_unpaywall_html(doi):
    """Fallback: get HTML full text from Unpaywall."""
    if not doi:
        return None
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}",
                         params={"email": "xyu1@bnl.gov"}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        locations = []
        if data.get("best_oa_location"):
            locations.append(data["best_oa_location"])
        for loc in data.get("oa_locations", []):
            if loc not in locations:
                locations.append(loc)

        for loc in locations:
            url = loc.get("url")
            if not url:
                continue
            try:
                r2 = requests.get(url, timeout=30,
                                  headers={"User-Agent": "SpectralBench/1.0"},
                                  allow_redirects=True)
                if r2.status_code == 200:
                    ct = r2.headers.get("Content-Type", "")
                    if "pdf" in ct:
                        continue
                    text = re.sub(r"<[^>]+>", " ", r2.text)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) > 1000:
                        return text[:80000]
            except Exception:
                continue
    except Exception:
        pass
    return None


def fetch_elsevier_fulltext(doi, api_key):
    if not api_key or not doi:
        return None
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    for accept in ["text/plain", "text/xml"]:
        try:
            r = requests.get(url, headers={"X-ELS-APIKEY": api_key, "Accept": accept}, timeout=30)
            if r.status_code == 200 and len(r.text) > 500:
                text = r.text
                if accept == "text/xml":
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 500:
                    return text[:80000]
        except Exception:
            pass
    return None


def fetch_crossref_fulltext(doi):
    if not doi:
        return None
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         params={"mailto": "xyu1@bnl.gov"}, timeout=15)
        if r.status_code != 200:
            return None
        links = r.json().get("message", {}).get("link", [])
        for link in links:
            url = link.get("URL", "")
            ct = link.get("content-type", "")
            if "xml" in ct or "plain" in ct:
                try:
                    r2 = requests.get(url, timeout=30,
                                      headers={"User-Agent": "SpectralBench/1.0 (mailto:xyu1@bnl.gov)"})
                    if r2.status_code == 200 and len(r2.text) > 500:
                        text = re.sub(r"<[^>]+>", " ", r2.text)
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) > 500:
                            return text[:80000]
                except Exception:
                    continue
    except Exception:
        pass
    return None


def download_paper(paper, api_keys):
    """
    Download paper — tries PDF first (for multimodal), then text.
    Returns (pdf_bytes, text, source).
    At least one of pdf_bytes/text will be set if successful.
    Saves to papers/ folder for caching.
    """
    doi = paper.get("doi", "")
    fname = doi_to_filename(doi)

    # Check local cache
    pdf_path = os.path.join(PAPERS_DIR, f"{fname}.pdf")
    txt_path = os.path.join(PAPERS_DIR, f"{fname}.txt")

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            return f.read(), None, "cached_pdf"
    if os.path.exists(txt_path):
        # Still look up pmcid for figure downloading
        pmcid = lookup_pmcid(doi)
        if pmcid:
            paper["pmcid"] = pmcid
        with open(txt_path) as f:
            return None, f.read(), "cached_text"

    pdf_bytes = None
    text = None
    source = None

    # For BNL / DOE-funded papers, try OSTI first (higher hit rate, bypasses paywalls)
    contributor = (paper.get("contributor") or "").lower()
    is_bnl = ("bnl" in contributor or "nsls" in contributor or "brookhaven" in contributor
              or "doe" in contributor)

    if is_bnl:
        pdf_bytes = download_pdf_osti(doi)
        if pdf_bytes:
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            return pdf_bytes, None, "osti_pdf"

    # 1. Try PDF from Unpaywall
    pdf_bytes = download_pdf_unpaywall(doi)
    if pdf_bytes:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        return pdf_bytes, None, "unpaywall_pdf"

    # 1b. Try OSTI as fallback for non-BNL papers
    if not is_bnl:
        pdf_bytes = download_pdf_osti(doi)
        if pdf_bytes:
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            return pdf_bytes, None, "osti_pdf"

    # 2. Try PDF from PMC
    pmcid = lookup_pmcid(doi)
    if pmcid:
        paper["pmcid"] = pmcid
        pdf_bytes = download_pdf_pmc(pmcid)
        if pdf_bytes:
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            return pdf_bytes, None, "pmc_pdf"

        # 3. PMC text (includes tables but no figures)
        text = fetch_pmc_fulltext(pmcid, api_keys.get("PMC"))
        if text:
            with open(txt_path, "w") as f:
                f.write(text)
            return None, text, "pmc_text"

    # 4. Elsevier
    if doi.lower().startswith("10.1016") or doi.lower().startswith("10.1006"):
        text = fetch_elsevier_fulltext(doi, api_keys.get("Elsevier"))
        if text:
            with open(txt_path, "w") as f:
                f.write(text)
            return None, text, "elsevier"

    # 5. Crossref links
    text = fetch_crossref_fulltext(doi)
    if text:
        with open(txt_path, "w") as f:
            f.write(text)
        return None, text, "crossref"

    # 6. Unpaywall HTML
    text = fetch_unpaywall_html(doi)
    if text:
        with open(txt_path, "w") as f:
            f.write(text)
        return None, text, "unpaywall_html"

    return None, None, None


def get_crossref_metadata(doi):
    """Get title and authors from Crossref."""
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         params={"mailto": "xyu1@bnl.gov"}, timeout=10)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title_list = msg.get("title", [])
            title = title_list[0] if title_list else None
            authors = []
            for a in msg.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)
            venue = msg.get("container-title", [""])[0] if msg.get("container-title") else ""
            return title, authors, venue
    except Exception:
        pass
    return None, [], ""


# ── Process one paper ─────────────────────────────

def process_one_paper(paper, api_keys, text_only=False):
    """Download, extract data, and find cited refs for one paper.
    Returns (result_dict_or_None, list_of_cited_dois)."""
    doi = paper["doi"]
    elem = paper.get("element", "")
    edge = paper.get("edge", "")
    material = paper.get("material", "")

    # Metadata
    title, authors, venue = get_crossref_metadata(doi)
    if title:
        paper["title"] = title
    if authors:
        paper["authors"] = authors
    if venue:
        paper["venue"] = venue

    # Download
    print(f"  Downloading...", end=" ", flush=True)
    pdf_bytes, text, source = download_paper(paper, api_keys)

    if not pdf_bytes and not text:
        print("FAILED — no content available")
        return None, "no_content", []

    has_pdf = pdf_bytes is not None
    print(f"OK via {source} ({'PDF ' + str(len(pdf_bytes)//1024) + 'KB' if has_pdf else str(len(text)) + ' chars text'})")

    content_type = "pdf" if has_pdf else "text"
    pdf_file = os.path.join(PAPERS_DIR, f"{doi_to_filename(doi)}.pdf")

    # ── Download PMC figures if we only got text ──
    fig_paths = []
    if not has_pdf and paper.get("pmcid"):
        pmcid = paper["pmcid"]
        fig_paths = download_pmc_figures(pmcid, doi)
        if fig_paths:
            print(f"  Downloaded {len(fig_paths)} PMC figures")

    # ── Step 1: Classify paper category ──
    classify_prompt = CLASSIFY_PROMPT.format(
        title=paper.get("title", material),
        doi=doi, element=elem, edge=edge, material=material,
    )

    print(f"  Classifying...", end=" ", flush=True)
    if has_pdf and not text_only and os.path.exists(pdf_file):
        classify_raw = call_gemini(classify_prompt, pdf_path=pdf_file, timeout=60)
    elif text:
        classify_raw = call_gemini(
            classify_prompt + f"\n\nPaper text (first 15000 chars):\n{text[:15000]}",
            timeout=60,
        )
    else:
        classify_raw = None

    classification = parse_json_response(classify_raw)
    if not classification or not classification.get("category"):
        classification = {"category": "synthesis", "confidence": "low",
                          "reasoning": "classification failed", "key_variables": []}
    category = classification["category"]
    paper["classification"] = classification
    print(f"[{category}] ({classification.get('confidence', '?')})")

    time.sleep(1)

    # ── Step 2: Extract with category-specific prompt ──
    extract_template = CATEGORY_EXTRACT_PROMPTS.get(category, EXTRACT_SYNTHESIS_PROMPT)
    prompt = extract_template.format(
        title=paper.get("title", material),
        doi=doi, element=elem, edge=edge,
        material=material, references=AVAILABLE_REFERENCES,
        category=category,
    )

    mode_str = "(multimodal PDF)" if has_pdf else (f"(text + {len(fig_paths)} figures)" if fig_paths else "(text only)")
    print(f"  Extracting with Gemini {mode_str}...", end=" ", flush=True)
    if has_pdf and not text_only and os.path.exists(pdf_file):
        raw = call_gemini(prompt, pdf_path=pdf_file)
    elif text and fig_paths and not text_only:
        raw = call_gemini(
            prompt + f"\n\nPaper text (may be truncated):\n{text[:60000]}",
            image_paths=fig_paths,
        )
    elif text:
        raw = call_gemini(prompt + f"\n\nPaper text (may be truncated):\n{text[:60000]}")
    else:
        raw = None

    result = parse_json_response(raw)
    if result:
        result["classification"] = classification

    if result and result.get("relevant") and result.get("sample_conditions"):
        n_conds = len(result["sample_conditions"])
        n_with_fracs = sum(
            1 for c in result["sample_conditions"]
            if isinstance(c.get("phase_fractions"), list) and len(c["phase_fractions"]) > 0
        )
        analysis_type = result.get("analysis_type", "unknown")
        uses = "MATCHES refs" if result.get("uses_our_phases") else "outside refs"
        paper["fulltext_analysis"] = result
        paper["fulltext_source"] = source
        paper["content_type"] = content_type
        print(f"OK — {n_with_fracs}/{n_conds} conditions ({analysis_type}, {uses})")
    elif result and not result.get("relevant"):
        print(f"not relevant (no quantitative spectral data)")
    else:
        print(f"extraction failed")
        if raw:
            print(f"    [response preview: {raw[:150]}...]")

    time.sleep(1)

    # ── Look up conditions from referenced papers ──
    if paper.get("fulltext_analysis"):
        conds = paper["fulltext_analysis"].get("sample_conditions", [])
        for c in conds:
            ref_doi = c.get("conditions_from_ref")
            citation = c.get("conditions_ref_citation") or {}

            # Resolve/verify DOI via Semantic Scholar + Crossref
            if citation.get("title"):
                resolved = resolve_doi(
                    title=citation.get("title"),
                    first_author=citation.get("first_author"),
                    year=citation.get("year"),
                    doi_hint=ref_doi,
                )
                if resolved and resolved != ref_doi:
                    print(f"  DOI resolved: {ref_doi} → {resolved} (via Semantic Scholar/Crossref)")
                    c["conditions_from_ref_original"] = ref_doi
                    ref_doi = resolved
                    c["conditions_from_ref"] = resolved
                elif not resolved and not ref_doi:
                    print(f"  Could not resolve DOI for '{citation.get('title','?')[:60]}'")
                elif not resolved and ref_doi:
                    print(f"  ⚠ DOI {ref_doi} not verified against citation title — using as-is")

            if not ref_doi or not ref_doi.startswith("10."):
                continue
            print(f"  Looking up conditions from {ref_doi}...", end=" ", flush=True)

            ref_paper = {"doi": ref_doi}
            ref_pdf, ref_text, ref_src = download_paper(ref_paper, api_keys)
            if not ref_pdf and not ref_text:
                print("can't download → flagging as missing context")
                c["has_missing_context"] = True
                c["missing_context_note"] = (
                    f"Synthesis/preparation conditions unknown — details reported in "
                    f"cited reference {ref_doi} which could not be accessed."
                )
                for k in ("preparation", "synthesis", "annealing"):
                    v = c.get(k)
                    if v and "cited reference" in str(v).lower():
                        c[k] = f"UNKNOWN — see cited reference {ref_doi} (not accessible)"
                continue

            lookup_prompt = CONDITION_LOOKUP_PROMPT.format(
                parent_doi=doi,
                sample_id=c.get("sample_id", ""),
                description=c.get("description", ""),
                preparation=c.get("preparation", ""),
                category=category,
            )

            # Multi-PDF fusion: send BOTH source paper + cited ref
            ref_pdf_file = os.path.join(PAPERS_DIR, f"{doi_to_filename(ref_doi)}.pdf")
            source_pdf_file = os.path.join(PAPERS_DIR, f"{doi_to_filename(doi)}.pdf")

            if ref_pdf and os.path.exists(ref_pdf_file):
                pdf_list = []
                if os.path.exists(source_pdf_file):
                    pdf_list.append(source_pdf_file)
                pdf_list.append(ref_pdf_file)
                fusion_note = (
                    f"\n\nNOTE: You have {len(pdf_list)} PDFs attached: "
                    f"(1) the source paper ({doi}) that references this sample, and "
                    f"(2) the cited reference ({ref_doi}) that contains preparation details. "
                    f"Extract preparation/synthesis from the cited reference; "
                    f"use the source paper only for context."
                ) if len(pdf_list) > 1 else ""
                lookup_raw = call_gemini(lookup_prompt + fusion_note, pdf_path=pdf_list)
            elif ref_text:
                lookup_raw = call_gemini(lookup_prompt + f"\n\nPaper text:\n{ref_text[:60000]}")
            else:
                lookup_raw = None

            lookup_result = parse_json_response(lookup_raw)
            if lookup_result and lookup_result.get("found"):
                # Fill in any non-null fields from the lookup
                skip_keys = {"found"}
                for k, v in lookup_result.items():
                    if k in skip_keys:
                        continue
                    if v is not None and v != "not reported" and v != "null":
                        c[k] = v
                c["conditions_source_doi"] = ref_doi
                c["has_missing_context"] = False
                print(f"OK — filled in from {ref_src}")
            else:
                print("conditions not found in ref → flagging as missing context")
                c["has_missing_context"] = True
                c["missing_context_note"] = (
                    f"Synthesis/preparation conditions unknown — cited reference "
                    f"{ref_doi} downloaded but did not contain the expected details."
                )
                for k in ("preparation", "synthesis", "annealing"):
                    v = c.get(k)
                    if v and "cited reference" in str(v).lower():
                        c[k] = f"UNKNOWN — see cited reference {ref_doi}"
            time.sleep(1)

    # ── Find cited references ──
    cited_dois = []
    print(f"  Finding cited XANES references...", end=" ", flush=True)
    refs_prompt = REFS_PROMPT.format(
        title=paper.get("title", material),
        doi=doi, element=elem, edge=edge,
    )
    if has_pdf and not text_only and os.path.exists(pdf_file):
        refs_raw = call_gemini(refs_prompt, pdf_path=pdf_file)
    elif text:
        refs_raw = call_gemini(refs_prompt + f"\n\nPaper text (may be truncated):\n{text[:60000]}")
    else:
        refs_raw = None

    refs_result = parse_json_response(refs_raw)
    if refs_result and refs_result.get("cited_papers"):
        for cp in refs_result["cited_papers"]:
            cited_doi = cp.get("doi", "")
            if cited_doi and cited_doi.startswith("10."):
                cited_dois.append(cp)
        if cited_dois:
            print(f"found {len(cited_dois)} cited papers")
            for cp in cited_dois:
                print(f"    → {cp['doi']} ({cp.get('reason', '')[:60]})")
        else:
            print("none found")
    else:
        print("none found")

    time.sleep(1)

    has_data = bool(paper.get("fulltext_analysis"))
    return (paper if has_data else None), content_type, cited_dois


# ── Post-extraction dedup & conflict resolution ──

def _fraction_signature(condition):
    """Build a hashable signature from a condition's phase fractions."""
    pf = condition.get("phase_fractions", [])
    if not isinstance(pf, list) or not pf:
        return None
    pairs = []
    for x in pf:
        phase = (x.get("phase") or "").strip().lower()
        frac = x.get("fraction", 0.0)
        if isinstance(frac, (int, float)):
            frac = round(frac, 4)
        pairs.append((phase, frac))
    return tuple(sorted(pairs))


RESOLVE_CONFLICT_PROMPT = """You are an expert in XANES/XAS spectroscopy. I extracted conditions from
a paper and found conflicts: conditions with the same description but different
phase fractions. Re-read the paper and decide which to keep or fix.

Paper DOI: {doi}
Title: {title}

CONFLICTING CONDITIONS:
{conflicts_json}

For each conflict group, re-read the paper and return ONLY a JSON object:
{{
  "resolved": [
    {{
      "sample_id": "the sample_id to KEEP (the correct one)",
      "remove": ["sample_id_1", "sample_id_2"],
      "reason": "why"
    }}
  ]
}}

Rules:
- Keep the condition whose fractions match the paper's actual data (table/figure).
- If BOTH are wrong, pick the closest and note it.
- If they describe legitimately different samples (e.g., different particles,
  different edges), keep both — set remove to an empty list and explain why."""


def _find_conflicts(conditions):
    """Find conditions with the same description but different fractions."""
    from collections import defaultdict
    by_desc = defaultdict(list)
    for c in conditions:
        desc = (c.get("description") or "").strip().lower()
        if desc:
            by_desc[desc].append(c)

    conflicts = []
    for desc, group in by_desc.items():
        if len(group) < 2:
            continue
        sigs = set()
        for c in group:
            sigs.add(_fraction_signature(c))
        if len(sigs) > 1:
            conflicts.append(group)
    return conflicts


def dedup_conditions(extracted):
    """Remove duplicates and resolve conflicts within each paper.

    Step 1 (programmatic): remove conditions with identical phase fractions.
    Step 2 (Gemini): for conditions with same description but different fractions,
    ask Gemini with the PDF to decide which is correct.
    """
    total_removed = 0
    total_conflict_removed = 0

    for p in extracted:
        a = p.get("fulltext_analysis", {})
        conds = a.get("sample_conditions", [])
        if len(conds) < 2:
            continue

        doi = p.get("doi", "?")

        # ── Step 1: remove exact fraction duplicates ──
        seen = {}
        keep = []
        removed = []
        for c in conds:
            sig = _fraction_signature(c)
            if sig is None:
                keep.append(c)
                continue
            if sig in seen:
                removed.append((c.get("sample_id", "?"), seen[sig]))
                total_removed += 1
            else:
                seen[sig] = c.get("sample_id", "?")
                keep.append(c)

        if removed:
            print(f"  [{doi}] removed {len(removed)} duplicate(s):")
            for dup_id, orig_id in removed:
                print(f"    - '{dup_id}' (same fractions as '{orig_id}')")
            a["sample_conditions"] = keep

        # ── Step 2: resolve conflicts via Gemini ──
        conflicts = _find_conflicts(a["sample_conditions"])
        if not conflicts:
            continue

        conflict_sids = set()
        for group in conflicts:
            for c in group:
                conflict_sids.add(c.get("sample_id", ""))

        conflicts_json = json.dumps(
            [c for c in a["sample_conditions"] if c.get("sample_id", "") in conflict_sids],
            indent=2,
        )

        print(f"  [{doi}] {len(conflicts)} conflict group(s), asking Gemini to resolve...")
        prompt = RESOLVE_CONFLICT_PROMPT.format(
            doi=doi,
            title=p.get("title", "?"),
            conflicts_json=conflicts_json,
        )

        pdf_file = os.path.join(PAPERS_DIR, f"{doi_to_filename(doi)}.pdf")
        if os.path.exists(pdf_file):
            raw = call_gemini(prompt, pdf_path=pdf_file, timeout=120)
        else:
            raw = call_gemini(prompt, timeout=120)

        result = parse_json_response(raw)
        if not result or not result.get("resolved"):
            print(f"    conflict resolution failed, keeping all")
            continue

        to_remove = set()
        for r in result["resolved"]:
            for sid in r.get("remove", []):
                to_remove.add(sid)
            kept = r.get("sample_id", "?")
            reason = r.get("reason", "?")
            if r.get("remove"):
                print(f"    KEEP '{kept}', REMOVE {r['remove']} — {reason}")
            else:
                print(f"    KEEP ALL — {reason}")

        if to_remove:
            before = len(a["sample_conditions"])
            a["sample_conditions"] = [
                c for c in a["sample_conditions"]
                if c.get("sample_id", "") not in to_remove
            ]
            n = before - len(a["sample_conditions"])
            total_conflict_removed += n

        time.sleep(1)

    print(f"\n  Dedup: removed {total_removed} exact duplicate(s), "
          f"{total_conflict_removed} conflict(s) resolved")
    return extracted


# ── Main ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract spectral analysis data from curated XANES paper list"
    )
    parser.add_argument("--xlsx", type=str, default=None,
                        help="Path to input xlsx (default: xanes_literature_examples.xlsx)")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--doi", type=str, default=None,
                        help="Process a specific DOI only")
    parser.add_argument("--dois", type=str, nargs="+", default=None,
                        help="Process multiple DOIs (space-separated)")
    parser.add_argument("--doi-file", type=str, default=None,
                        help="File with one DOI per line")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip PDF download, use text only (faster but misses figures)")
    parser.add_argument("--no-chase", action="store_true",
                        help="Skip citation chasing (don't follow referenced papers)")
    parser.add_argument("--max-depth", type=int, default=1,
                        help="Max citation chase depth (default: 1)")
    parser.add_argument("--model", type=str, default=None,
                        help="Gemini model (e.g. gemini-3.8-flash, gemini-3.1-pro-preview). "
                             "Default: gemini_call_v2.py default.")
    args = parser.parse_args()

    global GEMINI_MODEL
    GEMINI_MODEL = args.model
    if GEMINI_MODEL:
        print(f"Using Gemini model: {GEMINI_MODEL}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    api_keys = load_api_keys()
    print(f"API keys loaded: {list(api_keys.keys())}")
    print(f"Papers cached in: {PAPERS_DIR}")

    # Collect DOIs from all sources
    doi_list = []
    if args.doi:
        doi_list.append(args.doi)
    if args.dois:
        doi_list.extend(args.dois)
    if args.doi_file:
        with open(args.doi_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    doi_list.append(line)

    if doi_list:
        # Try to get metadata from xlsx for known DOIs
        try:
            all_xlsx = read_xlsx(args.xlsx)
            xlsx_by_doi = {p["doi"]: p for p in all_xlsx}
        except Exception:
            xlsx_by_doi = {}
        papers = []
        for d in doi_list:
            if d in xlsx_by_doi:
                papers.append(xlsx_by_doi[d])
            else:
                papers.append({"doi": d, "element": "", "edge": "", "material": "", "source": "doi_list"})
        print(f"Processing {len(papers)} DOIs ({sum(1 for d in doi_list if d in xlsx_by_doi)} matched xlsx metadata)")
    else:
        papers = read_xlsx(args.xlsx)
        print(f"Papers in xlsx ({args.xlsx or 'default'}): {len(papers)}")

    if args.max_papers:
        papers = papers[:args.max_papers]

    print(f"Processing: {len(papers)} papers")
    print(f"Mode: {'text-only' if args.text_only else 'PDF preferred (multimodal)'}")
    print(f"Citation chasing: {'OFF' if args.no_chase else f'depth={args.max_depth}'}\n")
    print("=" * 70)

    extracted = []
    all_cited = []
    processed_dois = set()
    stats = {"pdf": 0, "text": 0, "no_content": 0}

    # ── Pass 1: Process xlsx papers ──
    print("\n>>> PASS 1: xlsx papers")
    for i, paper in enumerate(papers):
        doi = paper["doi"]
        print(f"\n[{i+1}/{len(papers)}] {doi}")
        print(f"  {paper.get('element','')} {paper.get('edge','')}-edge | {paper.get('material','')[:50]}")

        processed_dois.add(doi)
        result, ctype, cited = process_one_paper(paper, api_keys, args.text_only)

        if ctype == "no_content":
            stats["no_content"] += 1
        elif ctype == "pdf":
            stats["pdf"] += 1
        else:
            stats["text"] += 1

        if result:
            extracted.append(result)
        if cited:
            all_cited.extend(cited)

    # ── Pass 2+: Chase cited references ──
    if not args.no_chase and all_cited:
        for depth in range(1, args.max_depth + 1):
            new_dois = [c for c in all_cited if c["doi"] not in processed_dois]
            # Deduplicate
            seen = set()
            unique_new = []
            for c in new_dois:
                if c["doi"] not in seen:
                    seen.add(c["doi"])
                    unique_new.append(c)
            new_dois = unique_new

            if not new_dois:
                break

            print(f"\n{'='*70}")
            print(f">>> PASS {depth+1}: Chasing {len(new_dois)} cited references (depth={depth})")
            print("=" * 70)

            next_cited = []
            for i, cp in enumerate(new_dois):
                doi = cp["doi"]
                elem = cp.get("element", "")
                edge = cp.get("edge", "")

                print(f"\n[cited {i+1}/{len(new_dois)}] {doi}")
                print(f"  {elem} {edge}-edge | reason: {cp.get('reason', '?')[:60]}")

                processed_dois.add(doi)
                paper = {
                    "doi": doi,
                    "element": elem,
                    "edge": edge,
                    "material": cp.get("reason", ""),
                    "source": "citation_chase",
                    "cited_by": cp.get("cited_by", ""),
                }

                result, ctype, cited = process_one_paper(paper, api_keys, args.text_only)

                if ctype == "no_content":
                    stats["no_content"] += 1
                elif ctype == "pdf":
                    stats["pdf"] += 1
                else:
                    stats["text"] += 1

                if result:
                    extracted.append(result)
                if cited:
                    next_cited.extend(cited)

            all_cited = next_cited

    # ── Consistency check: deduplicate and verify within each paper ──
    # Always deduplicate (remove conditions with identical phase fractions within same paper)
    extracted = dedup_conditions(extracted)

    # ── Save results — filename reflects input xlsx ──
    xlsx_base = os.path.splitext(os.path.basename(args.xlsx or XLSX_FILE))[0]
    output_path = os.path.join(BASE_DIR, f"{xlsx_base}_extracted_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(extracted, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total papers processed: {len(processed_dois)}")
    print(f"    xlsx:               {len(papers)}")
    print(f"    cited (chased):     {len(processed_dois) - len(papers)}")
    print(f"  PDF downloads:        {stats['pdf']}")
    print(f"  Text-only downloads:  {stats['text']}")
    print(f"  No content:           {stats['no_content']}")
    print(f"  Papers with data:     {len(extracted)}")

    total_conds = 0
    conds_with_fracs = 0
    matches_refs = 0
    by_element = {}
    by_type = {}
    for p in extracted:
        a = p.get("fulltext_analysis", {})
        el = a.get("element", p.get("element", "?"))
        atype = a.get("analysis_type", "unknown")
        conds = a.get("sample_conditions", [])
        total_conds += len(conds)
        n_fracs = sum(1 for c in conds
                      if isinstance(c.get("phase_fractions"), list) and len(c["phase_fractions"]) > 0)
        conds_with_fracs += n_fracs
        if a.get("uses_our_phases"):
            matches_refs += 1
        by_element[el] = by_element.get(el, 0) + n_fracs
        by_type[atype] = by_type.get(atype, 0) + n_fracs

    print(f"  Total conditions:     {total_conds}")
    print(f"  With phase fractions: {conds_with_fracs}")
    print(f"  Match our references: {matches_refs} papers")
    print(f"  By element:           {by_element}")
    print(f"  By analysis type:     {by_type}")
    print(f"\n  Saved: {output_path}")
    print(f"  Papers cached in: {PAPERS_DIR}")

    if matches_refs > 0:
        print(f"\n  To generate benchmark scenarios from matching papers:")
        print(f"    python benchmark/generate_stage1_prompts_v2.py --input {output_path}")


if __name__ == "__main__":
    main()
