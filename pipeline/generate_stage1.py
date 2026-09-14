"""
V2: Structured prompt generation for Stage 1 benchmark scenarios.

Key difference from v1 (generate_stage1_prompts.py):
  - Prompts use a mandatory structured template with explicit fields
    (Material, Method, Precursors, Temperature, Duration, Atmosphere, Cooling, etc.)
  - Missing values are filled with domain-expert estimates marked "(typical)"
  - Adds machine-readable "synthesis_conditions" field with reported/inferred tracking
  - Goal: provide enough information for a domain expert to make quantitative predictions

Usage:
    python generate_stage1_prompts_v2.py
    python generate_stage1_prompts_v2.py --max-papers 5
    python generate_stage1_prompts_v2.py --input literature/fulltext_analyzed_papers.json
"""

import os
import json
import re
import argparse
import subprocess

GEMINI_PYTHON = (
    "/home/idies/workspace/Storage/xyu1/persistent"
    "/pytorch_env/sam3_gcloud/bin/python"
)
GEMINI_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "gemini_call.py"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LITERATURE_FILE = os.path.join(BASE_DIR, "extracted_papers.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "stage1_examples.json")

AVAILABLE_REFERENCES = """
Available reference spectra we can use for simulation:

Fe K-edge:
  - metallic_iron (Fe, metallic iron)
  - wustite (FeO, Fe2+)
  - hematite (Fe2O3, Fe3+)
  - siderite (FeCO3)
  - austenite (Fe, fcc metallic iron)

Cu K-edge:
  - metallic_copper (Cu)
  - cuprite (Cu2O, Cu+)
  - tenorite (CuO, Cu2+)

Cu L3-edge:
  - metallic_copper_L3 (Cu0)
  - cuprite_L3 (Cu2O, Cu+)
  - tenorite_L3 (CuO, Cu2+)

Ni K-edge:
  - metallic_nickel (Ni)
  - nickel_oxide (NiO, Ni2+)
  - lithium_nickelate (LiNiO2, Ni3+)

Ni L3-edge:
  - metallic_nickel_L3 (Ni0)
  - nickel_oxide_L3 (NiO, Ni2+)

Mn K-edge:
  - manganese_oxide_II (MnO, Mn2+)
  - manganese_dioxide (MnO2, Mn4+)
  - manganese_sesquioxide (Mn2O3, Mn3+)

Ti K-edge:
  - metallic_titanium (Ti)
  - rutile (TiO2, Ti4+)
  - titanium_sesquioxide (Ti2O3, Ti3+)
  - titanium_disulfide (TiS2)
  - strontium_titanate (SrTiO3, Ti4+)

Zn K-edge:
  - metallic_zinc (Zn)
  - zincite (ZnO, Zn2+)
  - sphalerite (ZnS)
  - zinc_chloride (ZnCl2)

S K-edge:
  - zinc_sulfide_S (ZnS)
  - pyrite_S (FeS2)
  - lithium_thiophosphate_S (Li3PS4)
  - titanium_disulfide_S (TiS2)

C K-edge:
  - graphite (C, sp2 carbon)

P K-edge:
  - lithium_thiophosphate_P (Li3PS4)

O K-edge:
  - rutile_O (TiO2)
  - strontium_titanate_O (SrTiO3)

Li K-edge:
  - lithium_titanate_Li (Li4Ti5O12)

Pt L3-edge:
  - metallic_platinum (Pt)
  - platinum_dioxide (PtO2, Pt4+)

Pd K-edge:
  - metallic_palladium (Pd)
  - palladium_oxide (PdO, Pd2+)
"""

# =============================================================================
# Category-specific benchmark prompt templates (what the tested LLM sees)
# =============================================================================

BENCHMARK_PROMPT_TEMPLATES = {
    "synthesis": """## Sample Information
- **Material**: [specific material name, composition, and form]
- **Application/Context**: [what this material is for]

## Synthesis / Preparation
- **Method**: [specific method]
- **Precursors**: [ALL precursor chemicals with molar ratios if known]
- **Temperature**: [value in °C. Write 'not reported' if not in paper]
- **Duration**: [value with units. Write 'not reported' if not in paper]
- **Heating rate**: [if known, or 'not reported']
- **Atmosphere**: [specific gas environment. Write 'not reported' if not in paper]
- **Cooling**: [method. Write 'not reported' if not in paper]
- **Post-treatment**: [any additional steps, or 'none']

## Sample-Specific Conditions
- [The KEY variable that distinguishes this sample]
- [Any other relevant details]

## Measurement
We plan to measure the {element} {edge} XANES spectrum of this sample.

## Questions
Based on ALL the synthesis and preparation information above:
1. What phases (oxidation states / compounds of {element}) could possibly exist in this sample?
2. How many of these phases are significant (>5% fraction)?
3. For each significant phase, estimate its fraction (fractions should sum to 1.0).
4. What reference spectra should be prepared for linear combination fitting?
5. What scientific reasoning connects the specific preparation conditions to your predicted phases and fractions?""",

    "electrochemistry": """## Sample Information
- **Material**: [electrode material name, composition, and form]
- **Application/Context**: [battery type, cell chemistry]

## Electrode / Cell Details
- **Electrode type**: [cathode / anode / working electrode]
- **Active material**: [chemical formula and structure]
- **Electrolyte**: [electrolyte composition]
- **Cell type**: [coin cell / pouch cell / in-situ cell / etc.]

## Electrochemical Conditions
- **Charge/discharge state**: [charging / discharging / charged / discharged / OCV]
- **C-rate**: [rate. Write 'not reported' if not in paper]
- **Voltage**: [voltage or voltage window. Write 'not reported' if not in paper]
- **State of charge**: [SOC or ion content x. Write 'not reported' if not in paper]
- **Cycle number**: [which cycle. Write 'not reported' if not in paper]
- **Capacity**: [specific capacity if known. Write 'not reported' if not in paper]

## Sample-Specific Conditions
- [The KEY variable that distinguishes this electrochemical state]
- [Measurement mode: operando / in-situ / ex-situ]
- [Any other relevant details]

## Measurement
We plan to measure the {element} {edge} XANES spectrum of this electrode.

## Questions
Based on ALL the electrochemical conditions above:
1. What phases (oxidation states / compounds of {element}) could exist in this electrode at this state?
2. How many of these phases are significant (>5% fraction)?
3. For each significant phase, estimate its fraction (fractions should sum to 1.0).
4. What reference spectra should be prepared for linear combination fitting?
5. What scientific reasoning connects the specific electrochemical conditions to your predicted phases and fractions?""",

    "pure_phase": """## Sample Information
- **Material**: [chemical formula and mineral/common name]
- **Crystal structure**: [crystal system and space group if known]
- **Oxidation state**: [formal oxidation state of the measured element]
- **Coordination**: [coordination geometry if known]

## Sample Details
- **Sample form**: [powder / foil / pellet / thin film / single crystal]
- **Purity**: [purity or source information]
- **Measurement mode**: [transmission / fluorescence / electron yield / EELS]

## Measurement
This is a {element} {edge} XANES measurement of a known pure-phase reference compound.

## Questions
Based on the compound identity, crystal structure, and electronic configuration:

1. **Spectral shape overview**: Describe the overall {element} {edge} XANES spectral shape for this compound. How many distinct peaks or features are visible in the near-edge region? Label them (e.g., A1, A2, B, pre-edge, white line) and describe their relative positions and intensities.

2. **Peak positions and energies**: For each labeled feature, provide the approximate energy position (in eV) or energy relative to the absorption edge. Include the edge position itself.

3. **Electronic structure origin**: For each major spectral feature, explain its physical origin in terms of electronic transitions and orbital character (e.g., which bands or states: t2g, eg, 4p, hybridized states). Which transitions are dipole-allowed vs. quadrupole or hybridization-enabled?

4. **Structural sensitivity**: How do the spectral features depend on the crystal structure, coordination geometry, or bond lengths? Would a different polymorph or structural variant show different features, and if so, which ones and why?

5. **Distinguishing features**: What specific spectral features (peak positions, relative intensities, presence/absence of features) distinguish this compound from other common {element} phases with different oxidation states or crystal structures?""",

    "thin_film": """## Sample Information
- **Material**: [film composition and substrate]
- **Application/Context**: [what this film is for]

## Deposition / Film Details
- **Deposition method**: [PLD / sputtering / ALD / CVD / etc.]
- **Substrate**: [substrate material]
- **Deposition temperature**: [value. Write 'not reported' if not in paper]
- **Film thickness**: [value. Write 'not reported' if not in paper]
- **Post-deposition annealing**: [conditions if any, or 'none']

## Sample-Specific Conditions
- [The KEY variable: composition coordinate, thickness, position on combinatorial library, etc.]
- [Any other relevant details]

## Measurement
We plan to measure the {element} {edge} XANES spectrum of this film.

## Questions
Based on ALL the deposition and film information above:
1. What phases (oxidation states / structural motifs of {element}) could exist in this film?
2. How many of these phases are significant (>5% fraction)?
3. For each significant phase, estimate its fraction (fractions should sum to 1.0).
4. What reference spectra should be prepared for linear combination fitting?
5. What scientific reasoning connects the specific film conditions to your predicted phases and fractions?""",

    "catalyst": """## Sample Information
- **Material**: [catalyst composition, e.g., Pt/Al2O3]
- **Application/Context**: [catalytic reaction or application]

## Catalyst Details
- **Active metal**: [metal and loading]
- **Support**: [support material]
- **Particle size**: [if known. Write 'not reported' if not in paper]
- **Preparation method**: [synthesis method]
- **Pre-treatment / Activation**: [conditions]

## Sample-Specific Conditions
- [The KEY variable: treatment conditions, reaction environment, particle size, etc.]
- [Measurement mode: operando / in-situ / ex-situ]
- [Any other relevant details]

## Measurement
We plan to measure the {element} {edge} XANES spectrum of this catalyst.

## Questions
Based on ALL the catalyst preparation and condition information above:
1. What phases (oxidation states / compounds of {element}) could exist in this catalyst?
2. How many of these phases are significant (>5% fraction)?
3. For each significant phase, estimate its fraction (fractions should sum to 1.0).
4. What reference spectra should be prepared for linear combination fitting?
5. What scientific reasoning connects the specific catalyst conditions to your predicted phases and fractions?""",

    "environmental": """## Sample Information
- **Material**: [sample type: soil, sediment, mineral, corroded metal, etc.]
- **Application/Context**: [environmental study context]

## Sample Origin and Conditions
- **Sample origin**: [geographic/site description]
- **Environmental conditions**: [pH, redox, temperature, moisture, depth, etc.]
- **Weathering/exposure history**: [time and conditions if known]
- **Spatial location**: [if spatially resolved, where on the sample]

## Sample-Specific Conditions
- [The KEY variable that distinguishes this sample]
- [Any other relevant details]

## Measurement
We plan to measure the {element} {edge} XANES spectrum of this sample.

## Questions
Based on ALL the environmental context above:
1. What phases (oxidation states / compounds of {element}) could exist in this sample?
2. How many of these phases are significant (>5% fraction)?
3. For each significant phase, estimate its fraction (fractions should sum to 1.0).
4. What reference spectra should be prepared for linear combination fitting?
5. What scientific reasoning connects the specific environmental conditions to your predicted phases and fractions?""",
}


def get_benchmark_template(category):
    """Get the benchmark prompt template for a given category."""
    return BENCHMARK_PROMPT_TEMPLATES.get(
        category, BENCHMARK_PROMPT_TEMPLATES["synthesis"]
    )


GENERATION_PROMPT = """You are an expert in XANES spectroscopy and benchmark design.

I need you to convert ONE specific sample condition from a published XANES paper
into a Stage 1 benchmark scenario that tests whether an LLM can generate a good
hypothesis BEFORE measurement.

PAPER CONTEXT:
  Element: {element}
  Edge: {edge}
  Category: {category}
  Key finding: {key_finding}
  Causal reasoning: {causal_reasoning}

THIS SPECIFIC CONDITION:
{condition_data}

Available reference spectra:
{references}

Generate exactly ONE benchmark scenario for this condition.
Return a SINGLE JSON object (not an array).

The prompt field MUST use this template (fill in the bracketed fields):
--- TEMPLATE START ---
{benchmark_template}
--- TEMPLATE END ---

CRITICAL RULES FOR THE PROMPT:
- Fill every field using ONLY information from the paper or condition data above.
- If a value is not reported in the paper, write "not reported" — do NOT invent values.
- NEVER use "(typical)" or "(typical for this method)" — only use actual reported values or "not reported".
- The prompt must NOT reveal the answer (no phase fractions, no XANES results).
- If the condition has "has_missing_context": true, INCLUDE an explicit "**Note**: Synthesis / preparation conditions unknown (cited reference [DOI] not accessible)." line in the Sample Information section. Do NOT invent synthesis details in that case. Propagate the flag to condition_details.has_missing_context in the output.

The JSON object has this structure:

{{
  "id": "{suggested_id}",
  "element": "{element}",
  "edge": "{edge}",
  "category": "{category}",
  "source_paper": "{source_paper}",

  "prompt": "<filled-in template from above>",

  "ground_truth": {{
    "candidate_phases": ["list all plausible phases"],
    "significant_phases": ["only the ones >5%"],
    "n_significant": integer,
    "fractions": {{"phase_name": fraction, ...}},
    "recommended_references": ["reference spectra to use"],
    "key_reasoning": "Why these specific conditions produce these phase fractions — connect the key experimental variables to the phase composition"
  }},

  "condition_details": {{
    "key_variable": "the main variable that distinguishes this sample from others",
    "key_variable_value": "the specific value of that variable",
    ... other category-specific fields from the condition data, null for unreported ...
  }},

  "rubric": {{
    "phase_identification": {{
      "max_score": 30,
      "criteria": [
        {{"points": number, "description": "what earns these points"}}
      ]
    }},
    "n_phases": {{
      "max_score": 10,
      "criteria": [
        {{"points": number, "description": "what earns these points"}}
      ]
    }},
    "fraction_accuracy": {{
      "max_score": 30,
      "criteria": [
        {{"points": 30, "description": "best prediction range"}},
        {{"points": 20, "description": "good prediction range"}},
        {{"points": 10, "description": "fair prediction range"}},
        {{"points": 0, "description": "poor prediction"}}
      ]
    }},
    "reasoning_quality": {{
      "max_score": 30,
      "criteria": [
        {{"points": number, "description": "specific reasoning element that earns points — must reference the category-specific variables (e.g., for electrochemistry: C-rate effect, voltage-phase relationship; for synthesis: temperature-phase diagram, atmosphere effect)"}}
      ]
    }}
  }}
}}

IMPORTANT RULES:
1. Ground truth fractions must use ONLY phases that map to our available references.
   Map the paper's phases to reference names. If phases don't map to ANY reference,
   include them but note "outside_references": true.

2. Fractions must sum to 1.0 (normalize if needed). Convert percentages to decimals.

3. The prompt must NOT reveal the answer (no phase fractions, no XANES results).

4. CRITICAL — Prompt rules:
   a. The prompt MUST follow the category-specific template exactly.
   b. Use ONLY values reported in the paper or condition data.
   c. If the paper does not report a specific value, write "not reported".
   d. NEVER use "(typical)" or invent values not in the paper.
   e. Chemical formulas must be exact, not vague.

5. key_reasoning must explain the causal link between the category-specific
   experimental variables and the resulting phase fractions.

Return ONLY a single JSON object, no markdown fences, no extra text."""


# =============================================================================
# Pure-phase generation prompt (spectral fingerprint questions)
# =============================================================================

GENERATION_PROMPT_PURE_PHASE = """You are an expert in XANES spectroscopy and benchmark design.

I need you to convert ONE pure-phase reference compound from a published XANES paper
into a Stage 1 benchmark scenario that tests whether an LLM has deep XANES spectral
knowledge — at the level of specific peak labels, energy positions, orbital origins,
and structural sensitivity (like a detailed spectroscopy textbook or review paper).

PAPER CONTEXT:
  Element: {element}
  Edge: {edge}
  Key finding: {key_finding}

THIS SPECIFIC COMPOUND:
{condition_data}

Available reference spectra:
{references}

Generate exactly ONE benchmark scenario for this compound.
Return a SINGLE JSON object (not an array).

The prompt field MUST use this template (fill in the bracketed fields):
--- TEMPLATE START ---
{benchmark_template}
--- TEMPLATE END ---

CRITICAL RULES FOR THE PROMPT:
- Fill every field using ONLY information from the paper or condition data above.
- If a value is not in the paper, write "not reported".
- NEVER use "(typical)" or invent values.
- The prompt should NOT reveal spectral features — those are the answer.

The JSON object has this structure:

{{
  "id": "{suggested_id}",
  "element": "{element}",
  "edge": "{edge}",
  "category": "pure_phase",
  "source_paper": "{source_paper}",

  "prompt": "<filled-in template from above>",

  "ground_truth": {{
    "compound": "chemical formula",
    "oxidation_state": "formal oxidation state",
    "crystal_structure": "crystal system / space group",
    "coordination": "coordination geometry and number",
    "spectral_features": {{
      "edge_position_eV": "edge energy in eV",
      "labeled_peaks": [
        {{
          "label": "peak label (e.g., A1, pre-edge, white line, B)",
          "energy_eV": "approximate energy in eV or relative to edge",
          "intensity": "relative intensity (weak/moderate/strong)",
          "origin": "electronic transition and orbital character (e.g., 1s->3d t2g, 2p->4s continuum)",
          "selection_rule": "dipole-allowed / quadrupole / hybridization-enabled",
          "source": "paper_data or standard_knowledge"
        }}
      ],
      "spectral_shape_summary": "overall shape description — number of peaks, their pattern, key visual characteristics",
      "structural_sensitivity": "how features change with different polymorphs, coordination, or bond lengths (e.g., anatase vs rutile, octahedral vs tetrahedral)",
      "distinguishing_features": "specific features that distinguish this compound from other common {element} phases"
    }},
    "key_reasoning": "Connect the electronic configuration and crystal structure to each spectral feature — why this compound produces this specific spectral fingerprint"
  }},

  "condition_details": {{
    "compound": "chemical formula",
    "crystal_structure": "crystal system",
    "oxidation_state": "formal oxidation state",
    "coordination": "coordination geometry",
    "sample_form": "powder / foil / pellet / etc.",
    "measurement_mode": "transmission / fluorescence / electron yield / EELS"
  }},

  "rubric": {{
    "spectral_shape": {{
      "max_score": 20,
      "criteria": [
        {{"points": number, "description": "correctly describes overall spectral shape and identifies the right number of distinct features"}},
        {{"points": number, "description": "provides meaningful peak labels and correct relative positions/intensities"}}
      ]
    }},
    "peak_positions": {{
      "max_score": 20,
      "criteria": [
        {{"points": 20, "description": "correct edge position and peak energies within 1-2 eV"}},
        {{"points": 10, "description": "correct edge position within 3 eV, approximate peak energies"}},
        {{"points": 5, "description": "correct relative ordering of features only"}},
        {{"points": 0, "description": "wrong positions"}}
      ]
    }},
    "electronic_origin": {{
      "max_score": 25,
      "criteria": [
        {{"points": number, "description": "correctly identifies the electronic transitions and orbital character for each major feature (e.g., t2g/eg bands, 3d-4p hybridization, dipole vs quadrupole)"}},
        {{"points": number, "description": "correctly explains selection rules and why certain features are present/absent"}}
      ]
    }},
    "structural_sensitivity": {{
      "max_score": 15,
      "criteria": [
        {{"points": number, "description": "correctly describes how features change with different polymorphs or coordination environments"}},
        {{"points": number, "description": "identifies which specific peaks are sensitive to structural changes and why"}}
      ]
    }},
    "distinguishing_features": {{
      "max_score": 20,
      "criteria": [
        {{"points": number, "description": "correctly identifies key spectral differences from other common {element} phases"}},
        {{"points": number, "description": "names specific features (peak presence/absence, intensity ratios, energy shifts) that are diagnostic"}}
      ]
    }}
  }}
}}

IMPORTANT:
- The ground truth spectral features MUST come from the paper's description of this
  compound's spectrum. Use the extracted condition data fields: edge_position_eV,
  labeled_peaks, pre_edge_description, white_line_description, spectral_shape_summary,
  electronic_structure_interpretation, structural_sensitivity, distinguishing_features.
- For labeled_peaks: extract as many individually identifiable peaks as the paper describes.
  If the paper labels peaks (A1, A2, B, etc.), use those labels. If not, use descriptive
  labels (pre-edge, white line, post-edge shoulder, etc.).
- If the paper does not describe a specific spectral feature, you may supplement with
  well-established textbook XANES knowledge, but mark those as "source": "standard_knowledge".
- Do NOT fabricate spectral features. If a feature is unknown, write null.
- The ground truth should be detailed enough to evaluate whether an LLM truly knows this
  compound's spectral fingerprint at a research-paper level, not just textbook level.

Return ONLY a single JSON object, no markdown fences, no extra text."""


# =============================================================================
# Classify a paper from its extracted metadata (no fulltext needed)
# =============================================================================

CLASSIFY_FROM_METADATA_PROMPT = """You are an expert in XANES spectroscopy. Classify this paper
based on its metadata and extracted conditions.

Title: {title}
Element: {element}
Edge: {edge}
Key finding: {key_finding}
Application: {application}

Sample conditions summary:
{conditions_summary}

Classify into ONE category:
1. "synthesis" — Phases depend on synthesis conditions (temperature, precursors, atmosphere)
2. "electrochemistry" — Phases depend on electrochemical state (charge/discharge, C-rate, voltage, SOC, cycling)
3. "pure_phase" — Known pure-phase reference compounds (standards, 100% single phase)
4. "thin_film" — Phases depend on deposition or spatial composition gradients
5. "catalyst" — Phases depend on catalyst preparation, support, particle size, reaction conditions
6. "environmental" — Phases from natural/environmental samples

HINT: If ALL conditions from this paper have exactly one phase at fraction 1.0,
it is likely "pure_phase". If conditions mention charge/discharge, voltage, C-rate,
SOC, cycling, or operando — it is "electrochemistry".

Return ONLY a JSON object:
{{
  "category": "synthesis|electrochemistry|pure_phase|thin_film|catalyst|environmental",
  "confidence": "high|medium|low",
  "reasoning": "one-sentence explanation"
}}"""


def classify_paper_from_metadata(paper):
    """Classify a paper using its extracted metadata (no fulltext needed)."""
    a = paper.get("fulltext_analysis", {})
    conditions = a.get("sample_conditions", [])

    # Quick heuristic: all single-phase → pure_phase
    all_single = all(
        len(c.get("phase_fractions", [])) == 1
        and abs(c["phase_fractions"][0].get("fraction", 0) - 1.0) < 0.05
        for c in conditions
        if isinstance(c.get("phase_fractions"), list) and c["phase_fractions"]
    )

    # Quick heuristic: battery keywords
    text_blob = json.dumps(conditions).lower()
    title_lower = (paper.get("title", "") or "").lower()
    app_lower = (paper.get("application", "") or "").lower()
    combined = text_blob + " " + title_lower + " " + app_lower
    battery_kw = ["cycling", "charge", "discharge", "c-rate", "c_rate",
                  "voltage", "capacity", "operando", "lithiation",
                  "delithiation", "sodiation", "battery", "cathode",
                  "anode", "electrochemical", "state of charge", "soc"]
    is_battery = any(kw in combined for kw in battery_kw)

    thin_film_kw = ["thin film", "pld", "sputtering", "deposition", "ald",
                    "combinatorial", "substrate", "film thickness"]
    is_thin_film = any(kw in combined for kw in thin_film_kw)

    catalyst_kw = ["catalyst", "nanoparticle", "support", "loading",
                   "impregnation", "cluster", "thiolate"]
    is_catalyst = any(kw in combined for kw in catalyst_kw)

    # Decide based on heuristics first (faster, no API call)
    if all_single and not is_battery:
        return {"category": "pure_phase", "confidence": "high",
                "reasoning": "all conditions are single-phase (fraction=1.0)"}
    if is_battery:
        return {"category": "electrochemistry", "confidence": "high",
                "reasoning": "paper contains battery/electrochemistry keywords"}
    if is_thin_film and not is_battery:
        return {"category": "thin_film", "confidence": "medium",
                "reasoning": "paper contains thin film/deposition keywords"}
    if is_catalyst:
        return {"category": "catalyst", "confidence": "medium",
                "reasoning": "paper contains catalyst/nanoparticle keywords"}

    # Fall back to Gemini classification
    conditions_summary = ""
    for c in conditions[:5]:
        sid = c.get("sample_id", "?")
        desc = c.get("description", c.get("material", "?"))[:80]
        pf = c.get("phase_fractions", [])
        frac_str = ", ".join(
            f"{x['phase']}={x['fraction']}" for x in pf
        ) if isinstance(pf, list) else "?"
        conditions_summary += f"  - {sid}: {desc} | {frac_str}\n"

    prompt = CLASSIFY_FROM_METADATA_PROMPT.format(
        title=paper.get("title", ""),
        element=a.get("element", "?"),
        edge=a.get("edge", "K-edge"),
        key_finding=a.get("key_finding", ""),
        application=paper.get("application", ""),
        conditions_summary=conditions_summary,
    )

    raw = call_gemini(prompt)
    result = parse_json(raw) if raw else None
    if result and result.get("category"):
        return result

    return {"category": "synthesis", "confidence": "low",
            "reasoning": "classification failed, defaulting to synthesis"}


GEMINI_MODEL = None  # set from CLI in main()

def call_gemini(prompt):
    """Call Gemini via subprocess."""
    cmd = [GEMINI_PYTHON, GEMINI_SCRIPT]
    if GEMINI_MODEL:
        cmd += ["--model", GEMINI_MODEL]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"    subprocess error: {result.stderr[:200]}")
            return None
        response = json.loads(result.stdout.strip())
        if not response.get("ok"):
            print(f"    Gemini error: {response.get('error', 'unknown')}")
            return None
        return response["text"]
    except subprocess.TimeoutExpired:
        print("    subprocess timed out")
        return None
    except Exception as e:
        print(f"    subprocess exception: {e}")
        return None


def parse_json(raw_text):
    """Parse JSON from Gemini response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try as array
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(text[start_arr:end_arr + 1])
        except json.JSONDecodeError:
            pass

    # Try as single object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    print(f"    JSON parse error: {text[:200]}...")
    return None


def validate_scenarios(scenarios):
    """Check for duplicates, conflicts, and inconsistencies. Auto-fix where possible."""
    from collections import Counter, defaultdict
    n_issues = 0

    # --- 1. Duplicate IDs: auto-fix by appending suffix ---
    id_counts = Counter(s["id"] for s in scenarios)
    dupes = {k for k, v in id_counts.items() if v > 1}
    if dupes:
        seen = {}
        for s in scenarios:
            sid = s["id"]
            if sid in dupes:
                if sid not in seen:
                    seen[sid] = 1
                else:
                    seen[sid] += 1
                    new_id = f"{sid}_{seen[sid]}"
                    print(f"  FIX duplicate ID: '{sid}' -> '{new_id}'")
                    s["id"] = new_id
                    n_issues += 1
    else:
        print("  Duplicate IDs: none")

    # --- 2. Identical fractions within same paper + same element/edge ---
    paper_groups = defaultdict(list)
    for s in scenarios:
        paper = s.get("source_paper", "")[:60]
        elem = s.get("element", "")
        edge = s.get("edge", "")
        key = (paper, elem, edge)
        gt = s.get("ground_truth", {})
        fracs = gt.get("fractions", {})
        paper_groups[key].append({
            "id": s["id"],
            "fracs": fracs,
            "fracs_str": json.dumps(fracs, sort_keys=True),
        })

    ids_to_remove = set()
    frac_dupes = 0
    for (paper, elem, edge), items in paper_groups.items():
        frac_groups = defaultdict(list)
        for it in items:
            frac_groups[it["fracs_str"]].append(it["id"])
        for frac_str, ids in frac_groups.items():
            if len(ids) > 1 and frac_str != "{}":
                keep = ids[0]
                remove = ids[1:]
                print(f"  REMOVING redundant: {len(ids)} scenarios with same fractions={frac_str}")
                print(f"    KEEP:   {keep}")
                for i in remove:
                    print(f"    REMOVE: {i}")
                    ids_to_remove.add(i)
                frac_dupes += len(remove)
                n_issues += 1
    if ids_to_remove:
        scenarios = [s for s in scenarios if s["id"] not in ids_to_remove]
        print(f"  Removed {len(ids_to_remove)} redundant scenario(s), {len(scenarios)} remain")
    else:
        print("  Redundant fractions: none")

    # --- 3. Conflicting scenarios: same sample description, different fractions ---
    conflicts = 0
    for (paper, elem, edge), items in paper_groups.items():
        for i, a in enumerate(items):
            for b in items[i+1:]:
                if a["id"] in ids_to_remove or b["id"] in ids_to_remove:
                    continue
                # Only flag if IDs are exactly the same (after dedup fix)
                # or one is the _2 suffix version of the other
                if a["id"] == b["id"] or a["id"] + "_2" == b["id"] or b["id"] + "_2" == a["id"]:
                    if a["fracs_str"] != b["fracs_str"]:
                        print(f"  CONFLICT: same condition '{a['id']}' but different fractions:")
                        print(f"    KEEP:   {a['id']}: {a['fracs_str']}")
                        print(f"    REMOVE: {b['id']}: {b['fracs_str']}")
                        ids_to_remove.add(b["id"])
                        conflicts += 1
                        n_issues += 1
    if conflicts > 0:
        scenarios = [s for s in scenarios if s["id"] not in ids_to_remove]
        print(f"  Removed {conflicts} conflicting scenario(s), {len(scenarios)} remain")
    else:
        print("  Conflicting scenarios: none")

    # --- 4. Element/edge consistency: prompt must mention the right element ---
    elem_issues = 0
    for s in scenarios:
        elem = s.get("element", "")
        edge = s.get("edge", "")
        prompt = s.get("prompt", "")
        cat = s.get("category", "")

        if cat != "pure_phase":
            expected_measurement = f"{elem} {edge}"
            if expected_measurement not in prompt:
                alt = f"{elem} {edge}-edge"
                if alt not in prompt:
                    print(f"  ELEMENT MISMATCH: {s['id']} — element={elem} edge={edge} "
                          f"not found in prompt measurement section")
                    elem_issues += 1
                    n_issues += 1
    if elem_issues == 0:
        print("  Element/edge consistency: OK")

    # --- 5. Fraction sum check (non-pure-phase) ---
    sum_issues = 0
    junk_keys = {"outside_references", "is_reference", "fitting_method", "notes",
                 "reasoning", "confidence", "r_factor"}
    for s in scenarios:
        if s.get("category") == "pure_phase":
            continue
        gt = s.get("ground_truth", {})
        fracs = gt.get("fractions", {})
        if not fracs:
            continue
        # Strip non-numeric junk that Gemini sometimes leaks into fractions
        cleaned = {}
        stripped = []
        for k, v in fracs.items():
            if k in junk_keys or not isinstance(v, (int, float)) or isinstance(v, bool):
                stripped.append((k, v))
                continue
            cleaned[k] = v
        if stripped:
            print(f"  STRIPPED junk from {s['id']}: {stripped}")
            gt["fractions"] = cleaned
            fracs = cleaned
        if not fracs:
            continue
        total = sum(fracs.values())
        if abs(total - 1.0) > 0.05:
            print(f"  BAD SUM: {s['id']} — fractions sum to {total:.3f}: {fracs}")
            sum_issues += 1
            n_issues += 1
    if sum_issues == 0:
        print("  Fraction sums: all OK")

    # --- 6. Missing ground truth fields ---
    missing = 0
    for s in scenarios:
        gt = s.get("ground_truth", {})
        cat = s.get("category", "")
        if cat == "pure_phase":
            if not gt.get("compound"):
                print(f"  MISSING: {s['id']} — no compound in ground truth")
                missing += 1
                n_issues += 1
        else:
            if not gt.get("fractions"):
                print(f"  MISSING: {s['id']} — no fractions in ground truth")
                missing += 1
                n_issues += 1
            if not gt.get("key_reasoning"):
                print(f"  MISSING: {s['id']} — no key_reasoning in ground truth")
                missing += 1
                n_issues += 1
    if missing == 0:
        print("  Ground truth completeness: OK")

    print(f"\n  Total issues: {n_issues}")
    return scenarios, n_issues


def harmonize_paper_phases(paper_scenarios):
    """Normalize phase names across scenarios from the same paper.

    Finds cases like "core site" vs "Pd in core site" and uses the longer form.
    Only renames when the short form matches at a word boundary in the long form
    (e.g. "core site" in "Pd in core site"), not chemical substrings
    (e.g. "TiS2" should NOT match "NaTiS2").
    """
    if len(paper_scenarios) < 2:
        return paper_scenarios, 0

    all_phases = set()
    for s in paper_scenarios:
        gt = s.get("ground_truth", {})
        for key in ("fractions", "candidate_phases", "significant_phases"):
            val = gt.get(key, {})
            if isinstance(val, dict):
                all_phases.update(val.keys())
            elif isinstance(val, list):
                all_phases.update(val)

    phase_list = sorted(all_phases)
    rename_map = {}
    for i, a in enumerate(phase_list):
        for b in phase_list[i+1:]:
            if a == b:
                continue
            short, long = (a, b) if len(a) < len(b) else (b, a)
            if short.lower() not in long.lower():
                continue
            # Must match at word boundary: "core site" in "Pd in core site" OK,
            # "TiS2" in "NaTiS2" NOT OK
            # Skip if long form contains metadata artifacts
            if "outside_references" in long or "true)" in long or "false)" in long:
                continue
            # Skip if short form is a complete chemical formula inside a longer phrase
            # e.g. "Li4Ti5O12" inside "stable configurations (Li4Ti5O12/Li7Ti5O12)"
            if re.match(r'^[A-Z][a-z]?\d', short) and "(" in long:
                continue
            pattern = r'(?<![A-Za-z0-9])' + re.escape(short) + r'(?![A-Za-z0-9])'
            if re.search(pattern, long):
                rename_map[short] = long

    if not rename_map:
        return paper_scenarios, 0

    # Check for collisions: if two keys map to the same target, skip both
    targets = {}
    collisions = set()
    for src, tgt in rename_map.items():
        if tgt in targets:
            collisions.add(tgt)
        targets[tgt] = src
    rename_map = {k: v for k, v in rename_map.items() if v not in collisions}

    if not rename_map:
        return paper_scenarios, 0

    fixes = 0
    for s in paper_scenarios:
        gt = s.get("ground_truth", {})
        for key in ("fractions", "candidate_phases", "significant_phases"):
            val = gt.get(key)
            if isinstance(val, dict):
                new_dict = {}
                for k, v in val.items():
                    new_key = rename_map.get(k, k)
                    # Also skip if rename would collide with existing key
                    if new_key != k and new_key in val:
                        new_key = k
                    if new_key != k:
                        fixes += 1
                    new_dict[new_key] = v
                gt[key] = new_dict
            elif isinstance(val, list):
                new_list = []
                for item in val:
                    new_item = rename_map.get(item, item)
                    if new_item != item:
                        fixes += 1
                    new_list.append(new_item)
                gt[key] = new_list

    if fixes > 0:
        renames = ", ".join(f'"{k}" -> "{v}"' for k, v in rename_map.items())
        print(f"    Phase harmonization: {fixes} fixes ({renames})")

    return paper_scenarios, fixes


HARMONIZE_PROMPT = """You are reviewing benchmark scenarios generated from a single XANES paper.
Each scenario has phase fractions (dict of phase_name -> fraction).
Different scenarios may use inconsistent names for the SAME phase (e.g. "Pd staple site" vs "Pd in staple site" vs "staple motif").

Your task: return a canonical rename map that unifies equivalent phase names.
- DO NOT merge distinct phases (e.g. "core site" and "surface site" stay separate).
- DO NOT change fraction values.
- Prefer the most specific/descriptive form as canonical.
- Only include entries in the map when a rename is needed.

Input phase names (from this paper's scenarios):
{phase_list}

Return ONLY a JSON object mapping old_name -> canonical_name. No explanations.
If no renames are needed, return {{}}.
Example: {{"Pd staple site": "Pd in staple site", "staple motif": "Pd in staple site"}}
"""


def harmonize_via_gemini(paper_scenarios):
    """Use Gemini to unify inconsistent phase names across a paper's scenarios."""
    if len(paper_scenarios) < 2:
        return 0

    all_phases = set()
    for s in paper_scenarios:
        gt = s.get("ground_truth", {})
        fracs = gt.get("fractions", {})
        if isinstance(fracs, dict):
            all_phases.update(fracs.keys())

    if len(all_phases) < 2:
        return 0

    prompt = HARMONIZE_PROMPT.format(phase_list=json.dumps(sorted(all_phases), indent=2))
    raw = call_gemini(prompt)
    if not raw:
        return 0

    rename_map = parse_json(raw)
    if not isinstance(rename_map, dict) or not rename_map:
        return 0

    # Skip renames that would cause collisions within one scenario
    fixes = 0
    for s in paper_scenarios:
        gt = s.get("ground_truth", {})
        for key in ("fractions", "candidate_phases", "significant_phases"):
            val = gt.get(key)
            if isinstance(val, dict):
                new_dict = {}
                for k, v in val.items():
                    new_key = rename_map.get(k, k)
                    if new_key != k and new_key in val and new_key not in new_dict:
                        new_key = k
                    if new_key in new_dict:
                        new_dict[new_key] += v
                    else:
                        new_dict[new_key] = v
                    if new_key != k:
                        fixes += 1
                gt[key] = new_dict
            elif isinstance(val, list):
                new_list = []
                seen = set()
                for item in val:
                    new_item = rename_map.get(item, item)
                    if new_item != item:
                        fixes += 1
                    if new_item not in seen:
                        new_list.append(new_item)
                        seen.add(new_item)
                gt[key] = new_list

    if fixes > 0:
        renames = ", ".join(f'"{k}" -> "{v}"' for k, v in rename_map.items())
        print(f"    Gemini harmonization: {fixes} fixes ({renames})")

    return fixes


def check_paper_consistency(scenarios, paper_start_idx):
    """Run all per-paper checks on scenarios generated from one paper.

    1. Harmonize phase names (rule-based + Gemini)
    2. Detect near-duplicate prompts (same material + conditions → keep first)
    3. Flag conflicting fractions for same element/edge
    Returns updated scenarios list and number of removed scenarios.
    """
    paper_scenarios = scenarios[paper_start_idx:]
    if len(paper_scenarios) < 2:
        return scenarios, 0

    # 1a. Rule-based harmonization (fast, catches simple cases)
    harmonize_paper_phases(paper_scenarios)
    # 1b. Gemini-based harmonization (handles semantic equivalence)
    harmonize_via_gemini(paper_scenarios)

    # 2. Detect near-duplicate prompts by comparing key fields
    def prompt_signature(s):
        p = s.get("prompt", "")
        lines = []
        for line in p.split("\n"):
            line = line.strip()
            if line.startswith("- **Material**"):
                lines.append(line)
            elif line.startswith("- **Deposition method**"):
                lines.append(line)
            elif line.startswith("- **Film thickness**"):
                lines.append(line)
            elif line.startswith("- **Deposition temperature**"):
                lines.append(line)
            elif line.startswith("- **Electrode"):
                lines.append(line)
            elif line.startswith("- **Active metal**"):
                lines.append(line)
            elif line.startswith("- **Compound**"):
                lines.append(line)
            elif line.startswith("- **State of charge**"):
                lines.append(line)
            elif line.startswith("- Treatment"):
                lines.append(line)
            elif line.startswith("- Composition"):
                lines.append(line)
        elem = s.get("element", "")
        edge = s.get("edge", "")
        return elem + "|" + edge + "|" + "|".join(sorted(lines))

    seen_sigs = {}
    to_remove = set()
    sig_to_indices = {}
    for idx, s in enumerate(paper_scenarios):
        sig = prompt_signature(s)
        sig_to_indices.setdefault(sig, []).append(idx)
        if sig in seen_sigs:
            first = seen_sigs[sig]
            # Same input: check if output is also same
            fi = paper_scenarios[first].get("ground_truth", {}).get("fractions", {})
            fj = s.get("ground_truth", {}).get("fractions", {})
            if fi == fj:
                print(f"    REDUNDANT (same in → same out): #{paper_start_idx + idx + 1} "
                      f"'{s.get('id','?')[:50]}' duplicates #{paper_start_idx + first + 1} — removing")
            else:
                print(f"    INCONSISTENT (same in → diff out): #{paper_start_idx + idx + 1} "
                      f"vs #{paper_start_idx + first + 1}")
                print(f"      #{paper_start_idx + first + 1}: {fi}")
                print(f"      #{paper_start_idx + idx + 1}: {fj}")
                print(f"      → keeping first, removing latter (needs manual review)")
            to_remove.add(paper_start_idx + idx)
        else:
            seen_sigs[sig] = idx

    # 3. Check for identical fractions across ALL pairs (different input → same output)
    #    Includes cross-element pairs (e.g. Ti K vs Zn K of same sample)
    active = [(idx, s) for idx, s in enumerate(paper_scenarios)
              if (paper_start_idx + idx) not in to_remove]
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            idx_i, si = active[i]
            idx_j, sj = active[j]
            if paper_start_idx + idx_j in to_remove:
                continue
            fi = si.get("ground_truth", {}).get("fractions", {})
            fj = sj.get("ground_truth", {}).get("fractions", {})
            if not (fi and fj and fi == fj):
                continue
            sig_i = prompt_signature(si)
            sig_j = prompt_signature(sj)
            if sig_i == sig_j:
                continue  # already handled above
            elem_i, edge_i = si.get("element", ""), si.get("edge", "")
            elem_j, edge_j = sj.get("element", ""), sj.get("edge", "")
            print(f"    SUSPICIOUS (diff in → same out): "
                  f"#{paper_start_idx + idx_i + 1} ({elem_i} {edge_i}) and "
                  f"#{paper_start_idx + idx_j + 1} ({elem_j} {edge_j}) "
                  f"have identical fractions but different conditions")
            print(f"      #{paper_start_idx + idx_i + 1}: {si.get('id','?')[:60]}")
            print(f"      #{paper_start_idx + idx_j + 1}: {sj.get('id','?')[:60]}")
            print(f"      fractions: {fi}")
            print(f"      → removing latter (needs manual review)")
            to_remove.add(paper_start_idx + idx_j)

    if to_remove:
        scenarios = [s for i, s in enumerate(scenarios) if i not in to_remove]
        print(f"    Removed {len(to_remove)} redundant/conflicting scenario(s)")

    return scenarios, len(to_remove)


def resolve_condition_elements(condition, paper_element, paper_edge, all_conditions=None):
    """Resolve per-condition element/edge for multi-element papers.

    Returns list of (element, edge, id_suffix) tuples.
    Single-element papers return one tuple with empty suffix.
    Multi-element papers try: sample_id regex, material field, formula start, then split.

    Method 3 (split) only fires if the paper's extraction contains per-edge data,
    detected by ANY condition having an "(X K-edge)" tag in its sample_id.
    Otherwise the split would produce redundant scenarios with identical fractions.
    """
    sample_id = condition.get("sample_id", "")
    elements = [e.strip() for e in paper_element.split(",")]

    if len(elements) <= 1:
        return [(paper_element, paper_edge, "")]

    edge_match = re.search(r'\((\w+)\s+(K|L\d?)-edge\)', sample_id)
    if edge_match:
        return [(edge_match.group(1), edge_match.group(2) + "-edge", "")]

    material = condition.get("material") or ""
    if material:
        found = [e for e in elements if e.lower() in material.lower()]
        if len(found) == 1:
            return [(found[0], paper_edge, "")]
        # Binary compound without hyphen: first element is primary absorber
        if len(found) > 1 and "-" not in material:
            starts = [e for e in elements if material.startswith(e)]
            if len(starts) == 1:
                return [(starts[0], paper_edge, "")]

    # Method 3: only split if paper has per-edge data
    has_per_edge = False
    if all_conditions:
        for c in all_conditions:
            if re.search(r'\((\w+)\s+(K|L\d?)-edge\)', c.get("sample_id", "")):
                has_per_edge = True
                break

    if not has_per_edge:
        # Keep as single scenario using paper-level element (e.g. "Ti, Zn")
        return [(paper_element, paper_edge, "")]

    results = []
    for e in elements:
        suffix = f" ({e} {paper_edge})"
        results.append((e, paper_edge, suffix))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 benchmark scenarios using Gemini"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to fulltext_analyzed_papers*.json (default: latest)")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--model", type=str, default=None,
                        help="Gemini model (e.g. gemini-3.8-flash). Default: gemini_call.py's default.")
    args = parser.parse_args()

    global GEMINI_MODEL
    GEMINI_MODEL = args.model
    if GEMINI_MODEL:
        print(f"Using Gemini model: {GEMINI_MODEL}")

    input_file = args.input or LITERATURE_FILE
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    print(f"Reading: {input_file}")

    with open(input_file) as f:
        papers = json.load(f)

    # Filter to papers that have sample conditions with fractions
    usable = []
    for p in papers:
        a = p.get("fulltext_analysis", {})
        conditions = a.get("sample_conditions", [])
        if not conditions:
            continue
        has_fractions = any(
            isinstance(c.get("phase_fractions"), list) and len(c["phase_fractions"]) > 0
            for c in conditions
        )
        if not has_fractions:
            continue
        usable.append(p)

    print(f"Papers with reported phase fractions: {len(usable)}")

    if args.max_papers:
        usable = usable[:args.max_papers]

    print(f"Processing: {len(usable)} papers\n")
    print("=" * 60)

    # Count total conditions across all papers
    total_conditions = 0
    for p in usable:
        conditions = p.get("fulltext_analysis", {}).get("sample_conditions", [])
        total_conditions += sum(
            1 for c in conditions
            if isinstance(c.get("phase_fractions"), list) and len(c["phase_fractions"]) > 0
        )
    print(f"Total conditions with phase fractions: {total_conditions}\n")
    print("=" * 60)

    scenarios = []
    cond_idx = 0
    for i, paper in enumerate(usable):
        a = paper["fulltext_analysis"]
        title = paper.get("title", "?")[:60]
        elem = a.get("element", "?")
        edge = a.get("edge", "K-edge")

        # Get category: from existing classification, or classify now
        classification = (
            a.get("classification")
            or paper.get("classification")
            or None
        )
        if not classification:
            classification = classify_paper_from_metadata(paper)
            paper["classification"] = classification
        category = classification.get("category", "synthesis")

        paper_meta = {
            "title": paper.get("title", ""),
            "doi": paper.get("doi", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
        }

        conditions = a.get("sample_conditions", [])
        conditions_with_fracs = [
            c for c in conditions
            if isinstance(c.get("phase_fractions"), list) and len(c["phase_fractions"]) > 0
        ]

        if not conditions_with_fracs:
            continue

        print(f"\n  [{i+1}/{len(usable)}] {elem} [{category}] | {title} ({len(conditions_with_fracs)} conditions)")

        # Get the benchmark prompt template for this category
        benchmark_template = get_benchmark_template(category)

        paper_ok = 0
        paper_start_idx = len(scenarios)
        for ci, condition in enumerate(conditions_with_fracs):
            sample_id = condition.get("sample_id", f"sample_{ci}")
            paper_short = (paper.get("title", "paper") or "paper")[:20].lower()
            paper_short = "".join(c if c.isalnum() else "_" for c in paper_short).strip("_")

            resolved = resolve_condition_elements(condition, elem, edge, conditions_with_fracs)

            for cond_elem, cond_edge, id_suffix in resolved:
                cond_idx += 1
                effective_id = sample_id + id_suffix
                suggested_id = f"{paper_short}_{effective_id}"

                condition_data = json.dumps(condition, indent=2)

                # Route reference-material conditions through the pure_phase prompt
                effective_category = category
                if condition.get("is_reference") is True:
                    effective_category = "pure_phase"
                effective_template = (get_benchmark_template("pure_phase")
                                       if effective_category == "pure_phase"
                                       else benchmark_template)

                if effective_category == "pure_phase":
                    prompt = GENERATION_PROMPT_PURE_PHASE.format(
                        element=cond_elem,
                        edge=cond_edge,
                        key_finding=a.get("key_finding", ""),
                        condition_data=condition_data,
                        references=AVAILABLE_REFERENCES,
                        suggested_id=suggested_id,
                        source_paper=paper.get("title", "")[:80],
                        benchmark_template=effective_template.format(
                            element=cond_elem, edge=cond_edge,
                        ),
                    )
                else:
                    prompt = GENERATION_PROMPT.format(
                        element=cond_elem,
                        edge=cond_edge,
                        category=category,
                        key_finding=a.get("key_finding", ""),
                        causal_reasoning=a.get("causal_reasoning", "")[:500],
                        condition_data=condition_data,
                        references=AVAILABLE_REFERENCES,
                        suggested_id=suggested_id,
                        source_paper=paper.get("title", "")[:80],
                        benchmark_template=benchmark_template.format(
                            element=cond_elem, edge=cond_edge,
                        ),
                    )

                raw = call_gemini(prompt)
                if not raw:
                    print(f"    [{cond_idx}/{total_conditions}] {effective_id}: FAILED (Gemini call)")
                    continue

                parsed = parse_json(raw)
                if parsed:
                    if isinstance(parsed, list):
                        parsed = parsed[0] if parsed else None
                if parsed:
                    parsed["source_paper_full"] = paper_meta
                    parsed["category"] = effective_category
                    parsed["classification"] = classification
                    scenarios.append(parsed)
                    gt = parsed.get("ground_truth", {})
                    if effective_category == "pure_phase":
                        compound = gt.get("compound", "?")
                        print(f"    [{cond_idx}/{total_conditions}] OK: {parsed.get('id', '?')} — {compound} (spectral fingerprint)")
                    else:
                        fracs = gt.get("fractions", {})
                        frac_str = ", ".join(f"{k}={v}" for k, v in fracs.items())
                        print(f"    [{cond_idx}/{total_conditions}] OK: {parsed.get('id', '?')} — {frac_str}")
                    paper_ok += 1
                else:
                    print(f"    [{cond_idx}/{total_conditions}] {effective_id}: FAILED (parse)")

                import time
                time.sleep(2)

        print(f"    Generated {paper_ok}/{len(conditions_with_fracs)} scenarios from this paper")

        # Per-paper consistency check: phase naming, duplicates, conflicts
        scenarios, n_removed = check_paper_consistency(scenarios, paper_start_idx)

    # ── Post-generation validation: duplicates, conflicts, inconsistencies ──
    print(f"\n{'=' * 60}")
    print("VALIDATION 1 — Duplicates, conflicts, and consistency")
    print(f"{'=' * 60}")
    scenarios, n_issues = validate_scenarios(scenarios)

    # Validate structured prompts
    print(f"\n{'=' * 60}")
    print("VALIDATION 2 — Checking structured prompt completeness")
    print(f"{'=' * 60}")

    # Category-specific required headers
    category_headers = {
        "synthesis": ["## Sample Information", "## Synthesis / Preparation",
                      "## Measurement", "## Questions"],
        "electrochemistry": ["## Sample Information", "## Electrochemical Conditions",
                             "## Measurement", "## Questions"],
        "pure_phase": ["## Sample Information", "## Sample Details",
                       "## Measurement", "## Questions"],
        "thin_film": ["## Sample Information", "## Deposition / Film Details",
                      "## Measurement", "## Questions"],
        "catalyst": ["## Sample Information", "## Catalyst Details",
                     "## Measurement", "## Questions"],
        "environmental": ["## Sample Information", "## Sample Origin and Conditions",
                          "## Measurement", "## Questions"],
    }

    forbidden_phrases = ["(typical)", "(typical for", "(inferred)",
                         "solid-state method", "high temperature",
                         "elevated temperature", "standard method",
                         "conventional method", "metal oxide precursors"]

    n_pass, n_warn, n_typical = 0, 0, 0
    for s in scenarios:
        sid = s.get("id", "?")
        cat = s.get("category", "synthesis")
        prompt = s.get("prompt", "")
        issues = []

        headers = category_headers.get(cat, category_headers["synthesis"])
        for hdr in headers:
            if hdr not in prompt:
                issues.append(f"missing header: {hdr}")

        for phrase in forbidden_phrases:
            if phrase.lower() in prompt.lower():
                issues.append(f"forbidden phrase: '{phrase}'")
                if "typical" in phrase.lower():
                    n_typical += 1

        if "ground_truth" not in s:
            issues.append("missing ground_truth")

        if issues:
            n_warn += 1
            print(f"  WARNING {sid} [{cat}]: {'; '.join(issues)}")
        else:
            n_pass += 1

    print(f"\n  Passed: {n_pass}/{len(scenarios)}")
    print(f"  Warnings: {n_warn}/{len(scenarios)}")
    if n_typical:
        print(f"  Scenarios with (typical) values: {n_typical} — these should be 0")

    # Save with timestamp — filename reflects input basename
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    input_base = re.sub(r'_extracted_\d{8}_\d{6}$', '', input_base)
    if input_base in ("extracted_papers", "xlsx_extracted"):
        output_path = os.path.join(BASE_DIR, f"stage1_examples_v3_{timestamp}.json")
    else:
        output_path = os.path.join(BASE_DIR, f"{input_base}_stage1_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(scenarios)} scenarios (v2 structured prompts)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
