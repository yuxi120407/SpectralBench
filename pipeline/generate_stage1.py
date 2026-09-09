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
- **Measurement mode**: [transmission / fluorescence / electron yield]

## Measurement
This is a {element} {edge} XANES measurement of a known pure-phase reference compound.

## Questions
Based on the compound identity and crystal structure above:
1. What is the expected edge position (in eV) for this compound?
2. Describe the expected pre-edge features: how many peaks, their approximate energy positions relative to the edge, and their physical origin (e.g., 1s->3d, 1s->4p transitions).
3. Describe the expected white line (main absorption peak): its intensity relative to the edge jump, shape, and physical origin.
4. What spectral features distinguish this compound from other common {element} phases? (e.g., how does the spectrum differ from other oxidation states or crystal structures?)
5. Are there any other notable XANES features (shoulders, post-edge oscillations, multiple scattering resonances)?""",

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
knowledge.

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
      "edge_position_eV": "expected edge energy in eV, or description if exact value not known",
      "pre_edge": {{
        "description": "detailed description of pre-edge features",
        "n_peaks": "number of pre-edge peaks",
        "origin": "physical origin (e.g., 1s->3d quadrupole, 1s->3d/4p hybridized)",
        "intensity": "weak/moderate/strong and why (symmetry, d-electron count)"
      }},
      "white_line": {{
        "description": "white line shape and intensity",
        "intensity": "relative intensity (weak/moderate/strong) and why",
        "origin": "physical origin of the white line"
      }},
      "distinguishing_features": "what makes this spectrum unique vs other {element} compounds — specific features to look for",
      "other_features": "shoulders, post-edge oscillations, multiple scattering resonances, etc."
    }},
    "key_reasoning": "Why this crystal structure and oxidation state produce these specific spectral features — connect electronic structure to spectral shape"
  }},

  "condition_details": {{
    "compound": "chemical formula",
    "crystal_structure": "crystal system",
    "oxidation_state": "formal oxidation state",
    "coordination": "coordination geometry",
    "sample_form": "powder / foil / pellet / etc.",
    "measurement_mode": "transmission / fluorescence / electron yield"
  }},

  "rubric": {{
    "edge_position": {{
      "max_score": 20,
      "criteria": [
        {{"points": 20, "description": "correct edge position within 1 eV"}},
        {{"points": 10, "description": "correct edge position within 3 eV"}},
        {{"points": 5, "description": "correct relative position (higher/lower than related compounds)"}},
        {{"points": 0, "description": "wrong edge position"}}
      ]
    }},
    "pre_edge_features": {{
      "max_score": 25,
      "criteria": [
        {{"points": number, "description": "correct number of pre-edge peaks"}},
        {{"points": number, "description": "correct physical origin (transition type)"}},
        {{"points": number, "description": "correct intensity assessment and symmetry reasoning"}}
      ]
    }},
    "white_line": {{
      "max_score": 20,
      "criteria": [
        {{"points": number, "description": "correct white line shape description"}},
        {{"points": number, "description": "correct intensity and physical origin"}}
      ]
    }},
    "distinguishing_ability": {{
      "max_score": 20,
      "criteria": [
        {{"points": number, "description": "correctly identifies key differences from related compounds"}},
        {{"points": number, "description": "names specific spectral features that distinguish this compound"}}
      ]
    }},
    "reasoning_quality": {{
      "max_score": 15,
      "criteria": [
        {{"points": number, "description": "correctly connects crystal structure / electronic structure to spectral features"}}
      ]
    }}
  }}
}}

IMPORTANT:
- The ground truth spectral features MUST come from the paper's description of this
  compound's spectrum. Use the extracted condition data fields: edge_position_eV,
  pre_edge_description, white_line_description, spectral_description.
- If the paper does not describe a specific spectral feature, you may supplement with
  well-established textbook XANES knowledge, but mark those as "source": "standard_knowledge"
  in the ground truth.
- Do NOT fabricate spectral features. If a feature is unknown, write null.
- Focus on features that can be verified against actual reference spectra we have.

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


def call_gemini(prompt):
    """Call Gemini via subprocess."""
    try:
        result = subprocess.run(
            [GEMINI_PYTHON, GEMINI_SCRIPT],
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 benchmark scenarios using Gemini"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to fulltext_analyzed_papers*.json (default: latest)")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Max papers to process")
    args = parser.parse_args()

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
        for ci, condition in enumerate(conditions_with_fracs):
            cond_idx += 1
            sample_id = condition.get("sample_id", f"sample_{ci}")
            paper_short = (paper.get("title", "paper") or "paper")[:20].lower()
            paper_short = "".join(c if c.isalnum() else "_" for c in paper_short).strip("_")
            suggested_id = f"{paper_short}_{sample_id}"

            # Per-condition element/edge: parse from sample_id if present
            # e.g., "Combinatorial film, fTi = 0.1 (Ti K-edge)" → Ti, K-edge
            cond_elem = elem
            cond_edge = edge
            import re as _re
            edge_match = _re.search(r'\((\w+)\s+(K|L\d?)-edge\)', sample_id)
            if edge_match:
                cond_elem = edge_match.group(1)
                cond_edge = edge_match.group(2) + "-edge"

            condition_data = json.dumps(condition, indent=2)

            # Select generation prompt based on category
            if category == "pure_phase":
                prompt = GENERATION_PROMPT_PURE_PHASE.format(
                    element=cond_elem,
                    edge=cond_edge,
                    key_finding=a.get("key_finding", ""),
                    condition_data=condition_data,
                    references=AVAILABLE_REFERENCES,
                    suggested_id=suggested_id,
                    source_paper=paper.get("title", "")[:80],
                    benchmark_template=benchmark_template.format(
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
                print(f"    [{cond_idx}/{total_conditions}] {sample_id}: FAILED (Gemini call)")
                continue

            parsed = parse_json(raw)
            if parsed:
                if isinstance(parsed, list):
                    parsed = parsed[0] if parsed else None
                if parsed:
                    parsed["source_paper_full"] = paper_meta
                    parsed["category"] = category
                    parsed["classification"] = classification
                    scenarios.append(parsed)
                    gt = parsed.get("ground_truth", {})
                    if category == "pure_phase":
                        compound = gt.get("compound", "?")
                        print(f"    [{cond_idx}/{total_conditions}] OK: {parsed.get('id', '?')} — {compound} (spectral fingerprint)")
                    else:
                        fracs = gt.get("fractions", {})
                        frac_str = ", ".join(f"{k}={v}" for k, v in fracs.items())
                        print(f"    [{cond_idx}/{total_conditions}] OK: {parsed.get('id', '?')} — {frac_str}")
                    paper_ok += 1
                else:
                    print(f"    [{cond_idx}/{total_conditions}] {sample_id}: FAILED (empty parse)")
            else:
                print(f"    [{cond_idx}/{total_conditions}] {sample_id}: FAILED (JSON parse)")

            import time
            time.sleep(2)

        print(f"    Generated {paper_ok}/{len(conditions_with_fracs)} scenarios from this paper")

    # Validate structured prompts
    print(f"\n{'=' * 60}")
    print("VALIDATION — Checking structured prompt completeness")
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

    # Save with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(BASE_DIR, f"stage1_examples_v3_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(scenarios)} scenarios (v2 structured prompts)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
