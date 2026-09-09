# SpectralBench

A benchmark for evaluating LLM domain knowledge of X-ray absorption near-edge structure (XANES) spectroscopy. Given sample preparation and measurement conditions, LLMs are asked to predict phase compositions or describe spectral features.

## Benchmark overview

- **68 scenarios** from **18 published papers**
- **6 categories**: synthesis, electrochemistry, pure phase, thin film, catalyst, environmental
- **12 elements**: As, C, Cu, Fe, Li, Mn, Ni, O, Pd, Pt, S, Ti
- Each scenario includes a structured prompt (LLM input), ground truth (from the paper), and a scoring rubric

| Category         | Scenarios | Papers | Description |
|------------------|-----------|--------|-------------|
| Thin film        | 20        | 3      | Phase composition from deposition conditions and composition gradients |
| Electrochemistry | 16        | 4      | Phase evolution during charge/discharge, varying C-rate, voltage, SOC |
| Pure phase       | 12        | 5      | Spectral feature description for known reference compounds |
| Catalyst         | 12        | 2      | Phase composition from catalyst preparation and treatment |
| Synthesis        | 5         | 3      | Phase composition from synthesis conditions |
| Environmental    | 3         | 1      | Phase composition from geochemical context |

## Review

Browse all scenarios interactively: open `review.html` in a browser.

Review materials for domain experts are in `review/`:
- `review_checklist.xlsx` -- spreadsheet with all scenarios, color-coded by category
- `review_docs/` -- one Word document per paper with detailed prompts and ground truth

## Data

| File | Description |
|------|-------------|
| `data/scenarios_v3.json` | All 68 benchmark scenarios (prompts, ground truth, rubrics) |
| `data/extracted_papers.json` | Extracted conditions from 18 papers |
| `data/xanes_literature_examples.xlsx` | Source DOI list with paper metadata |

### Scenario structure

Each scenario in `scenarios_v3.json` contains:

```json
{
  "id": "scenario_name",
  "element": "Ti",
  "edge": "K",
  "category": "thin_film",
  "prompt": "## Sample Information\n- **Material**: ...\n\n## Questions\n...",
  "ground_truth": {
    "fractions": {"phase_A": 0.6, "phase_B": 0.4},
    "key_reasoning": "..."
  },
  "rubric": { ... }
}
```

The `prompt` field is the LLM input. The `ground_truth` and `rubric` fields are for scoring.

## Pipeline

The benchmark is generated from published papers through a multi-stage pipeline:

```
DOI list (xlsx) --> Extract (Gemini) --> Deduplicate --> Generate scenarios (Gemini) --> Review
```

### Step 1: Extract data from papers

Reads PDFs via Gemini multimodal. Uses category-specific extraction prompts (electrochemistry gets battery fields, pure phase gets spectral descriptions, etc.). Deduplicates identical conditions and resolves conflicts.

```bash
cd pipeline
python extract_from_xlsx.py
python extract_from_xlsx.py --doi 10.1126/science.aax3520   # single paper
python extract_from_xlsx.py --no-chase                       # skip citation chasing
```

**Requirements**: Google Cloud Vertex AI authentication (ADC), Gemini API access.

The extraction output is saved as `xlsx_extracted_<timestamp>.json`.

### Step 2: Generate benchmark scenarios

Converts extracted conditions into structured benchmark scenarios with prompts, ground truth, and scoring rubrics.

```bash
python generate_stage1.py --input ../data/extracted_papers.json
```

Output: `stage1_examples_v3_<timestamp>.json`

### Step 3: Generate review page

Builds a self-contained HTML review page from the scenarios JSON.

```bash
python generate_review_html.py --input ../data/scenarios_v3.json
```

Output: `review.html`

## Dependencies

```
openpyxl
python-docx
requests
google-cloud-aiplatform
```

## Source papers

The 18 papers in v3 cover XANES studies of batteries, catalysts, thin films, environmental samples, and reference compounds. The full DOI list is in `data/xanes_literature_examples.xlsx`. PDFs are not included in this repo.
