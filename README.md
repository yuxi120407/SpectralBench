# SpectralBench

A benchmark for evaluating LLM domain knowledge of X-ray absorption near-edge structure (XANES) spectroscopy. Given sample preparation and measurement conditions, LLMs are asked to predict phase compositions, identify reference spectra, and describe spectral features.

## Benchmark overview

- **61 scenarios** from **18 published papers**
- **6 categories**: synthesis, electrochemistry, pure phase, thin film, catalyst, environmental
- **14 elements**: As, C, Cu, Fe, Li, Mn, Ni, O, Pd, Pt, S, Ti, V, Zn
- Each scenario includes: structured prompt (LLM input), paper-grounded ground truth, scoring rubric, reference answers
- All ground truth is **paper-only** -- no textbook/prior knowledge added
- Questions are **LLM-generated** from the ground truth, tailored per scenario
- Reference answers are **LLM-generated** from GT + sample conditions, with reasoning for every answer

| Category         | Scenarios | Description |
|------------------|-----------|-------------|
| Thin film        | 20        | Phase composition from deposition conditions and composition gradients |
| Electrochemistry | 13        | Phase evolution during charge/discharge, varying C-rate, voltage, SOC |
| Pure phase       | 12        | Spectral feature description for known reference compounds |
| Catalyst         | 11        | Phase composition from catalyst preparation and treatment |
| Environmental    | 3         | Phase composition from geochemical context |
| Synthesis        | 2         | Phase composition from synthesis conditions |

## Review

Browse all scenarios interactively:
- **Online**: https://yuxi120407.github.io/SpectralBench/review.html
- **Local**: open `review.html` in a browser
- Each scenario has a **Report Issue** button that opens a pre-filled GitHub issue

## Data

| File | Description |
|------|-------------|
| `data/scenarios_v5.json` | Latest benchmark (v5: LLM-generated questions, paper-only GT, verified) |
| `data/scenarios_v4.json` | Previous version (v4: template questions, paper-only GT) |
| `data/xanes_literature_examples.xlsx` | Source DOI list (18 papers) |
| `data/bnl_nsls_2_8_ID.xlsx` | BNL NSLS-II 8-ID beamline papers (339 papers, for scaling) |

### Scenario structure (v5)

```json
{
  "id": "scenario_name",
  "element": "Ti",
  "edge": "K",
  "category": "thin_film",
  "prompt": "## Sample Information\n...\n## Questions\n1. ...\n2. ...",
  "ground_truth": {
    "material": "Combinatorial Ti-Zn oxide thin film at fTi = 0.4",
    "measurement": "Ti K-edge XANES",
    "fit_method": "Cluster blind-signal-separation (cBSS)",
    "fit_basis": ["TiUD", "Ti6L", "Ti6H"],
    "fractions": {"TiUD": 0.45, "Ti6L": 0.45, "Ti6H": 0.1},
    "fractions_uncertainty_pct": 10,
    "source_evidence": "Figure 10",
    "key_reasoning": "At fTi = 0.4, the film enters Region IV...",
    "reasoning_source": "Figure 10 and Section IV B"
  },
  "rubric": {
    "q1": {"question": "...", "type": "identification", "max_score": 30, "criteria": "...", "answer": "The phases are ... because ..."},
    "q2": {"question": "...", "type": "quantification", "max_score": 35, "criteria": "...", "answer": "The fractions are ... because ..."},
    "q3": {"question": "...", "type": "reasoning", "max_score": 35, "criteria": "...", "answer": "Given the conditions ... therefore ..."}
  },
  "source_paper_full": {"title": "...", "doi": "...", "authors": [...]}
}
```

## Pipeline

The benchmark is generated from published papers through a multi-stage pipeline:

```
Stage 0: Curation
    Curate DOI list in xlsx with element/edge/material metadata
    |
    v
Stage 1: Extraction (extract_from_xlsx_v2.py)
    PDF download -> classify -> extract conditions -> resolve cited refs (Semantic Scholar)
    |
    v
Stage 2: Generation (v5/generate_combined.py)
    Paper PDF + conditions -> GT + tailored questions (1 Gemini call per scenario)
    |
    v
Stage 3a: GT Verification (v5/verify_stage3_batch.py)
    Source PDF + scenarios -> verify GT, fix contradictions, remove textbook content
    |
    v
Stage 3b: Question Validation (same script, separate LLM call)
    GT + questions -> remove unanswerable/method/answer-revealing questions
    |
    v
Stage 3c: Answer Generation (pipeline/generate_answers.py)
    GT + sample conditions + questions -> reference answers with reasoning (1 Gemini call per scenario)
    |
    v
Stage 4: Review HTML (pipeline/generate_review_html.py)
    Scenarios -> review.html (GitHub Pages)
```

### Running the pipeline

**Prerequisites**:
- Google Cloud Vertex AI authentication (ADC)
- Gemini API access (project: geminienterpriseprod-485218)
- Python packages: `openpyxl`, `requests`, `google-auth`

**Step 1: Extract data from papers**

```bash
cd /path/to/auto_literature_download

# Default (18-paper set):
python extract_from_xlsx_v2.py

# With specific model:
python extract_from_xlsx_v2.py --model gemini-3.8-flash

# BNL papers:
python extract_from_xlsx_v2.py --xlsx bnl_nsls_2_8_ID.xlsx --model gemini-3.8-flash

# Single paper:
python extract_from_xlsx_v2.py --doi 10.1021/acs.jpcc.3c02029 --model gemini-3.8-flash --no-chase
```

Output: `<xlsx_name>_extracted_<timestamp>.json`

**Step 2: Generate benchmark scenarios (v5)**

```bash
cd v5

# 1-call version (faster -- GT + questions in one call):
python generate_combined.py --input ../xlsx_extracted_merged_20260914.json --model gemini-3.8-flash

# 2-call version (better question alignment -- GT first, then questions):
python generate_stage1.py --input ../xlsx_extracted_merged_20260914.json --model gemini-3.8-flash
```

Output: `<input_base>_v5_<timestamp>.json`

**Step 3: Verify and patch (Stage 3)**

```bash
# GT verification + question validation + auto-patch:
python verify_stage3_batch.py \
  --input <scenarios>.json \
  --extraction ../xlsx_extracted_merged_20260914.json \
  --model gemini-3.1-pro-preview \
  --auto-patch \
  --output xlsx_v5_final.json \
  --report v5_verify_report.json \
  --sleep 3

# Single paper only:
python verify_stage3_batch.py \
  --input xlsx_v5_final.json \
  --doi 10.1103/PhysRevMaterials.9.023802 \
  --model gemini-3.1-pro-preview \
  --auto-patch \
  --output xlsx_v5_final.json \
  --report v5_verify_resolving.json \
  --sleep 3
```

**Step 3c: Generate reference answers**

```bash
# From v5 folder:
python generate_answers.py \
  --input xlsx_v5_final.json \
  --output xlsx_v5_final_with_ans.json \
  --model gemini-3.8-flash

# Or from SpectralBench/pipeline:
cd /path/to/SpectralBench/pipeline
python generate_answers.py \
  --input ../data/scenarios_v5.json \
  --output ../data/scenarios_v5_with_ans.json \
  --model gemini-3.8-flash
```

**Step 4: Generate review HTML**

```bash
# Copy to SpectralBench data:
cp v5/xlsx_v5_final_with_ans.json /path/to/SpectralBench/data/scenarios_v5_with_ans.json

# Generate HTML:
cd /path/to/SpectralBench/pipeline
python generate_review_html.py --input ../data/scenarios_v5_with_ans.json
cp review.html ../review.html
```

**Step 5: Push to GitHub**

```bash
cd /path/to/SpectralBench
git add data/scenarios_v5.json review.html pipeline/ README.md
git commit -m "Update benchmark scenarios"
git push
```

## Pipeline versions

| Version | Schema | Questions | Status |
|---------|--------|-----------|--------|
| v3 | candidate_phases, recommended_references, key_reasoning | Template per category | Deprecated |
| v4 | fit_basis, source_evidence, fractions_uncertainty_pct | Template per category | Production (backup) |
| v5 | Same as v4 | LLM-generated from GT | Current |

## Key design decisions

- **Paper-only GT**: every value must come from the source paper's text, figures, or tables. No textbook knowledge.
- **Semantic Scholar DOI verification**: catches Gemini's DOI hallucinations by looking up citations by title/author.
- **Multi-PDF fusion**: reads source paper + cited reference PDFs together for complete context.
- **Stage 3 verification**: cross-checks GT against source PDF; catches textbook injection via ADDED_NOT_IN_PAPER status.
- **Question validation**: separate LLM call to verify each question is answerable from GT.
- **Fractions safety**: auto-patch never reduces number of phases (prevents data loss).

## Source papers

The 18 papers in the current benchmark cover XANES studies of batteries, catalysts, thin films, environmental samples, and reference compounds. The full DOI list is in `data/xanes_literature_examples.xlsx`. PDFs are not included in this repo.

For scaling: 339 additional papers from NSLS-II beamline 8-ID are available in `data/bnl_nsls_2_8_ID.xlsx` (210 successfully extracted, 711 conditions).
