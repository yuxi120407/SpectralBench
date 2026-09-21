"""
Generate a self-contained HTML review page from v5 scenarios JSON,
with embedded PDF screenshots of referenced figures and tables.

Usage:
    python generate_review_html_v2.py --input ../data/scenarios_v5_with_ans.json
    python generate_review_html_v2.py --input ../data/scenarios_v5_with_ans.json --papers-dir /path/to/papers
"""

import json
import glob
import os
import re
import html
import base64
import argparse
from collections import OrderedDict

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PAPERS_DIR = os.path.join(
    os.path.expanduser("~"),
    "workspace/Storage/xyu1/persistent/Nanospectroscopy/python_version/"
    "benchmark/auto_literature_download/papers"
)

DPI = 200

CAT_COLORS = {
    "synthesis": "#e2efda",
    "electrochemistry": "#d6e4f0",
    "pure_phase": "#fff2cc",
    "thin_film": "#f2dcdb",
    "catalyst": "#e4dfec",
    "environmental": "#d9ead3",
}

CAT_LABELS = {
    "synthesis": "Synthesis",
    "electrochemistry": "Electrochemistry",
    "pure_phase": "Pure Phase",
    "thin_film": "Thin Film",
    "catalyst": "Catalyst",
    "environmental": "Environmental",
}


# ---------------------------------------------------------------------------
# PDF figure/table extraction
# ---------------------------------------------------------------------------

def doi_to_filename(doi):
    return doi.replace("/", "_").replace(":", "_") + ".pdf"


def parse_evidence_refs(source_evidence, reasoning_source=""):
    """Extract figure, table, and page references from source_evidence and reasoning_source."""
    refs = {"figures": set(), "tables": set(), "pages": set()}
    combined = f"{source_evidence} {reasoning_source}"

    # Figures: "Figure 10b", "Figure 2E", "Fig. 3", "FIG. 10", "Figure S8", "fig. S6"
    for m in re.finditer(r'(?:Fig(?:ure|\.)\s*)(S?\d+[a-zA-Z]?)', combined, re.IGNORECASE):
        refs["figures"].add(m.group(1).upper().rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ").rstrip("abcdefghijklmnopqrstuvwxyz") if False else m.group(1))

    # Tables: "Table II", "Table S2", "Table 1", "TABLE II"
    for m in re.finditer(r'Table\s+(S?\d+|[IVX]+)', combined, re.IGNORECASE):
        refs["tables"].add(m.group(1))

    # Pages: "Page 8", "page 21", "Pages 13, 14"
    for m in re.finditer(r'[Pp]age[s]?\s+([\d,\s\-and]+)', combined):
        for num in re.findall(r'\d+', m.group(1)):
            refs["pages"].add(int(num))

    return refs


def _normalize_fig_label(label):
    """Normalize a figure label for matching: strip sub-panel letters."""
    label = label.strip().upper()
    # "10B" -> "10", "S8" -> "S8"
    m = re.match(r'^(S?\d+)', label)
    return m.group(1) if m else label


def _normalize_table_label(label):
    """Normalize table label: roman -> arabic or keep as-is."""
    label = label.strip().upper()
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
                 "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    if label in roman_map:
        return str(roman_map[label])
    return label


def find_evidence_pages(pdf_path, refs):
    """Find PDF pages that contain the referenced figures/tables.

    Returns a dict of {label: [page_indices]} where label is e.g. "Figure 10" or "Table II".
    Prefers pages with captions (actual figure location) over mere text references.
    """
    if not fitz or not os.path.exists(pdf_path):
        return {}

    doc = fitz.open(pdf_path)
    results = {}

    for fig_label in refs["figures"]:
        norm = _normalize_fig_label(fig_label)
        key = f"Figure {fig_label}"
        caption_pages = []
        mention_pages = []

        if norm.startswith("S"):
            num = norm[1:]
            caption_pats = [
                re.compile(rf'Fig(?:ure|\.)\s*S\s*{num}\s*[.:|\s]', re.IGNORECASE),
            ]
            mention_pats = [
                re.compile(rf'Fig(?:ure|\.)\s*S\s*{num}\b', re.IGNORECASE),
            ]
        else:
            # Caption: "FIG. 10." / "Figure 5:" / "Fig. 3 |" / "Fig. 3." (case-insensitive)
            caption_pats = [
                re.compile(rf'FIG\.\s*{norm}\s*[.:|]'),
                re.compile(rf'Fig(?:ure|\.)\s*{norm}\s*[.:|]', re.IGNORECASE),
            ]
            mention_pats = [
                re.compile(rf'Fig(?:ure|\.)\s*{norm}\b', re.IGNORECASE),
            ]

        for i, page in enumerate(doc):
            text = page.get_text()
            if any(p.search(text) for p in caption_pats):
                caption_pages.append(i)
            elif any(p.search(text) for p in mention_pats):
                mention_pages.append(i)

        # Prefer caption pages (where the actual figure is); fall back to first mention
        pages = caption_pages[:2] if caption_pages else mention_pages[:1]
        if pages:
            results[key] = pages

    for tbl_label in refs["tables"]:
        norm = _normalize_table_label(tbl_label)
        key = f"Table {tbl_label}"
        caption_pages = []
        mention_pages = []

        roman_map_rev = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
                         6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}
        try:
            arabic = int(norm)
            roman = roman_map_rev.get(arabic, "")
        except ValueError:
            arabic = None
            roman = ""

        caption_pats = []
        mention_pats = []
        if tbl_label.startswith("S") or tbl_label.startswith("s"):
            caption_pats.append(re.compile(rf'Table\s+S\s*{tbl_label[1:]}\s*[.:]', re.IGNORECASE))
            caption_pats.append(re.compile(rf'Table\s+S\s*{tbl_label[1:]}\s\s', re.IGNORECASE))
            mention_pats.append(re.compile(rf'Table\s+S\s*{tbl_label[1:]}\b', re.IGNORECASE))
        else:
            if arabic is not None:
                caption_pats.append(re.compile(rf'TABLE\s+{norm}\s*[.:]'))
                caption_pats.append(re.compile(rf'Table\s+{norm}\s*[.:]'))
                caption_pats.append(re.compile(rf'Table\s+{norm}\s\s'))
                mention_pats.append(re.compile(rf'TABLE\s+{norm}\b'))
                mention_pats.append(re.compile(rf'Table\s+{norm}\b'))
            if roman:
                caption_pats.append(re.compile(rf'TABLE\s+{roman}\s*[.:]'))
                caption_pats.append(re.compile(rf'Table\s+{roman}\s*[.:]'))
                mention_pats.append(re.compile(rf'TABLE\s+{roman}\b'))
                mention_pats.append(re.compile(rf'Table\s+{roman}\b'))

        for i, page in enumerate(doc):
            text = page.get_text()
            if any(p.search(text) for p in caption_pats):
                caption_pages.append(i)
            elif any(p.search(text) for p in mention_pats):
                mention_pages.append(i)

        pages = caption_pages[:2] if caption_pages else mention_pages[:1]
        if pages:
            results[key] = pages

    # Direct page references (0-indexed)
    for pg in refs["pages"]:
        idx = pg - 1  # PDF pages are 0-indexed
        if 0 <= idx < len(doc):
            key = f"Page {pg}"
            if key not in results:
                results[key] = [idx]

    doc.close()
    return results


_evidence_dir = None  # set by build_html
_page_cache = {}      # tracks rendered pages: (pdf_path, page_idx) -> relative img path


def render_page_to_file(pdf_path, page_idx, dpi=DPI):
    """Render a PDF page to a PNG file in the evidence directory.
    Returns the relative path (e.g. 'evidence/10.1103_..._p22.png')."""
    cache_key = (pdf_path, page_idx)
    if cache_key in _page_cache:
        return _page_cache[cache_key]

    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        doc.close()
        return None
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    filename = f"{pdf_stem}_p{page_idx + 1}.png"
    out_path = os.path.join(_evidence_dir, filename)
    pix.save(out_path)
    doc.close()

    rel_path = f"evidence_original/{filename}"
    _page_cache[cache_key] = rel_path
    return rel_path


def build_evidence_images_html(pdf_path, source_evidence, reasoning_source=""):
    """Build HTML with embedded page images for a scenario's evidence references.

    Only includes figure and table pages (not raw page references) to keep output focused.
    """
    if not fitz or not pdf_path or not os.path.exists(pdf_path):
        if not os.path.exists(pdf_path or ""):
            return '<div class="evidence-note">PDF not available</div>'
        return '<div class="evidence-note">pymupdf not installed — no screenshots</div>'

    refs = parse_evidence_refs(source_evidence, reasoning_source)
    if not refs["figures"] and not refs["tables"]:
        return '<div class="evidence-note">No figure/table references found in source evidence</div>'

    evidence_pages = find_evidence_pages(pdf_path, refs)
    if not evidence_pages:
        return '<div class="evidence-note">Referenced figures/tables not found in PDF text</div>'

    # Deduplicate pages across labels, but keep label association
    seen_pages = set()
    parts = []
    for label, pages in sorted(evidence_pages.items()):
        # Skip raw page references — only show figures and tables
        if label.startswith("Page "):
            continue
        for pidx in pages:
            if pidx in seen_pages:
                continue
            seen_pages.add(pidx)
            img_path = render_page_to_file(pdf_path, pidx)
            if img_path:
                parts.append(
                    f'<div class="evidence-img-wrapper">'
                    f'<div class="evidence-img-label">{html.escape(label)} (PDF page {pidx + 1})</div>'
                    f'<div class="evidence-img-scroll">'
                    f'<img class="evidence-img" src="{img_path}" '
                    f'alt="{html.escape(label)}" loading="lazy"/>'
                    f'</div>'
                    f'</div>'
                )

    if not parts:
        return '<div class="evidence-note">Referenced figures/tables not located in PDF</div>'

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML formatting helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def clean_title(title):
    """Clean paper titles: strip any residual XML/MathML tags, fix whitespace."""
    if not title or len(title.strip()) < 5:
        return ""
    cleaned = re.sub(r'<[^>]+>', '', title)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def extract_element_edge(scenario):
    prompt = scenario.get("prompt", "")
    for line in prompt.split("\n"):
        if "measure the" in line.lower():
            m = re.search(r'measure the (\w+)\s+([\w-]+)\s+XANES', line)
            if m:
                return m.group(1), m.group(2)
    return scenario.get("element", "?"), scenario.get("edge", "?")


def extract_material(prompt):
    for line in prompt.split("\n"):
        if "**Material**" in line:
            return line.split(":", 1)[-1].strip().strip("*")
    return ""


def format_fractions(gt):
    fracs = gt.get("fractions", {})
    if not fracs:
        return "(pure phase — spectral features)"
    return ", ".join(f"{k}: {v}" for k, v in fracs.items())


def format_ground_truth_html(gt, category):
    parts = []

    if category == "pure_phase":
        parts.append(f'<div class="gt-field"><span class="gt-key">Compound:</span> {html.escape(str(gt.get("compound", "?")))}</div>')
        parts.append(f'<div class="gt-field"><span class="gt-key">Oxidation state:</span> {html.escape(str(gt.get("oxidation_state", "?")))}</div>')
        cs = gt.get("crystal_structure")
        if cs:
            parts.append(f'<div class="gt-field"><span class="gt-key">Crystal structure:</span> {html.escape(str(cs))}</div>')
        coord = gt.get("coordination")
        if coord:
            parts.append(f'<div class="gt-field"><span class="gt-key">Coordination:</span> {html.escape(str(coord))}</div>')
        meas = gt.get("measurement")
        if meas:
            parts.append(f'<div class="gt-field"><span class="gt-key">Measurement:</span> {html.escape(str(meas))}</div>')

        sf = gt.get("spectral_features", {})
        if sf:
            parts.append(f'<div class="gt-field"><span class="gt-key">Edge position:</span> {html.escape(str(sf.get("edge_position_eV", "?")))}</div>')
            shape = sf.get("spectral_shape_summary", "")
            if shape:
                parts.append(f'<div class="gt-field"><span class="gt-key">Spectral shape:</span> {html.escape(shape)}</div>')
            peaks = sf.get("labeled_peaks", [])
            if peaks:
                parts.append('<div class="gt-field"><span class="gt-key">Labeled peaks:</span></div>')
                parts.append('<table class="gt-table"><tr><th>Label</th><th>Energy (eV)</th><th>Intensity</th><th>Origin</th><th>Source</th></tr>')
                for pk in peaks:
                    parts.append(
                        f'<tr><td>{html.escape(str(pk.get("label", "")))}</td>'
                        f'<td>{html.escape(str(pk.get("energy_eV", "")))}</td>'
                        f'<td>{html.escape(str(pk.get("intensity", "")))}</td>'
                        f'<td>{html.escape(str(pk.get("origin", "")))}</td>'
                        f'<td>{html.escape(str(pk.get("source", "")))}</td></tr>'
                    )
                parts.append('</table>')

            pre = sf.get("pre_edge", {})
            if isinstance(pre, dict) and pre.get("description"):
                parts.append(f'<div class="gt-field"><span class="gt-key">Pre-edge:</span> {html.escape(pre["description"])}</div>')
            wl = sf.get("white_line", {})
            if isinstance(wl, dict) and wl.get("description"):
                parts.append(f'<div class="gt-field"><span class="gt-key">White line:</span> {html.escape(wl["description"])}</div>')

            sens = sf.get("structural_sensitivity", "")
            if sens:
                parts.append(f'<div class="gt-field"><span class="gt-key">Structural sensitivity:</span> {html.escape(sens)}</div>')
            dist = sf.get("distinguishing_features", "")
            if dist:
                parts.append(f'<div class="gt-field"><span class="gt-key">Distinguishing features:</span> {html.escape(dist)}</div>')
    else:
        material = gt.get("material", "")
        if material:
            parts.append(f'<div class="gt-field"><span class="gt-key">Material:</span> {html.escape(material)}</div>')
        measurement = gt.get("measurement", "")
        if measurement:
            parts.append(f'<div class="gt-field"><span class="gt-key">Measurement:</span> {html.escape(measurement)}</div>')
        fit_method = gt.get("fit_method", "")
        if fit_method:
            parts.append(f'<div class="gt-field"><span class="gt-key">Fit method:</span> {html.escape(fit_method)}</div>')

        fracs = gt.get("fractions", {})
        if fracs:
            unc = gt.get("fractions_uncertainty_pct")
            unc_label = f' <span style="color:#666;font-size:0.9em">(±{unc}%)</span>' if unc else ''
            parts.append(f'<div class="gt-field"><span class="gt-key">Phase fractions:{unc_label}</span></div>')
            parts.append('<table class="gt-table"><tr><th>Phase</th><th>Fraction</th></tr>')
            for phase, frac in fracs.items():
                parts.append(f'<tr><td>{html.escape(phase)}</td><td>{frac}</td></tr>')
            parts.append('</table>')

        fit_basis = gt.get("fit_basis", [])
        if fit_basis:
            parts.append('<div class="gt-field"><span class="gt-key">Fit basis (paper\'s reference spectra):</span></div>')
            parts.append('<ul style="margin:4px 0 4px 20px;padding:0">')
            for r in fit_basis:
                parts.append(f'<li>{html.escape(str(r))}</li>')
            parts.append('</ul>')
        else:
            candidates = gt.get("candidate_phases", [])
            if candidates:
                parts.append(f'<div class="gt-field"><span class="gt-key">Candidate phases:</span> {html.escape(", ".join(candidates))}</div>')
            refs = gt.get("recommended_references", [])
            if refs:
                parts.append(f'<div class="gt-field"><span class="gt-key">Recommended references:</span> {html.escape(", ".join(refs))}</div>')

        se = gt.get("source_evidence", "")
        if se:
            parts.append(f'<div class="gt-field"><span class="gt-key">Source evidence:</span> {html.escape(se)}</div>')

    reasoning = gt.get("key_reasoning", "")
    if reasoning:
        parts.append(f'<div class="gt-field"><span class="gt-key">Key reasoning:</span> {html.escape(reasoning)}</div>')
    rs = gt.get("reasoning_source", "")
    if rs:
        parts.append(f'<div class="gt-field"><span class="gt-key">Reasoning source:</span> {html.escape(rs)}</div>')

    return "\n".join(parts)


def format_rubric_html(rubric):
    if not rubric:
        return ""

    has_answers = any(cat_data.get("answer") for cat_data in rubric.values())
    header = '<tr><th>Q</th><th>Type</th><th>Max</th><th>Question</th>'
    if has_answers:
        header += '<th>Reference Answer</th>'
    header += '<th>Scoring Criteria</th></tr>'
    parts = [f'<table class="rubric-table">{header}']

    for cat_name, cat_data in rubric.items():
        max_score = cat_data.get("max_score", "?")
        criteria = cat_data.get("criteria", [])

        if isinstance(criteria, str):
            criteria_html = html.escape(criteria)
        elif isinstance(criteria, list):
            criteria_lines = []
            for c in criteria:
                if isinstance(c, dict):
                    pts = c.get("points", "?")
                    desc = c.get("description", "")
                    criteria_lines.append(f'{pts} pts — {html.escape(desc)}')
                elif isinstance(c, str):
                    criteria_lines.append(html.escape(c))
            criteria_html = "<br>".join(criteria_lines)
        else:
            criteria_html = str(criteria)

        question = cat_data.get("question", "")
        qtype = cat_data.get("type", "")
        answer = cat_data.get("answer", "")

        row = f'<tr><td class="rubric-cat">{html.escape(cat_name)}</td>'
        row += f'<td class="rubric-type">{html.escape(qtype)}</td>'
        row += f'<td class="rubric-max">{max_score}</td>'
        row += f'<td class="rubric-question">{html.escape(question)}</td>'
        if has_answers:
            row += f'<td class="rubric-answer">{html.escape(answer)}</td>'
        row += f'<td class="rubric-criteria">{criteria_html}</td></tr>'
        parts.append(row)

    parts.append('</table>')
    return "\n".join(parts)


def prompt_to_html(prompt):
    lines = prompt.split("\n")
    out = []
    for line in lines:
        line_esc = html.escape(line)
        if line.startswith("## "):
            out.append(f'<h4 class="prompt-h">{line_esc[3:]}</h4>')
        elif line.startswith("- **"):
            m = re.match(r'- \*\*(.+?)\*\*:\s*(.*)', line)
            if m:
                out.append(f'<div class="prompt-field"><span class="prompt-key">{html.escape(m.group(1))}:</span> {html.escape(m.group(2))}</div>')
            else:
                out.append(f'<div class="prompt-line">{line_esc}</div>')
        elif line.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
            out.append(f'<div class="prompt-q">{line_esc}</div>')
        elif line.strip():
            out.append(f'<div class="prompt-line">{line_esc}</div>')
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main HTML builder
# ---------------------------------------------------------------------------

def build_html(scenarios, papers_dir, output_dir):
    global _evidence_dir, _page_cache
    _evidence_dir = os.path.join(output_dir, "evidence_original")
    os.makedirs(_evidence_dir, exist_ok=True)
    _page_cache = {}

    by_doi_unsorted = {}
    for s in scenarios:
        doi = s.get("source_paper_full", {}).get("doi", "?")
        if doi not in by_doi_unsorted:
            by_doi_unsorted[doi] = {
                "meta": s.get("source_paper_full", {}),
                "category": s.get("category", "?"),
                "scenarios": [],
            }
        by_doi_unsorted[doi]["scenarios"].append(s)

    # Sort: Original-17 papers first, then others
    by_doi = OrderedDict(
        sorted(by_doi_unsorted.items(),
               key=lambda x: (0 if x[1]["meta"].get("facility") == "Original-17" else 1, x[0]))
    )

    cat_counts = {}
    for s in scenarios:
        c = s.get("category", "?")
        cat_counts[c] = cat_counts.get(c, 0) + 1

    facility_counts = {}
    for s in scenarios:
        f = s.get("source_paper_full", {}).get("facility", "")
        if f:
            facility_counts[f] = facility_counts.get(f, 0) + 1

    filter_buttons = []
    filter_buttons.append(f'<button class="filter-btn active" data-cat="all">All ({len(scenarios)})</button>')
    for cat in ["synthesis", "electrochemistry", "pure_phase", "thin_film", "catalyst", "environmental"]:
        if cat in cat_counts:
            label = CAT_LABELS.get(cat, cat)
            color = CAT_COLORS.get(cat, "#eee")
            filter_buttons.append(
                f'<button class="filter-btn" data-cat="{cat}" style="background:{color}">'
                f'{label} ({cat_counts[cat]})</button>'
            )

    facility_buttons = []
    if facility_counts:
        facility_buttons.append('<button class="filter-btn facility-filter active" data-facility="all">All Facilities</button>')
        for fac, count in sorted(facility_counts.items()):
            facility_buttons.append(
                f'<button class="filter-btn facility-filter" data-facility="{html.escape(fac)}" '
                f'style="background:#bee3f8">{html.escape(fac)} ({count})</button>'
            )

    papers_html = []
    scenario_idx = 0
    total_images = 0

    for doi, paper in by_doi.items():
        meta = paper["meta"]
        cat = paper["category"]
        title = meta.get("title", "?")
        authors = meta.get("authors", [])
        author_str = authors[0] + " et al." if len(authors) > 3 else ", ".join(authors)
        color = CAT_COLORS.get(cat, "#f5f5f5")
        cat_label = CAT_LABELS.get(cat, cat)
        n = len(paper["scenarios"])

        pdf_path = os.path.join(papers_dir, doi_to_filename(doi))
        is_original = (meta.get("facility") == "Original-17")
        pdf_exists = is_original and os.path.exists(pdf_path)

        # --- Paper-level evidence: only for Original-17 papers with PDFs ---
        paper_evidence_html = ""
        if pdf_exists and fitz:
            all_refs = {"figures": set(), "tables": set(), "pages": set()}
            for s in paper["scenarios"]:
                gt = s.get("ground_truth", {})
                sr = parse_evidence_refs(
                    gt.get("source_evidence", ""),
                    gt.get("reasoning_source", ""),
                )
                all_refs["figures"] |= sr["figures"]
                all_refs["tables"] |= sr["tables"]
                all_refs["pages"] |= sr["pages"]

            has_any_refs = all_refs["figures"] or all_refs["tables"] or all_refs["pages"]

            if has_any_refs:
                evidence_pages = find_evidence_pages(pdf_path, all_refs)
                seen_pages = set()
                img_parts = []
                # Show figures and tables first, then page refs
                for label, pages in sorted(evidence_pages.items(),
                                           key=lambda x: (x[0].startswith("Page "), x[0])):
                    for pidx in pages:
                        if pidx in seen_pages:
                            continue
                        seen_pages.add(pidx)
                        img_path = render_page_to_file(pdf_path, pidx)
                        if img_path:
                            img_parts.append(
                                f'<div class="evidence-img-wrapper">'
                                f'<div class="evidence-img-label">{html.escape(label)} (PDF page {pidx + 1})</div>'
                                f'<div class="evidence-img-scroll">'
                                f'<img class="evidence-img" src="{img_path}" '
                                f'alt="{html.escape(label)}" loading="lazy"/>'
                                f'</div></div>'
                            )
            else:
                # No refs at all — show first page of PDF as fallback
                img_parts = []
                img_path = render_page_to_file(pdf_path, 0)
                if img_path:
                    img_parts.append(
                        f'<div class="evidence-img-wrapper">'
                        f'<div class="evidence-img-label">Title page (PDF page 1)</div>'
                        f'<div class="evidence-img-scroll">'
                        f'<img class="evidence-img" src="{img_path}" '
                        f'alt="Title page" loading="lazy"/>'
                        f'</div></div>'
                    )

            if img_parts:
                ref_parts = sorted(all_refs["figures"]) + sorted(all_refs["tables"])
                if all_refs["pages"]:
                    ref_parts += [f"p.{p}" for p in sorted(all_refs["pages"])]
                ref_summary = ", ".join(ref_parts) if ref_parts else "Title page"
                paper_evidence_html = (
                    f'<div class="paper-evidence">'
                    f'<div class="paper-evidence-header" onclick="togglePaperEvidence(this)">'
                    f'<span class="section-title">Key Paper Evidence</span>'
                    f'<span class="paper-evidence-refs">{html.escape(ref_summary)}</span>'
                    f'<span class="toggle-icon">&#9654;</span>'
                    f'</div>'
                    f'<div class="paper-evidence-body" style="display:none">'
                    f'<div class="evidence-box evidence-grid">{"".join(img_parts)}</div>'
                    f'</div></div>'
                )
        elif not pdf_exists:
            paper_evidence_html = ""

        conditions_html = []
        for s in paper["scenarios"]:
            scenario_idx += 1
            gt = s.get("ground_truth", {})
            cd = s.get("condition_details", {})
            elem, edge = extract_element_edge(s)
            material = extract_material(s.get("prompt", ""))
            fracs = format_fractions(gt)
            kv = cd.get("key_variable", "")
            kvv = cd.get("key_variable_value", "")
            kv_str = f"{kv} = {kvv}" if kv else ""
            prompt_html = prompt_to_html(s.get("prompt", ""))
            gt_html = format_ground_truth_html(gt, s.get("category", ""))
            rubric_html = format_rubric_html(s.get("rubric", {}))

            conditions_html.append(f"""
            <div class="condition">
                <div class="condition-header" onclick="toggleCondition(this)">
                    <span class="condition-num">#{scenario_idx}</span>
                    <span class="condition-material">{html.escape(material or s.get('id', '?'))}</span>
                    <span class="condition-elem">{html.escape(elem)} {html.escape(edge)}</span>
                    <span class="condition-kv">{html.escape(kv_str)}</span>
                    <span class="toggle-icon">&#9654;</span>
                </div>
                <div class="condition-body" style="display:none">
                    <div class="section">
                        <div class="section-title">LLM Prompt</div>
                        <div class="prompt-box">{prompt_html}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">Ground Truth</div>
                        <div class="gt-box">{gt_html}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">Scoring Rubric (total: {sum(v.get('max_score', 0) for v in s.get('rubric', dict()).values())} pts)</div>
                        <div class="rubric-box">{rubric_html}</div>
                    </div>
                    <div class="section review-section">
                        <div class="section-title">Review</div>
                        <div class="review-controls">
                            <button class="review-btn correct" onclick="markReview(this, 'correct')">&#10003; Correct</button>
                            <button class="review-btn incorrect" onclick="markReview(this, 'incorrect')">&#10007; Incorrect</button>
                            <input type="text" class="review-comment" placeholder="Comments..." />
                            <button class="review-btn issue" onclick="reportIssue(this, {scenario_idx}, '{html.escape(s.get("id",""))}', '{html.escape(elem)}', '{html.escape(cat)}', '{html.escape(doi)}')">&#9888; Report Issue</button>
                        </div>
                    </div>
                </div>
            </div>
            """)

        facility = meta.get("facility", "")
        facility_badge = f'<span class="facility-badge">{html.escape(facility)}</span>' if facility else ''
        display_title = clean_title(title) or f"DOI: {doi}"

        papers_html.append(f"""
        <div class="paper" data-cat="{cat}" data-facility="{html.escape(facility)}">
            <div class="paper-header" style="border-left: 5px solid {color};" onclick="togglePaper(this)">
                <div class="paper-title">{html.escape(display_title)}</div>
                <div class="paper-meta">
                    {facility_badge}
                    <span class="cat-badge" style="background:{color}">{cat_label}</span>
                    <span class="paper-authors">{html.escape(author_str)}</span>
                    <span class="paper-doi">DOI: {html.escape(doi)}</span>
                    <span class="paper-count">{n} scenario{'s' if n > 1 else ''}</span>
                    {'<span class="pdf-badge">PDF</span>' if pdf_exists else ''}
                </div>
                <span class="toggle-icon">&#9654;</span>
            </div>
            <div class="paper-body" style="display:none">
                {paper_evidence_html}
                {''.join(conditions_html)}
            </div>
        </div>
        """)

    total_images = len(_page_cache)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpectralBench v5 — Review (with PDF evidence)</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; color: #333; line-height: 1.5; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 5px; }}
.subtitle {{ color: #666; margin-bottom: 20px; }}
.stats {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
.stat {{ background: white; border-radius: 8px; padding: 12px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stat-num {{ font-size: 1.6em; font-weight: 700; color: #2c5282; }}
.stat-label {{ font-size: 0.85em; color: #666; }}
.filters {{ margin-bottom: 20px; display: flex; gap: 8px; flex-wrap: wrap; }}
.filter-btn {{ border: 1px solid #ddd; border-radius: 20px; padding: 6px 16px; cursor: pointer; font-size: 0.85em; transition: all 0.2s; }}
.filter-btn:hover {{ border-color: #999; }}
.filter-btn.active {{ border-color: #333; font-weight: 600; }}
.paper {{ background: white; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
.paper.hidden {{ display: none; }}
.paper-header {{ padding: 16px 20px; cursor: pointer; position: relative; }}
.paper-header:hover {{ background: #fafafa; }}
.paper-title {{ font-size: 1.05em; font-weight: 600; padding-right: 30px; }}
.paper-meta {{ margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; font-size: 0.85em; color: #666; }}
.cat-badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 500; }}
.pdf-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; background: #c6f6d5; color: #276749; }}
.pdf-badge.missing {{ background: #fed7d7; color: #9b2c2c; }}
.facility-badge {{ padding: 2px 10px; border-radius: 12px; font-size: 0.75em; font-weight: 600; background: #bee3f8; color: #2a4365; }}
.paper-count {{ color: #2c5282; font-weight: 500; }}
.toggle-icon {{ position: absolute; right: 16px; top: 18px; font-size: 0.8em; color: #999; transition: transform 0.2s; }}
.toggle-icon.open {{ transform: rotate(90deg); }}
.paper-body {{ padding: 0 20px 16px; }}
.condition {{ border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 10px; }}
.condition-header {{ padding: 10px 14px; cursor: pointer; display: flex; align-items: center; gap: 12px; position: relative; }}
.condition-header:hover {{ background: #f7fafc; }}
.condition-num {{ font-weight: 700; color: #2c5282; min-width: 30px; }}
.condition-material {{ font-weight: 500; flex: 1; }}
.condition-elem {{ background: #edf2f7; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
.condition-kv {{ color: #666; font-size: 0.85em; }}
.condition-body {{ padding: 12px 14px; border-top: 1px solid #e2e8f0; }}
.section {{ margin-bottom: 14px; }}
.section-title {{ font-weight: 600; font-size: 0.85em; color: #4a5568; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.gt-box {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; font-size: 0.85em; }}
.gt-field {{ margin: 4px 0; }}
.gt-key {{ font-weight: 600; color: #2d3748; }}
.gt-table {{ border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 0.9em; }}
.gt-table th {{ background: #edf2f7; padding: 6px 10px; text-align: left; font-weight: 600; font-size: 0.85em; border: 1px solid #e2e8f0; }}
.gt-table td {{ padding: 5px 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
.paper-evidence {{ border: 1px solid #fbd38d; border-radius: 6px; margin: 10px 0 16px; background: #fffaf0; }}
.paper-evidence-header {{ padding: 10px 14px; cursor: pointer; display: flex; align-items: center; gap: 12px; position: relative; }}
.paper-evidence-header:hover {{ background: #fff8e1; }}
.paper-evidence-header .section-title {{ margin: 0; }}
.paper-evidence-refs {{ font-size: 0.8em; color: #975a16; flex: 1; }}
.paper-evidence-body {{ padding: 0 14px 14px; }}
.evidence-box {{ background: #fffaf0; border: 1px solid #fbd38d; border-radius: 4px; padding: 12px; }}
.evidence-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 12px; }}
.evidence-note {{ font-size: 0.85em; color: #975a16; font-style: italic; }}
.evidence-img-wrapper {{ margin: 8px 0; }}
.evidence-img-label {{ font-size: 0.8em; font-weight: 600; color: #744210; margin-bottom: 4px; padding: 3px 8px; background: #fefcbf; border-radius: 4px; display: inline-block; }}
.evidence-img-scroll {{ max-height: 400px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 4px; background: #fff; }}
.evidence-img {{ width: 100%; display: block; cursor: pointer; }}
.evidence-img.zoomed {{ position: fixed; top: 5%; left: 5%; width: 90%; height: 90%; object-fit: contain; z-index: 1000; background: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 8px; cursor: zoom-out; }}
.evidence-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 999; cursor: zoom-out; }}
.rubric-box {{ font-size: 0.85em; }}
.rubric-table {{ border-collapse: collapse; width: 100%; margin: 4px 0; }}
.rubric-table th {{ background: #edf2f7; padding: 6px 10px; text-align: left; font-weight: 600; font-size: 0.85em; border: 1px solid #e2e8f0; }}
.rubric-table td {{ padding: 5px 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
.rubric-cat {{ font-weight: 500; white-space: nowrap; }}
.rubric-type {{ font-size: 0.85em; color: #666; white-space: nowrap; }}
.rubric-max {{ text-align: center; font-weight: 600; color: #2c5282; }}
.rubric-question {{ font-size: 0.9em; }}
.rubric-answer {{ font-size: 0.9em; color: #276749; background: #f0fff4; }}
.rubric-criteria {{ font-size: 0.9em; color: #4a5568; }}
.prompt-box {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; font-size: 0.85em; }}
.prompt-h {{ font-weight: 600; margin: 8px 0 4px; color: #2d3748; }}
.prompt-field {{ margin: 2px 0; }}
.prompt-key {{ font-weight: 600; }}
.prompt-q {{ margin: 2px 0; color: #2c5282; }}
.prompt-line {{ margin: 2px 0; }}
.review-section {{ background: #fffbeb; padding: 12px; border-radius: 6px; }}
.review-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.review-btn {{ border: 1px solid #ddd; border-radius: 4px; padding: 6px 14px; cursor: pointer; font-size: 0.85em; }}
.review-btn.correct:hover, .review-btn.correct.selected {{ background: #c6f6d5; border-color: #38a169; }}
.review-btn.incorrect:hover, .review-btn.incorrect.selected {{ background: #fed7d7; border-color: #e53e3e; }}
.review-btn.issue {{ background: #fff; color: #b7791f; border-color: #ecc94b; }}
.review-btn.issue:hover {{ background: #fefcbf; border-color: #d69e2e; }}
.review-comment {{ flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 4px; padding: 6px 10px; font-size: 0.85em; }}
.condition.reviewed-correct {{ border-left: 4px solid #38a169; }}
.condition.reviewed-incorrect {{ border-left: 4px solid #e53e3e; }}
.expand-all {{ margin-bottom: 15px; }}
.expand-all button {{ border: 1px solid #ddd; border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 0.85em; margin-right: 8px; }}
</style>
</head>
<body>
<div class="container">
    <h1>SpectralBench v5 — Scenario Review</h1>
    <div class="subtitle">XANES benchmark scenarios with PDF evidence screenshots</div>

    <div class="stats">
        <div class="stat"><div class="stat-num">{len(scenarios)}</div><div class="stat-label">Scenarios</div></div>
        <div class="stat"><div class="stat-num">{len(by_doi)}</div><div class="stat-label">Papers</div></div>
        <div class="stat"><div class="stat-num">{len(cat_counts)}</div><div class="stat-label">Categories</div></div>
        <div class="stat"><div class="stat-num">{len(facility_counts)}</div><div class="stat-label">Facilities</div></div>
    </div>

    <div class="filters">
        {''.join(filter_buttons)}
    </div>
    {'<div class="filters">' + "".join(facility_buttons) + '</div>' if facility_buttons else ''}

    <div class="expand-all">
        <button onclick="expandAllPapers()">Expand All Papers</button>
        <button onclick="collapseAllPapers()">Collapse All</button>
    </div>

    <div id="papers">
        {''.join(papers_html)}
    </div>
</div>

<div class="evidence-overlay" id="imgOverlay" onclick="closeZoom()"></div>

<script>
function togglePaper(el) {{
    const body = el.nextElementSibling;
    const icon = el.querySelector('.toggle-icon');
    if (body.style.display === 'none') {{
        body.style.display = 'block';
        icon.classList.add('open');
    }} else {{
        body.style.display = 'none';
        icon.classList.remove('open');
    }}
}}

function togglePaperEvidence(el) {{
    const body = el.nextElementSibling;
    const icon = el.querySelector('.toggle-icon');
    if (body.style.display === 'none') {{
        body.style.display = 'block';
        icon.classList.add('open');
    }} else {{
        body.style.display = 'none';
        icon.classList.remove('open');
    }}
}}

function toggleCondition(el) {{
    const body = el.nextElementSibling;
    const icon = el.querySelector('.toggle-icon');
    if (body.style.display === 'none') {{
        body.style.display = 'block';
        icon.classList.add('open');
    }} else {{
        body.style.display = 'none';
        icon.classList.remove('open');
    }}
}}

function expandAllPapers() {{
    document.querySelectorAll('.paper-body').forEach(b => b.style.display = 'block');
    document.querySelectorAll('.paper-header .toggle-icon').forEach(i => i.classList.add('open'));
}}

function collapseAllPapers() {{
    document.querySelectorAll('.paper-body').forEach(b => b.style.display = 'none');
    document.querySelectorAll('.paper-header .toggle-icon').forEach(i => i.classList.remove('open'));
    document.querySelectorAll('.condition-body').forEach(b => b.style.display = 'none');
    document.querySelectorAll('.condition-header .toggle-icon').forEach(i => i.classList.remove('open'));
}}

let activeCat = 'all';
let activeFacility = 'all';

function applyFilters() {{
    document.querySelectorAll('.paper').forEach(p => {{
        const catMatch = activeCat === 'all' || p.dataset.cat === activeCat;
        const facMatch = activeFacility === 'all' || p.dataset.facility === activeFacility
                         || (activeFacility === '__none__' && !p.dataset.facility);
        if (catMatch && facMatch) {{
            p.classList.remove('hidden');
        }} else {{
            p.classList.add('hidden');
        }}
    }});
}}

document.querySelectorAll('.filter-btn:not(.facility-filter)').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn:not(.facility-filter)').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeCat = btn.dataset.cat;
        applyFilters();
    }});
}});

document.querySelectorAll('.filter-btn.facility-filter').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn.facility-filter').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFacility = btn.dataset.facility;
        applyFilters();
    }});
}});

function markReview(btn, status) {{
    const condition = btn.closest('.condition');
    const btns = btn.parentElement.querySelectorAll('.review-btn:not(.issue)');
    btns.forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    condition.classList.remove('reviewed-correct', 'reviewed-incorrect');
    condition.classList.add('reviewed-' + status);
}}

function reportIssue(btn, scenarioNum, scenarioId, element, category, doi) {{
    const comment = btn.parentElement.querySelector('.review-comment').value || '';
    const title = encodeURIComponent('[#' + scenarioNum + '] ' + scenarioId + ' (DOI: ' + doi + ')');
    let body = '**Scenario #:** ' + scenarioNum + '\\n';
    body += '**Scenario ID:** `' + scenarioId + '`\\n';
    body += '**DOI:** ' + doi + '\\n';
    body += '**Element/Edge:** ' + element + '\\n';
    body += '**Category:** ' + category + '\\n\\n';
    body += '**Issue:**\\n' + (comment || '(describe the issue here)') + '\\n';
    const url = 'https://github.com/yuxi120407/SpectralBench/issues/new?title=' + title + '&body=' + encodeURIComponent(body) + '&labels=scenario-review';
    window.open(url, '_blank');
}}

// Click-to-zoom for evidence images
document.addEventListener('click', function(e) {{
    if (e.target.classList.contains('evidence-img')) {{
        if (e.target.classList.contains('zoomed')) {{
            closeZoom();
        }} else {{
            document.querySelectorAll('.evidence-img.zoomed').forEach(i => i.classList.remove('zoomed'));
            e.target.classList.add('zoomed');
            document.getElementById('imgOverlay').style.display = 'block';
        }}
    }}
}});

function closeZoom() {{
    document.querySelectorAll('.evidence-img.zoomed').forEach(i => i.classList.remove('zoomed'));
    document.getElementById('imgOverlay').style.display = 'none';
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeZoom();
}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate review HTML with PDF evidence screenshots")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--papers-dir", type=str, default=DEFAULT_PAPERS_DIR,
                        help="Directory containing PDF files named by DOI")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI for PDF page rendering (default: 150)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: review_v2.html in script dir)")
    args = parser.parse_args()

    global DPI
    DPI = args.dpi

    if args.input:
        input_file = args.input
    else:
        data_dir = os.path.join(BASE_DIR, "..", "data")
        files = sorted(glob.glob(os.path.join(data_dir, "scenarios_v5*.json")))
        input_file = files[-1] if files else None

    if not input_file or not os.path.exists(input_file):
        print(f"Not found: {input_file}")
        return

    if not fitz:
        print("WARNING: pymupdf (fitz) not installed — HTML will have no PDF screenshots")

    with open(input_file) as f:
        scenarios = json.load(f)

    print(f"Loaded {len(scenarios)} scenarios from {os.path.basename(input_file)}")
    print(f"Papers directory: {args.papers_dir}")
    print(f"DPI: {DPI}")

    output = args.output or os.path.join(BASE_DIR, "review_v2.html")
    output_dir = os.path.dirname(os.path.abspath(output))

    html_content = build_html(scenarios, args.papers_dir, output_dir)

    with open(output, "w") as f:
        f.write(html_content)

    evidence_dir = os.path.join(output_dir, "evidence_original")
    evidence_size = sum(
        os.path.getsize(os.path.join(evidence_dir, f))
        for f in os.listdir(evidence_dir)
    ) / 1024 / 1024

    print(f"Rendered {len(_page_cache)} unique PDF pages -> evidence/")
    print(f"Evidence images: {evidence_size:.1f} MB ({len(os.listdir(evidence_dir))} files)")
    print(f"HTML: {os.path.getsize(output) / 1024:.0f} KB")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
