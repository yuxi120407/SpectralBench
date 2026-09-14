"""
Generate a self-contained HTML review page from stage1 scenarios JSON.

Usage:
    python generate_review_html.py
    python generate_review_html.py --input stage1_examples_v3_20260909_181132.json
"""

import json
import glob
import os
import re
import html
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    """Format full ground truth as HTML for display."""
    parts = []

    if category == "pure_phase":
        parts.append(f'<div class="gt-field"><span class="gt-key">Compound:</span> {html.escape(str(gt.get("compound", "?")))}</div>')
        parts.append(f'<div class="gt-field"><span class="gt-key">Oxidation state:</span> {html.escape(str(gt.get("oxidation_state", "?")))}</div>')
        parts.append(f'<div class="gt-field"><span class="gt-key">Crystal structure:</span> {html.escape(str(gt.get("crystal_structure", "?")))}</div>')
        parts.append(f'<div class="gt-field"><span class="gt-key">Coordination:</span> {html.escape(str(gt.get("coordination", "?")))}</div>')

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

            # old format fallback
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
        # v4 material/measurement/fit_method (top of GT block for context)
        material = gt.get("material", "")
        if material:
            parts.append(f'<div class="gt-field"><span class="gt-key">Material:</span> {html.escape(material)}</div>')
        measurement = gt.get("measurement", "")
        if measurement:
            parts.append(f'<div class="gt-field"><span class="gt-key">Measurement:</span> {html.escape(measurement)}</div>')
        fit_method = gt.get("fit_method", "")
        if fit_method:
            parts.append(f'<div class="gt-field"><span class="gt-key">Fit method:</span> {html.escape(fit_method)}</div>')

        # Phase fractions
        fracs = gt.get("fractions", {})
        if fracs:
            unc = gt.get("fractions_uncertainty_pct")
            unc_label = f' <span style="color:#666;font-size:0.9em">(±{unc}%)</span>' if unc else ''
            parts.append(f'<div class="gt-field"><span class="gt-key">Phase fractions:{unc_label}</span></div>')
            parts.append('<table class="gt-table"><tr><th>Phase</th><th>Fraction</th></tr>')
            for phase, frac in fracs.items():
                parts.append(f'<tr><td>{html.escape(phase)}</td><td>{frac}</td></tr>')
            parts.append('</table>')

        # v4 fit_basis (preferred) or v3 candidate_phases / recommended_references
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

        # v4 source_evidence
        se = gt.get("source_evidence", "")
        if se:
            parts.append(f'<div class="gt-field"><span class="gt-key">Source evidence:</span> {html.escape(se)}</div>')

    # Key reasoning (both v3 and v4)
    reasoning = gt.get("key_reasoning", "")
    if reasoning:
        parts.append(f'<div class="gt-field"><span class="gt-key">Key reasoning:</span> {html.escape(reasoning)}</div>')

    # v4 reasoning_source
    rs = gt.get("reasoning_source", "")
    if rs:
        parts.append(f'<div class="gt-field"><span class="gt-key">Reasoning source:</span> {html.escape(rs)}</div>')

    return "\n".join(parts)


def format_rubric_html(rubric):
    """Format rubric as HTML table."""
    if not rubric:
        return ""
    parts = ['<table class="rubric-table"><tr><th>Category</th><th>Max</th><th>Criteria</th></tr>']
    for cat_name, cat_data in rubric.items():
        max_score = cat_data.get("max_score", "?")
        criteria = cat_data.get("criteria", [])
        criteria_lines = []
        for c in criteria:
            pts = c.get("points", "?")
            desc = c.get("description", "")
            criteria_lines.append(f'{pts} pts — {html.escape(desc)}')
        criteria_html = "<br>".join(criteria_lines)
        display_name = cat_name.replace("_", " ").title()
        parts.append(f'<tr><td class="rubric-cat">{html.escape(display_name)}</td><td class="rubric-max">{max_score}</td><td class="rubric-criteria">{criteria_html}</td></tr>')
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


def build_html(scenarios):
    from collections import OrderedDict

    by_doi = OrderedDict()
    for s in scenarios:
        doi = s.get("source_paper_full", {}).get("doi", "?")
        if doi not in by_doi:
            by_doi[doi] = {
                "meta": s.get("source_paper_full", {}),
                "category": s.get("category", "?"),
                "scenarios": [],
            }
        by_doi[doi]["scenarios"].append(s)

    cat_counts = {}
    for s in scenarios:
        c = s.get("category", "?")
        cat_counts[c] = cat_counts.get(c, 0) + 1

    filter_buttons = []
    filter_buttons.append('<button class="filter-btn active" data-cat="all">All ({0})</button>'.format(len(scenarios)))
    for cat in ["synthesis", "electrochemistry", "pure_phase", "thin_film", "catalyst", "environmental"]:
        if cat in cat_counts:
            label = CAT_LABELS.get(cat, cat)
            color = CAT_COLORS.get(cat, "#eee")
            filter_buttons.append(
                f'<button class="filter-btn" data-cat="{cat}" style="background:{color}">'
                f'{label} ({cat_counts[cat]})</button>'
            )

    papers_html = []
    scenario_idx = 0
    for doi, paper in by_doi.items():
        meta = paper["meta"]
        cat = paper["category"]
        title = meta.get("title", "?")
        authors = meta.get("authors", [])
        author_str = authors[0] + " et al." if len(authors) > 3 else ", ".join(authors)
        color = CAT_COLORS.get(cat, "#f5f5f5")
        cat_label = CAT_LABELS.get(cat, cat)
        n = len(paper["scenarios"])

        conditions_html = []
        for s in paper["scenarios"]:
            scenario_idx += 1
            gt = s.get("ground_truth", {})
            cd = s.get("condition_details", {})
            elem, edge = extract_element_edge(s)
            material = extract_material(s.get("prompt", ""))
            fracs = format_fractions(gt)
            reasoning = gt.get("key_reasoning", "")
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

        papers_html.append(f"""
        <div class="paper" data-cat="{cat}">
            <div class="paper-header" style="border-left: 5px solid {color};" onclick="togglePaper(this)">
                <div class="paper-title">{html.escape(title)}</div>
                <div class="paper-meta">
                    <span class="cat-badge" style="background:{color}">{cat_label}</span>
                    <span class="paper-authors">{html.escape(author_str)}</span>
                    <span class="paper-doi">DOI: {html.escape(doi)}</span>
                    <span class="paper-count">{n} scenario{'s' if n > 1 else ''}</span>
                </div>
                <span class="toggle-icon">&#9654;</span>
            </div>
            <div class="paper-body" style="display:none">
                {''.join(conditions_html)}
            </div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SpectralBench v3 — Review</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; color: #333; line-height: 1.5; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
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
.fractions {{ font-family: 'SF Mono', Monaco, monospace; font-size: 0.9em; background: #f7fafc; padding: 8px 12px; border-radius: 4px; }}
.reasoning {{ font-size: 0.9em; color: #4a5568; }}
.gt-box {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 12px; font-size: 0.85em; }}
.gt-field {{ margin: 4px 0; }}
.gt-key {{ font-weight: 600; color: #2d3748; }}
.gt-table {{ border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 0.9em; }}
.gt-table th {{ background: #edf2f7; padding: 6px 10px; text-align: left; font-weight: 600; font-size: 0.85em; border: 1px solid #e2e8f0; }}
.gt-table td {{ padding: 5px 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
.rubric-box {{ font-size: 0.85em; }}
.rubric-table {{ border-collapse: collapse; width: 100%; margin: 4px 0; }}
.rubric-table th {{ background: #edf2f7; padding: 6px 10px; text-align: left; font-weight: 600; font-size: 0.85em; border: 1px solid #e2e8f0; }}
.rubric-table td {{ padding: 5px 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
.rubric-cat {{ font-weight: 500; white-space: nowrap; }}
.rubric-max {{ text-align: center; font-weight: 600; color: #2c5282; }}
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
    <h1>SpectralBench v3 — Scenario Review</h1>
    <div class="subtitle">XANES benchmark scenarios for domain expert review</div>

    <div class="stats">
        <div class="stat"><div class="stat-num">{len(scenarios)}</div><div class="stat-label">Scenarios</div></div>
        <div class="stat"><div class="stat-num">{len(by_doi)}</div><div class="stat-label">Papers</div></div>
        <div class="stat"><div class="stat-num">{len(cat_counts)}</div><div class="stat-label">Categories</div></div>
    </div>

    <div class="filters">
        {''.join(filter_buttons)}
    </div>

    <div class="expand-all">
        <button onclick="expandAllPapers()">Expand All Papers</button>
        <button onclick="collapseAllPapers()">Collapse All</button>
    </div>

    <div id="papers">
        {''.join(papers_html)}
    </div>
</div>

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

document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const cat = btn.dataset.cat;
        document.querySelectorAll('.paper').forEach(p => {{
            if (cat === 'all' || p.dataset.cat === cat) {{
                p.classList.remove('hidden');
            }} else {{
                p.classList.add('hidden');
            }}
        }});
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
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    if args.input:
        input_file = args.input
    else:
        files = sorted(glob.glob(os.path.join(BASE_DIR, "stage1_examples_v3_*.json")))
        input_file = files[-1] if files else None

    if not input_file or not os.path.exists(input_file):
        print(f"Not found: {input_file}")
        return

    with open(input_file) as f:
        scenarios = json.load(f)

    print(f"Loaded {len(scenarios)} scenarios from {os.path.basename(input_file)}")

    html_content = build_html(scenarios)
    output = os.path.join(BASE_DIR, "review.html")
    with open(output, "w") as f:
        f.write(html_content)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
