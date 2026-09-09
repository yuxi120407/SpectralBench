"""
Detect parameter-sweep series within extracted paper data.

Groups correlated conditions from the same paper (e.g., 5 discharge states
of the same electrode, 10 composition points on a combinatorial film) into
series that should be presented as a single multi-condition benchmark scenario.

Usage:
    python detect_series.py xlsx_extracted_20260908_203418.json
    python detect_series.py xlsx_extracted_*.json --verbose
"""

import json
import re
import sys
import os
from collections import Counter, defaultdict


def parse_numeric(val):
    """Try to extract a numeric value from a key_variable_value string.

    Only returns a number if the value is primarily numeric (not text with
    an incidental number like 'chemically treated at 4°C').
    """
    if val is None:
        return None
    val = str(val).strip()
    # Direct float
    try:
        return float(val)
    except ValueError:
        pass
    # Common numeric patterns: "x=0.5", "40%", "0.06 (Ti fraction)", "2.5 V"
    # But NOT "chemically treated", "as prepared", "TZ-2", "pristine"
    # Reject if the string contains letters (except short prefixes like "x=")
    alpha_chars = sum(1 for c in val if c.isalpha())
    if alpha_chars > 1:
        return None
    # Patterns like "x=0.5", "0.06", "40%"
    m = re.match(r'^[a-zA-Z_]?\s*=?\s*([-+]?\d*\.?\d+)\s*%?\s*(?:\(.*\))?$', val)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def group_by_element_edge(conditions):
    """Split conditions into groups by element/edge parsed from sample_id or phase names."""
    groups = defaultdict(list)
    for c in conditions:
        sid = c.get("sample_id", "")
        # Try to parse "Ti K-edge" or "Zn K-edge" from sample_id
        edge_match = re.match(r'^(\w+)\s+[KL]\d?-edge', sid)
        if edge_match:
            key = edge_match.group(0)
        else:
            key = "all"
        groups[key].append(c)
    return dict(groups)


def find_sweep_variable(conditions):
    """Find the most common key_variable across conditions."""
    kvs = [c.get("key_variable", "") for c in conditions if c.get("key_variable")]
    if not kvs:
        return None, 0
    counter = Counter(kvs)
    best, count = counter.most_common(1)[0]
    return best, count


def split_sub_series(conditions, sweep_var):
    """Split conditions into sub-series if there's a secondary grouping variable.

    E.g., Pd-Au catalyst has "Au25 + 6 Pd" and "Au25 + 0.5 Pd" sub-groups,
    each varying treatment (as prepared / chemical / thermal).

    Strategy: group by longest common prefix in sample_id, splitting at the
    last comma/separator before the sweep variable value changes.
    """
    if len(conditions) < 4:
        return [conditions]

    # Approach: find the part of sample_id that stays constant within a group.
    # Split sample_id on comma and take the part before the last comma.
    groups = defaultdict(list)
    for c in conditions:
        sid = c.get("sample_id", "")
        # Split on last comma to get "base, variation"
        if "," in sid:
            base = sid.rsplit(",", 1)[0].strip()
        else:
            base = sid
        groups[base].append(c)

    # Only split if we get multiple groups each with 2+ conditions
    if len(groups) > 1 and min(len(v) for v in groups.values()) >= 2:
        return list(groups.values())

    # Fallback: try splitting on common prefix patterns
    groups2 = defaultdict(list)
    sids = [c.get("sample_id", "") for c in conditions]
    # Find numeric differences in sample_ids (e.g., "6 Pd" vs "0.5 Pd")
    for c in conditions:
        sid = c.get("sample_id", "")
        # Replace numbers with a placeholder to find the structural pattern
        pattern = re.sub(r'\d+\.?\d*', '#', sid)
        groups2[pattern].append(c)

    if len(groups2) > 1 and min(len(v) for v in groups2.values()) >= 2:
        return list(groups2.values())

    return [conditions]


def detect_series(paper):
    """Detect parameter-sweep series within a single paper's conditions.

    Returns:
        series: list of dicts, each with:
            - series_id, sweep_variable, sweep_type, sweep_values, conditions
        standalone: list of conditions not part of any series
    """
    a = paper.get("fulltext_analysis", {})
    conditions = a.get("sample_conditions", [])
    cl = paper.get("classification") or a.get("classification", {})
    category = cl.get("category", "synthesis") if cl else "synthesis"

    # Pure phase: never form series (individual reference compounds)
    if category == "pure_phase":
        return [], conditions

    if len(conditions) < 3:
        return [], conditions

    doi = paper.get("doi", "unknown")
    element = a.get("element", "?")
    edge = a.get("edge", "K")

    # Group by element/edge for multi-element papers
    elem_groups = group_by_element_edge(conditions)

    all_series = []
    all_standalone = []

    for group_key, group_conds in elem_groups.items():
        if len(group_conds) < 3:
            all_standalone.extend(group_conds)
            continue

        sweep_var, sweep_count = find_sweep_variable(group_conds)

        if not sweep_var or sweep_count < 3:
            all_standalone.extend(group_conds)
            continue

        # Separate conditions that share the sweep variable from outliers
        in_sweep = [c for c in group_conds if c.get("key_variable") == sweep_var]
        outliers = [c for c in group_conds if c.get("key_variable") != sweep_var]

        if len(in_sweep) < 3:
            all_standalone.extend(group_conds)
            continue

        # Try to split into sub-series (e.g., different loadings)
        sub_groups = split_sub_series(in_sweep, sweep_var)

        for sub_conds in sub_groups:
            if len(sub_conds) < 3:
                all_standalone.extend(sub_conds)
                continue

            # Try numeric ordering
            numeric_pairs = []
            for c in sub_conds:
                num = parse_numeric(c.get("key_variable_value"))
                if num is not None:
                    numeric_pairs.append((num, c))

            sweep_values = None
            sorted_conds = None
            sweep_type = None

            # Validate numeric series: need distinct values that span a range
            if len(numeric_pairs) >= 3:
                vals = sorted(set(p[0] for p in numeric_pairs))
                if len(vals) >= 3:
                    numeric_pairs.sort(key=lambda x: x[0])
                    sweep_values = [p[0] for p in numeric_pairs]
                    sorted_conds = [p[1] for p in numeric_pairs]
                    sweep_type = "numeric"

            # Fall back to categorical if numeric didn't work
            if sweep_values is None and len(sub_conds) >= 3:
                cat_values = [c.get("key_variable_value", "?") for c in sub_conds]
                # Reject categorical if too many duplicate values
                n_distinct = len(set(cat_values))
                if n_distinct >= len(cat_values) * 0.6:
                    sweep_values = cat_values
                    sorted_conds = sub_conds
                    sweep_type = "categorical"

            if sweep_values is None:
                all_standalone.extend(sub_conds)
                continue

            # Build series ID
            paper_short = (paper.get("title", "paper") or "paper")[:20].lower()
            paper_short = "".join(c if c.isalnum() else "_" for c in paper_short).strip("_")
            elem_short = group_key.replace(" ", "_") if group_key != "all" else element
            series_id = f"series_{paper_short}_{elem_short}_{sweep_var}"
            series_id = re.sub(r'[^a-zA-Z0-9_]', '_', series_id)

            # Extract expected trends from ground truth
            trends = compute_trends(sorted_conds)

            all_series.append({
                "series_id": series_id,
                "doi": doi,
                "element": elem_short if group_key != "all" else element,
                "edge": edge,
                "category": category,
                "sweep_variable": sweep_var,
                "sweep_type": sweep_type,
                "sweep_values": sweep_values,
                "n_conditions": len(sorted_conds),
                "conditions": sorted_conds,
                "expected_trends": trends,
                "causal_reasoning": a.get("causal_reasoning", ""),
                "condition_fraction_rules": a.get("condition_fraction_rules", []),
                "key_finding": a.get("key_finding", ""),
            })

        all_standalone.extend(outliers)

    return all_series, all_standalone


def compute_trends(sorted_conditions):
    """Compute expected phase fraction trends across a sorted series."""
    # Collect all phase names
    all_phases = set()
    for c in sorted_conditions:
        pf = c.get("phase_fractions", [])
        if isinstance(pf, list):
            for x in pf:
                all_phases.add(x.get("phase", ""))

    trends = {}
    for phase in all_phases:
        if not phase:
            continue
        values = []
        for c in sorted_conditions:
            pf = c.get("phase_fractions", [])
            frac = 0.0
            if isinstance(pf, list):
                for x in pf:
                    if x.get("phase") == phase:
                        frac = x.get("fraction", 0.0)
                        break
            values.append(frac)

        if not values:
            continue

        start = values[0]
        end = values[-1]

        # Determine direction
        if end > start + 0.05:
            direction = "increasing"
        elif end < start - 0.05:
            direction = "decreasing"
        else:
            direction = "constant"

        # Check monotonicity
        monotonic = True
        for i in range(1, len(values)):
            if direction == "increasing" and values[i] < values[i-1] - 0.02:
                monotonic = False
                break
            elif direction == "decreasing" and values[i] > values[i-1] + 0.02:
                monotonic = False
                break

        trends[phase] = {
            "direction": direction,
            "monotonic": monotonic,
            "start": round(start, 3),
            "end": round(end, 3),
            "values": [round(v, 3) for v in values],
        }

    return trends


def detect_all_series(extracted_papers):
    """Run series detection on all extracted papers.

    Returns:
        series_list: all detected series across all papers
        standalone_list: all standalone conditions (grouped by paper)
        paper_metadata: dict mapping DOI to paper-level metadata
    """
    series_list = []
    standalone_papers = []

    for p in extracted_papers:
        series, standalone = detect_series(p)
        if series:
            for s in series:
                s["paper_metadata"] = {
                    "title": p.get("title", ""),
                    "doi": p.get("doi", ""),
                    "authors": p.get("authors", []),
                    "year": p.get("year"),
                    "venue": p.get("venue", ""),
                }
                series_list.append(s)

        if standalone:
            standalone_papers.append({
                "paper": p,
                "conditions": standalone,
            })

    return series_list, standalone_papers


# ── CLI ───────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_series.py <extracted_papers.json> [--verbose]")
        sys.exit(1)

    verbose = "--verbose" in sys.argv

    with open(sys.argv[1]) as f:
        papers = json.load(f)

    series_list, standalone_papers = detect_all_series(papers)

    print(f"{'='*70}")
    print(f"SERIES DETECTION RESULTS")
    print(f"{'='*70}")
    print(f"Total papers: {len(papers)}")
    print(f"Series detected: {len(series_list)}")
    total_series_conds = sum(s["n_conditions"] for s in series_list)
    total_standalone = sum(len(sp["conditions"]) for sp in standalone_papers)
    print(f"Conditions in series: {total_series_conds}")
    print(f"Standalone conditions: {total_standalone}")

    for s in series_list:
        print(f"\n  SERIES: {s['series_id']}")
        print(f"    DOI: {s['doi']}")
        print(f"    Category: {s['category']} | {s['element']} {s['edge']}")
        print(f"    Sweep: {s['sweep_variable']} ({s['sweep_type']})")
        print(f"    Values: {s['sweep_values']}")
        print(f"    Conditions: {s['n_conditions']}")

        if verbose:
            for i, c in enumerate(s["conditions"]):
                sid = c.get("sample_id", "?")
                pf = c.get("phase_fractions", [])
                fstr = ", ".join(
                    f"{x.get('phase','?')}={x.get('fraction','?')}" for x in pf
                ) if isinstance(pf, list) else "?"
                print(f"      [{i+1}] {sid}: {fstr}")

            print(f"    Trends:")
            for phase, trend in s.get("expected_trends", {}).items():
                print(f"      {phase}: {trend['direction']} "
                      f"({trend['start']} → {trend['end']}) "
                      f"{'monotonic' if trend['monotonic'] else 'NON-monotonic'}")

    if standalone_papers:
        print(f"\n  STANDALONE CONDITIONS:")
        for sp in standalone_papers:
            doi = sp["paper"].get("doi", "?")
            print(f"    {doi}: {len(sp['conditions'])} conditions")
            if verbose:
                for c in sp["conditions"]:
                    sid = c.get("sample_id", "?")
                    print(f"      - {sid}")


if __name__ == "__main__":
    main()
