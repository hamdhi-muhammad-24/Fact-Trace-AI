# eda.py
"""
CM3604 Deep Learning Coursework â€” Exploratory Data Analysis & Quality Audit
Phase 1: Dataset Loading, Structural Metrics, Anomaly Auditing, and Visualizations

This module executes comprehensive EDA and data-quality analysis over the FActScore
ChatGPT development dataset and BM25 retrieval corpus.

Outputs generated:
  - outputs/eda/summary.json (structured audit metrics & anomaly records)
  - outputs/eda/figures/ (8 publication-grade high-DPI analytical figures)
"""

import os
import sys
import json
import re
import time
import textwrap
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np

# Safe optional Matplotlib import with headless Agg backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Import official data loader helpers from factcheck.py
from factcheck import load_passages_data, load_dev_examples, FactExample


def run_data_quality_audit(passages_path: str, dev_path: str, examples: List[FactExample],
                           raw_bios: List[dict], raw_passages: List[dict]) -> Dict[str, Any]:
    """
    Performs comprehensive data-quality checks across both JSONL datasets.
    """
    audit = {}

    # 1. Fact text integrity
    empty_facts = [i for i, ex in enumerate(examples) if not ex.fact or not ex.fact.strip()]
    audit["empty_or_missing_fact_texts"] = len(empty_facts)

    # 2. Evidence passage integrity
    facts_with_empty_passages = []
    missing_titles = []
    empty_passage_texts = []
    for i, ex in enumerate(examples):
        if not ex.passages:
            facts_with_empty_passages.append(i)
        for p in ex.passages:
            if not p.get("title") or not p.get("title").strip():
                missing_titles.append(i)
            if not p.get("text") or not p.get("text").strip():
                empty_passage_texts.append(i)

    audit["facts_with_empty_passage_lists"] = len(facts_with_empty_passages)
    audit["passages_with_missing_titles"] = len(missing_titles)
    audit["passages_with_empty_text"] = len(empty_passage_texts)

    # 3. Duplicate passages within the same fact
    facts_with_duplicate_passages = 0
    for ex in examples:
        seen_texts = set()
        has_dup = False
        for p in ex.passages:
            t = p.get("text", "")
            if t in seen_texts:
                has_dup = True
                break
            seen_texts.add(t)
        if has_dup:
            facts_with_duplicate_passages += 1
    audit["facts_with_duplicate_passages"] = facts_with_duplicate_passages

    # 4. Duplicate atomic facts across dev examples
    fact_counts = Counter(ex.fact for ex in examples)
    duplicate_facts = {f: c for f, c in fact_counts.items() if c > 1}
    audit["duplicate_fact_strings_count"] = len(duplicate_facts)
    audit["duplicate_fact_details"] = []

    for fact_str, count in duplicate_facts.items():
        matching = [(i, ex) for i, ex in enumerate(examples) if ex.fact == fact_str]
        labels = [ex.label for _, ex in matching]
        converted = ["NS" if l == "IR" else l for l in labels]
        audit["duplicate_fact_details"].append({
            "fact": fact_str,
            "occurrences": count,
            "indices": [i for i, _ in matching],
            "zero_based_indices": [i for i, _ in matching],
            "one_based_positions": [i + 1 for i, _ in matching],
            "raw_labels": labels,
            "converted_labels": converted,
            "has_label_conflict": len(set(converted)) > 1
        })

    # 5. Dynamic Malformed JSON verification on raw lines
    def _scan_jsonl_file(file_path: str) -> Dict[str, int]:
        total_lines = 0
        valid_json = 0
        malformed = 0
        empty_lines = 0
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        empty_lines += 1
                        continue
                    total_lines += 1
                    try:
                        json.loads(s)
                        valid_json += 1
                    except Exception:
                        malformed += 1
        return {
            "total_non_empty_lines": total_lines,
            "valid_json_records": valid_json,
            "malformed_json_records": malformed,
            "empty_lines": empty_lines
        }

    passages_scan = _scan_jsonl_file(passages_path)
    dev_scan = _scan_jsonl_file(dev_path)

    audit["raw_lines_passages_file"] = passages_scan["total_non_empty_lines"]
    audit["valid_json_in_passages_file"] = passages_scan["valid_json_records"]
    audit["malformed_records_in_passages_file"] = passages_scan["malformed_json_records"]

    audit["raw_lines_dev_file"] = dev_scan["total_non_empty_lines"]
    audit["valid_json_in_dev_file"] = dev_scan["valid_json_records"]
    audit["malformed_records_in_dev_file"] = dev_scan["malformed_json_records"]

    return audit


