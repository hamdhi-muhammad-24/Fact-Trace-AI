# eda.py
"""
CM3604 Deep Learning Coursework — Exploratory Data Analysis & Quality Audit
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


def compute_eda_metrics(examples: List[FactExample], raw_bios: List[dict],
                        raw_passages: List[dict],
                        fact_to_passage_dict: Dict[str, List[dict]]) -> Dict[str, Any]:
    """
    Computes all structural, lexical, and distributional measurements dynamically from data.
    """
    metrics = {}

    # 1. Macro retrieval dataset metrics
    metrics["retrieval_total_records"] = len(raw_passages)
    total_retrieval_passage_entries = sum(len(r.get("passages", [])) for r in raw_passages)
    metrics["retrieval_total_passage_entries"] = total_retrieval_passage_entries
    metrics["retrieval_unique_fact_queries"] = len(fact_to_passage_dict)

    # 2. Evaluated development set metrics
    metrics["dev_total_biographies"] = len(raw_bios)
    metrics["dev_total_fact_instances"] = len(examples)
    metrics["dev_unique_fact_strings"] = len(set(ex.fact for ex in examples))

    # 3. Label distributions
    raw_labels = [ex.label for ex in examples]
    raw_label_counts = Counter(raw_labels)
    metrics["raw_label_counts"] = {
        "S": raw_label_counts.get("S", 0),
        "NS": raw_label_counts.get("NS", 0),
        "IR": raw_label_counts.get("IR", 0)
    }
    metrics["raw_label_percentages"] = {
        k: (v / len(examples)) * 100 for k, v in metrics["raw_label_counts"].items()
    }

    converted_labels = ["NS" if l == "IR" else l for l in raw_labels]
    converted_label_counts = Counter(converted_labels)
    metrics["converted_label_counts"] = {
        "S": converted_label_counts.get("S", 0),
        "NS": converted_label_counts.get("NS", 0)
    }
    metrics["converted_label_percentages"] = {
        k: (v / len(examples)) * 100 for k, v in metrics["converted_label_counts"].items()
    }

    # 4. Length distributions for atomic facts
    fact_words = [len(ex.fact.split()) for ex in examples]
    fact_tokens = [len(re.findall(r"\b[a-zA-Z0-9]+\b", ex.fact)) for ex in examples]
    fact_chars = [len(ex.fact) for ex in examples]

    metrics["fact_length_words"] = {
        "mean": float(np.mean(fact_words)),
        "median": float(np.median(fact_words)),
        "std": float(np.std(fact_words)),
        "min": int(np.min(fact_words)),
        "max": int(np.max(fact_words))
    }
    metrics["fact_length_tokens"] = {
        "mean": float(np.mean(fact_tokens)),
        "median": float(np.median(fact_tokens)),
        "std": float(np.std(fact_tokens)),
        "min": int(np.min(fact_tokens)),
        "max": int(np.max(fact_tokens))
    }
    metrics["fact_length_chars"] = {
        "mean": float(np.mean(fact_chars)),
        "median": float(np.median(fact_chars)),
        "min": int(np.min(fact_chars)),
        "max": int(np.max(fact_chars))
    }

    # 5. Length breakdown conditioned on converted ground-truth label
    s_fact_words = [w for w, l in zip(fact_words, converted_labels) if l == "S"]
    ns_fact_words = [w for w, l in zip(fact_words, converted_labels) if l == "NS"]
    metrics["class_conditioned_fact_words"] = {
        "S_mean": float(np.mean(s_fact_words)),
        "S_median": float(np.median(s_fact_words)),
        "NS_mean": float(np.mean(ns_fact_words)),
        "NS_median": float(np.median(ns_fact_words))
    }

    # 6. Associated evidence metrics for evaluated facts
    passages_per_fact = [len(ex.passages) for ex in examples]
    metrics["dev_total_associated_passage_entries"] = sum(passages_per_fact)
    metrics["passages_per_fact"] = {
        "mean": float(np.mean(passages_per_fact)),
        "median": float(np.median(passages_per_fact)),
        "min": int(np.min(passages_per_fact)),
        "max": int(np.max(passages_per_fact)),
        "distribution": dict(Counter(passages_per_fact))
    }

    all_dev_passage_texts = [p.get("text", "") for ex in examples for p in ex.passages]
    unique_dev_passage_texts = set(all_dev_passage_texts)
    metrics["dev_unique_passage_texts"] = len(unique_dev_passage_texts)

    passage_words = [len(t.split()) for t in all_dev_passage_texts]
    passage_tokens = [len(re.findall(r"\b[a-zA-Z0-9]+\b", t)) for t in all_dev_passage_texts]
    passage_chars = [len(t) for t in all_dev_passage_texts]

    metrics["passage_length_words"] = {
        "mean": float(np.mean(passage_words)),
        "median": float(np.median(passage_words)),
        "std": float(np.std(passage_words)),
        "min": int(np.min(passage_words)),
        "max": int(np.max(passage_words))
    }
    metrics["passage_length_tokens"] = {
        "mean": float(np.mean(passage_tokens)),
        "median": float(np.median(passage_tokens)),
        "min": int(np.min(passage_tokens)),
        "max": int(np.max(passage_tokens))
    }

    # 7. Extreme length asymmetry ratio
    metrics["length_asymmetry_word_ratio"] = float(np.mean(passage_words) / np.mean(fact_words))

    # 8. Numerical tokens in claims
    facts_with_numbers = sum(1 for ex in examples if any(c.isdigit() for c in ex.fact))
    metrics["facts_with_numbers_count"] = facts_with_numbers
    metrics["facts_with_numbers_percentage"] = (facts_with_numbers / len(examples)) * 100

    # 9. Topic-level distributions
    topic_fact_counts = defaultdict(lambda: {"total": 0, "S": 0, "NS": 0})
    for bio in raw_bios:
        t = bio.get("topic", "Unknown")
        if bio.get("annotations"):
            for sent in bio["annotations"]:
                if sent.get("human-atomic-facts"):
                    for f in sent["human-atomic-facts"]:
                        topic_fact_counts[t]["total"] += 1
                        conv = "NS" if f.get("label") == "IR" else f.get("label")
                        if conv == "S":
                            topic_fact_counts[t]["S"] += 1
                        elif conv == "NS":
                            topic_fact_counts[t]["NS"] += 1

    metrics["topic_breakdown"] = dict(topic_fact_counts)
    return metrics


def generate_eda_visualizations(metrics: Dict[str, Any], audit: Dict[str, Any],
                                examples: List[FactExample],
                                output_dir: str = "outputs/eda/figures") -> List[str]:
    """
    Renders all Phase 1 figures and image tables using Matplotlib Agg backend.
    All sample sizes, percentages, duplicate groups, and bounds are derived dynamically
    from the current dataset without hardcoded numbers.
    Returns list of saved image paths.
    """
    if not _HAS_MATPLOTLIB:
        print("[Warning] Matplotlib not installed. Skipping figure generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    N = len(examples)
    dup_details = audit.get("duplicate_fact_details", [])
    dup_count = audit.get("duplicate_fact_strings_count", 0)

    # Dynamic label counts and percentages
    raw_s = metrics["raw_label_counts"]["S"]
    raw_ns = metrics["raw_label_counts"]["NS"]
    raw_ir = metrics["raw_label_counts"]["IR"]
    raw_s_pct = (raw_s / N * 100.0) if N > 0 else 0.0
    raw_ns_pct = (raw_ns / N * 100.0) if N > 0 else 0.0
    raw_ir_pct = (raw_ir / N * 100.0) if N > 0 else 0.0

    conv_s = metrics["converted_label_counts"]["S"]
    conv_ns = metrics["converted_label_counts"]["NS"]
    conv_s_pct = (conv_s / N * 100.0) if N > 0 else 0.0
    conv_ns_pct = (conv_ns / N * 100.0) if N > 0 else 0.0

    # Dynamically compute deterministic upper bound from identical model inputs with conflicting labels
    input_groups = defaultdict(list)
    for ex in examples:
        key = (ex.fact, tuple(p.get("text", "") for p in ex.passages))
        input_groups[key].append("NS" if ex.label == "IR" else ex.label)
    unavoidable_errors = 0
    for lbls in input_groups.values():
        s_c = lbls.count("S")
        ns_c = lbls.count("NS")
        unavoidable_errors += len(lbls) - max(s_c, ns_c)
    deterministic_ceiling = ((N - unavoidable_errors) / N * 100.0) if N > 0 else 100.0

    # 1. Summary table rendered as an image
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    if dup_details:
        d_first = dup_details[0]
        c_first_raw = ", ".join(d_first["raw_labels"])
        c_first_conv = ", ".join(d_first["converted_labels"])
        conflict_status = "Conflicting gold" if d_first["has_label_conflict"] else "Consistent gold"
        dup_row_val = f"{d_first['occurrences']} occurrences (0-based idx {d_first['zero_based_indices']})\n{conflict_status}: {c_first_raw} -> {c_first_conv}"
        dup_row_label = f"Duplicate Fact Anomaly ('{d_first['fact']}')"
    else:
        dup_row_label = "Duplicate Fact Anomaly"
        dup_row_val = "None detected (all fact strings unique)"

    dup_text = f"{metrics['dev_unique_fact_strings']} ({dup_count} duplicate{'s' if dup_count != 1 else ''})" if dup_count > 0 else f"{metrics['dev_unique_fact_strings']} (0 duplicates)"

    table_data = [
        ["Benchmark Retrieval Records (JSON lines)", f"{metrics['retrieval_total_records']:,}"],
        ["Benchmark Total Passage Entries", f"{metrics['retrieval_total_passage_entries']:,}"],
        ["Benchmark Unique Query Fact Strings", f"{metrics['retrieval_unique_fact_queries']:,}"],
        ["Evaluated Biographies (ChatGPT topics)", f"{metrics['dev_total_biographies']}"],
        ["Evaluated Atomic Fact Instances", f"{metrics['dev_total_fact_instances']}"],
        ["Unique Atomic Fact Text Strings", dup_text],
        ["Raw Annotations Distribution", f"S: {raw_s} ({raw_s_pct:.2f}%) | NS: {raw_ns} ({raw_ns_pct:.2f}%) | IR: {raw_ir} ({raw_ir_pct:.2f}%)"],
        ["Binary Evaluation Labels (IR -> NS)", f"S: {conv_s} ({conv_s_pct:.2f}%) | NS: {conv_ns} ({conv_ns_pct:.2f}%)"],
        ["Total Associated Passage Entries", f"{metrics['dev_total_associated_passage_entries']}"],
        ["Unique Retrieved Passage Text Chunks", f"{metrics['dev_unique_passage_texts']} (reused across claims)"],
        ["Passages per Fact (min / mean / max)", f"{metrics['passages_per_fact']['min']} / {metrics['passages_per_fact']['mean']:.2f} / {metrics['passages_per_fact']['max']}"],
        ["Atomic Fact Word Length (mean / median / max)", f"{metrics['fact_length_words']['mean']:.2f} / {metrics['fact_length_words']['median']:.1f} / {metrics['fact_length_words']['max']}"],
        ["Passage Word Length (mean / median / max)", f"{metrics['passage_length_words']['mean']:.2f} / {metrics['passage_length_words']['median']:.1f} / {metrics['passage_length_words']['max']}"],
        ["Length Asymmetry Ratio (Passage / Fact)", f"{metrics['length_asymmetry_word_ratio']:.2f}x"],
        ["Claims Containing Numbers / Dates", f"{metrics['facts_with_numbers_count']} ({metrics['facts_with_numbers_percentage']:.1f}%)"],
        [dup_row_label, dup_row_val],
        ["Deterministic Upper Bound Accuracy Ceiling", f"{N - unavoidable_errors} / {N} = {deterministic_ceiling:.2f}%\n(Assumes deterministic f(claim, evidence); {unavoidable_errors} conflicting label error{'s' if unavoidable_errors != 1 else ''})"]
    ]

    col_widths = [0.48, 0.52]
    table = ax.table(cellText=table_data, colLabels=["Dataset Characteristic / Metric", "Empirically Verified Value"],
                     loc="center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("#ffffff")
            if col == 1:
                cell.set_text_props(weight="semibold")

    plt.title("FActScore Development Dataset Structural Summary", fontsize=12, fontweight="bold", pad=12, color="#1E293B")
    path = os.path.join(output_dir, "eda_summary_table.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 2. Label distribution chart (Raw 3-class vs Converted 2-class)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    raw_cats = ["Supported (S)", "Not Supported (NS)", "Irrelevant (IR)"]
    raw_vals = [raw_s, raw_ns, raw_ir]
    raw_pcts = [raw_s_pct, raw_ns_pct, raw_ir_pct]
    colors_raw = ["#0F766E", "#DC2626", "#64748B"]

    bars1 = ax1.bar(raw_cats, raw_vals, color=colors_raw, edgecolor="#1E293B", width=0.55)
    ax1.set_title(f"Raw FActScore Annotations (N = {N})", fontsize=11, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Count of Atomic Facts", fontweight="bold", color="#1E293B")
    ax1.set_ylim(0, max(raw_vals) * 1.25 if raw_vals else 10)
    for bar, val, pct in zip(bars1, raw_vals, raw_pcts):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + (max(raw_vals)*0.02), f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    conv_cats = ["Supported (S)", "Not Supported (NS)\n[incl. IR -> NS]"]
    conv_vals = [conv_s, conv_ns]
    conv_pcts = [conv_s_pct, conv_ns_pct]
    colors_conv = ["#0F766E", "#DC2626"]

    bars2 = ax2.bar(conv_cats, conv_vals, color=colors_conv, edgecolor="#1E293B", width=0.45)
    ax2.set_title(f"Official Binary Evaluation Labels (N = {N})", fontsize=11, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Count of Atomic Facts", fontweight="bold", color="#1E293B")
    ax2.set_ylim(0, max(conv_vals) * 1.25 if conv_vals else 10)
    for bar, val, pct in zip(bars2, conv_vals, conv_pcts):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + (max(conv_vals)*0.02), f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_label_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    saved_files.append(path)

    # 3. Length distributions comparing facts vs passages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    all_facts_words = [len(ex.fact.split()) for ex in examples]
    min_f_len = min(all_facts_words) if all_facts_words else 1
    max_f_len = max(all_facts_words) if all_facts_words else 15
    fact_bins = range(min_f_len, max_f_len + 2)
    ax1.hist(all_facts_words, bins=fact_bins, color="#2563EB", edgecolor="#1E293B", alpha=0.85, rwidth=0.85)
    ax1.axvline(metrics["fact_length_words"]["mean"], color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {metrics['fact_length_words']['mean']:.2f}")
    ax1.axvline(metrics["fact_length_words"]["median"], color="#0F766E", linestyle=":", linewidth=2, label=f"Median: {metrics['fact_length_words']['median']:.1f}")
    ax1.set_title("Atomic Fact Length Distribution", fontsize=11, fontweight="bold", color="#1E293B")
    ax1.set_xlabel("Word Count per Claim", fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Frequency", fontweight="bold", color="#1E293B")
    ax1.legend(loc="upper right")
    tick_step = max(1, (max_f_len - min_f_len) // 10)
    ax1.set_xticks(range(min_f_len, max_f_len + 1, tick_step))

    all_passage_words = [len(p.get("text", "").split()) for ex in examples for p in ex.passages]
    ax2.hist(all_passage_words, bins=20, color="#0F766E", edgecolor="#1E293B", alpha=0.85)
    ax2.axvline(metrics["passage_length_words"]["mean"], color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {metrics['passage_length_words']['mean']:.2f}")
    ax2.axvline(metrics["passage_length_words"]["median"], color="#1E293B", linestyle=":", linewidth=2, label=f"Median: {metrics['passage_length_words']['median']:.1f}")
    ax2.set_title("Retrieved Wikipedia Passage Length Distribution", fontsize=11, fontweight="bold", color="#1E293B")
    ax2.set_xlabel("Word Count per Passage", fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Frequency", fontweight="bold", color="#1E293B")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_length_distributions.png")
    plt.savefig(path, dpi=200)
    plt.close()
    saved_files.append(path)

    # 4. Length asymmetry ratio graphic
    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    categories = ["Atomic Claim (Hypothesis)", "Wikipedia Passage (Premise)"]
    means = [metrics["fact_length_words"]["mean"], metrics["passage_length_words"]["mean"]]
    colors = ["#2563EB", "#0F766E"]

    bars = ax.barh(categories, means, color=colors, edgecolor="#1E293B", height=0.45)
    ax.set_xlabel("Mean Word Count", fontweight="bold", color="#1E293B")
    ax.set_title(f"Passage-to-Claim Length Asymmetry Ratio: {metrics['length_asymmetry_word_ratio']:.2f}x", fontsize=12, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, max(means) * 1.25 if means else 200)

    for bar, val in zip(bars, means):
        ax.text(val + 3, bar.get_y() + bar.get_height()/2, f"{val:.2f} words", va="center", fontsize=10, fontweight="bold")

    annotation_text = (
        f"Length Disparity: {metrics['length_asymmetry_word_ratio']:.2f}x\n"
        "Passage length substantially exceeds claim length;\n"
        "motivates evaluating both symmetric and asymmetric metrics."
    )
    ax.annotate(
        annotation_text,
        xy=(metrics["fact_length_words"]["mean"] + 3, 0),
        xytext=(max(means) * 0.35, 0.45),
        arrowprops=dict(facecolor="#1E293B", shrink=0.08, width=1, headwidth=5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1")
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_length_asymmetry_ratio.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 5. Passages per fact distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    dist = metrics["passages_per_fact"]["distribution"]
    x_vals = sorted(dist.keys())
    y_vals = [dist[k] for k in x_vals]
    pcts = [(v / N) * 100 for v in y_vals] if N > 0 else [0.0] * len(y_vals)

    bars = ax.bar([str(x) for x in x_vals], y_vals, color="#2563EB", edgecolor="#1E293B", width=0.5)
    ax.set_title("Distribution of Retrieved Passages per Atomic Claim", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xlabel("Number of Passages Retrieved (BM25)", fontweight="bold", color="#1E293B")
    ax.set_ylabel("Number of Facts", fontweight="bold", color="#1E293B")
    max_y = max(y_vals) if y_vals else 10
    ax.set_ylim(0, max_y * 1.25)

    for bar, val, pct in zip(bars, y_vals, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + (max_y * 0.02), f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_passages_per_fact.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 6. Data quality and anomalies summary table image
    fig, ax = plt.subplots(figsize=(13.0, 7.8), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    def wrap_cell(val: str, width: int) -> str:
        return "\n".join(textwrap.fill(line, width=width) for line in str(val).split("\n"))

    raw_anomaly_data = [
        ["Empty or Missing Atomic Fact Texts", f"{audit['empty_or_missing_fact_texts']}",
         "Clean (No missing inputs)" if audit['empty_or_missing_fact_texts'] == 0 else f"Anomaly ({audit['empty_or_missing_fact_texts']} empty claims)"],
        ["Facts with Empty Passage Lists", f"{audit['facts_with_empty_passage_lists']}",
         "Clean (All claims have candidate evidence)" if audit['facts_with_empty_passage_lists'] == 0 else f"Anomaly ({audit['facts_with_empty_passage_lists']} empty lists)"],
        ["Passages with Missing Wikipedia Titles", f"{audit['passages_with_missing_titles']}",
         "Clean (All passages mapped to source article)" if audit['passages_with_missing_titles'] == 0 else f"Anomaly ({audit['passages_with_missing_titles']} missing titles)"],
        ["Passages with Empty Body Text", f"{audit['passages_with_empty_text']}",
         "Clean (All evidence contains textual content)" if audit['passages_with_empty_text'] == 0 else f"Anomaly ({audit['passages_with_empty_text']} empty bodies)"],
        ["Duplicate Passages within Same Fact", f"{audit['facts_with_duplicate_passages']}",
         "Clean (No redundant duplicate passages per claim)" if audit['facts_with_duplicate_passages'] == 0 else f"Anomaly ({audit['facts_with_duplicate_passages']} redundant claims)"]
    ]

    if not dup_details:
        raw_anomaly_data.append(["Duplicate Fact Instances across Dataset", "0 duplicate strings", "All evaluated atomic fact strings are unique"])
        raw_anomaly_data.append(["Label Contradiction on Identical Inputs", "None Detected", "No identical inputs with conflicting ground truth"])
        raw_anomaly_data.append(["Theoretical Maximum Accuracy Ceiling", f"{N} / {N} = 100.00%", "No conflicting labels on identical model inputs"])
    else:
        for d_idx, d in enumerate(dup_details):
            raw_anomaly_data.append([
                f"Duplicate Fact #{d_idx+1} ('{d['fact']}')",
                f"{d['occurrences']} occurrences (indices {d['zero_based_indices']})",
                f"Identical claim string appears in {d['occurrences']} instances across dataset"
            ])
            for inst_i, (z_idx, r_lbl, c_lbl) in enumerate(zip(d["zero_based_indices"], d["raw_labels"], d["converted_labels"])):
                raw_anomaly_data.append([
                    f"  - Instance #{inst_i+1} (idx {z_idx}, pos {z_idx+1})",
                    f"Raw: {r_lbl} -> Converted: {c_lbl}",
                    f"Ground-truth label assignment for instance {inst_i+1}"
                ])

        has_conflict = any(d.get("has_label_conflict") for d in dup_details)
        conflict_finding = "CONTRADICTION DETECTED" if has_conflict else "Consistent Labels (No Contradiction)"
        conflict_impl = "Deterministic classifiers cannot predict both S and NS on identical inputs" if has_conflict else "Duplicate inputs share identical ground-truth label"
        raw_anomaly_data.append(["Label Contradiction on Identical Inputs", conflict_finding, conflict_impl])

        ceiling_finding = f"{N - unavoidable_errors} / {N} = {deterministic_ceiling:.2f}%"
        ceiling_impl = f"{unavoidable_errors} error is mathematically unavoidable on identical inputs" if unavoidable_errors > 0 else "100% accuracy remains theoretically achievable"
        raw_anomaly_data.append(["Theoretical Maximum Accuracy Ceiling", ceiling_finding, ceiling_impl])

    wrapped_anomaly_data = [
        [wrap_cell(row[0], 42), wrap_cell(row[1], 36), wrap_cell(row[2], 46)]
        for row in raw_anomaly_data
    ]

    col_widths = [0.35, 0.28, 0.37]
    table = ax.table(cellText=wrapped_anomaly_data, colLabels=["Data Quality Check", "Finding / Value", "Implication for Modeling"],
                     loc="center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.85)

    for (row, col), cell in table.get_celld().items():
        cell_text = cell.get_text().get_text()
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold")
        else:
            if "CONTRADICTION DETECTED" in cell_text:
                cell.set_facecolor("#FEE2E2")
                cell.set_text_props(weight="bold", color="#DC2626")
            elif row % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("#ffffff")

    plt.title("Data Quality & Benchmark Anomaly Audit", fontsize=12, fontweight="bold", pad=16, color="#1E293B")
    path = os.path.join(output_dir, "eda_data_quality_anomalies.png")
    plt.tight_layout(pad=1.5)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close()
    saved_files.append(path)

    # 7. Topic-level claim distribution and label balance
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    topics = sorted(metrics["topic_breakdown"].keys())
    s_counts = [metrics["topic_breakdown"][t]["S"] for t in topics]
    ns_counts = [metrics["topic_breakdown"][t]["NS"] for t in topics]

    y_pos = np.arange(len(topics))
    ax.barh(y_pos, s_counts, color="#0F766E", label="Supported (S)", edgecolor="#1E293B", height=0.6)
    ax.barh(y_pos, ns_counts, left=s_counts, color="#DC2626", label="Not Supported (NS)", edgecolor="#1E293B", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Atomic Facts", fontweight="bold", color="#1E293B")
    ax.set_title(f"Topic-Level Fact Counts and Label Balance ({len(topics)} Biographies)", fontsize=11, fontweight="bold", color="#1E293B")
    ax.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_topic_distribution.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 8. Evidence reuse within biographies
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    total_entries = metrics["dev_total_associated_passage_entries"]
    unique_texts = metrics["dev_unique_passage_texts"]
    shared_ratio = total_entries / unique_texts if unique_texts > 0 else 1.0

    ax.bar(["Total Associated Passage Entries", "Unique Wikipedia Text Chunks"], [total_entries, unique_texts],
           color=["#2563EB", "#0F766E"], edgecolor="#1E293B", width=0.45)
    ax.set_ylabel("Count", fontweight="bold", color="#1E293B")
    ax.set_title(f"Evidence Reuse Across Evaluated Facts (Reuse Factor: {shared_ratio:.2f}x)", fontsize=11, fontweight="bold", color="#1E293B")
    max_ev = max(total_entries, unique_texts) if max(total_entries, unique_texts) > 0 else 10
    ax.text(0, total_entries + (max_ev * 0.015), f"{total_entries}", ha="center", fontweight="bold")
    ax.text(1, unique_texts + (max_ev * 0.015), f"{unique_texts}", ha="center", fontweight="bold")
    ax.set_ylim(0, max_ev * 1.25)

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_evidence_reuse.png")
    plt.savefig(path, dpi=200)
    plt.close()
    saved_files.append(path)

    return saved_files


def run_eda(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
            dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
            figures_dir: str = "outputs/eda/figures",
            outputs_dir: str = "outputs/eda") -> None:
    """
    Main entry point for Phase 1 EDA pipeline.
    Executes dataset loading, data-quality audit, metric computation,
    and saves both summary JSON and visual figures.
    """
    print("[Phase 1 EDA] Starting dataset loading and dynamic analysis...")

    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)

    print(f"[Phase 1 EDA] Loaded {len(raw_passages):,} retrieval lines and {len(examples)} development fact instances.")

    audit = run_data_quality_audit(passages_path, dev_path, examples, raw_bios, raw_passages)
    print(f"[Phase 1 EDA] Data-quality audit complete. Duplicate fact strings found: {audit['duplicate_fact_strings_count']}")

    metrics = compute_eda_metrics(examples, raw_bios, raw_passages, passages_dict)
    print(f"[Phase 1 EDA] Computed metrics. Claim-to-passage length asymmetry: {metrics['length_asymmetry_word_ratio']:.2f}x")

    os.makedirs(outputs_dir, exist_ok=True)
    summary_record = {
        "metrics": metrics,
        "data_quality_audit": audit
    }
    json_path = os.path.join(outputs_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)
    print(f"[Phase 1 EDA] Saved machine-readable audit record to '{json_path}'.")

    saved_figures = generate_eda_visualizations(metrics, audit, examples, figures_dir)
    print(f"[Phase 1 EDA] Generated {len(saved_figures)} figures in '{figures_dir}/':")
    for fig_path in saved_figures:
        print(f"  - {fig_path}")

    print("[Phase 1 EDA] Execution completed successfully.")


if __name__ == "__main__":
    run_eda()
