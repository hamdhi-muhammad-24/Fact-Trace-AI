"""
CM3604 Deep Learning Coursework — Phase 8: Model Comparison Module
Course: CM3604 Deep Learning

This module executes the comprehensive model comparison and evidence integrity audit
across all four fact-checking paradigms developed in the coursework:
  1. Random Guess Baseline (stochastic lower bound, seed=42)
  2. Always Supported Baseline (majority class baseline)
  3. Task 1: Word Overlap Fact-Checking (P4 Porter-Stemmed Number-Aware Word Recall)
  4. Task 2: Textual Entailment Fact-Checking (DeBERTa-v3-base-mnli-fever-anli,
             Sentence Formulation, Lexical Pruning gamma >= 0.05, Optimal Threshold tau* = 0.33)

Artifacts Generated:
  - outputs/comparison/overall_results.json
  - outputs/comparison/comparison.csv
  - outputs/comparison/figures/overall_model_comparison_table.png
  - outputs/comparison/figures/overall_accuracy_progression.png
  - outputs/comparison/figures/overall_model_comparison_metrics.png
  - outputs/comparison/figures/overall_model_comparison_profile.png

Strict Contract:
  - Silent import rule: Zero stdout/stderr output when imported (`python -c "import model_comparison"`).
  - Dynamic provenance: All figures and JSON metrics are populated strictly from genuine executed results.
"""

import os
import json
import csv
import textwrap
from typing import Dict, Any, List, Optional

import numpy as np

# Configure non-interactive backend for headless rendering
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def load_source_dev_records(dev_path: str = "data/dev_labeled_ChatGPT.jsonl") -> List[Dict[str, Any]]:
    """
    Extracts canonical ordered ground-truth fact records from the development JSONL dataset.
    Returns ordered list of dicts with keys: index, topic, fact, gold_label.
    """
    records = []
    idx = 0
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bio = json.loads(line)
            topic = bio.get("topic", "")
            for annot in bio.get("annotations", []):
                for f_obj in annot.get("human-atomic-facts", []):
                    raw_lbl = f_obj.get("label", "")
                    bin_lbl = "NS" if raw_lbl in ("NS", "IR") else raw_lbl
                    records.append({
                        "index": idx,
                        "topic": topic,
                        "fact": f_obj.get("text", "").strip(),
                        "gold_label": bin_lbl
                    })
                    idx += 1
    return records


def load_all_system_results(outputs_dir: str = "outputs",
                            cache_dir: str = "outputs/task2/cache",
                            dev_path: str = "data/dev_labeled_ChatGPT.jsonl") -> Dict[str, Any]:
    """
    Loads and cross-validates experimental results for all four coursework systems.
    Supports both modular subdirectories (outputs/baselines, outputs/task1, outputs/task2)
    and root legacy paths.
    """
    # 1. Resolve Baseline path
    baseline_path = os.path.join(outputs_dir, "baselines", "results.json")
    if not os.path.exists(baseline_path):
        baseline_path = "outputs/baselines/results.json"

    # 2. Resolve Task 1 path
    task1_path = os.path.join(outputs_dir, "task1", "experiments.json")
    if not os.path.exists(task1_path):
        task1_path = "outputs/task1/experiments.json"

    # 3. Resolve Task 2 results path
    task2_path = os.path.join(outputs_dir, "task2", "results.json")
    if not os.path.exists(task2_path):
        task2_path = "outputs/task2/results.json"

    # 4. Resolve Benchmarks path
    benchmarks_path = os.path.join(cache_dir, "inference_benchmarks.json")
    if not os.path.exists(benchmarks_path):
        benchmarks_path = "outputs/task2/cache/inference_benchmarks.json"

    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Missing canonical baseline results file: '{baseline_path}'. Run 'python experiments.py --baselines' first.")
    if not os.path.exists(task1_path):
        raise FileNotFoundError(f"Missing canonical Task 1 experiments file: '{task1_path}'. Run 'python experiments.py --task1' first.")
    if not os.path.exists(task2_path):
        raise FileNotFoundError(f"Missing canonical Task 2 results file: '{task2_path}'. Run 'python experiments.py --task2' first.")

    with open(baseline_path, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(task1_path, "r", encoding="utf-8") as f:
        t1_data = json.load(f)
    with open(task2_path, "r", encoding="utf-8") as f:
        t2_data = json.load(f)

    # Validate against source ground-truth records
    if os.path.exists(dev_path):
        source_records = load_source_dev_records(dev_path)
        expected_count = len(source_records)
        if expected_count == 0:
            raise ValueError(f"No source records found in '{dev_path}'")

        # Validate Task 1 predictions
        t1_preds = t1_data.get("predictions_221") or t1_data.get("predictions", [])
        if len(t1_preds) != expected_count:
            raise ValueError(f"Task 1 prediction count mismatch: expected {expected_count}, got {len(t1_preds)}")
        for idx, (p, src) in enumerate(zip(t1_preds, source_records)):
            p_idx = p.get("index", p.get("fact_index"))
            if p_idx != idx:
                raise ValueError(f"Task 1 fact index mismatch at position {idx}: expected {idx}, got {p_idx}")
            if p.get("fact", "").strip() != src["fact"]:
                raise ValueError(f"Task 1 fact text mismatch at index {idx}: expected '{src['fact']}', got '{p.get('fact')}'")
            if p.get("gold_label") != src["gold_label"]:
                raise ValueError(f"Task 1 gold label mismatch at index {idx}: expected '{src['gold_label']}', got '{p.get('gold_label')}'")

        # Validate Task 2 predictions
        t2_preds = t2_data.get("predictions", [])
        if len(t2_preds) != expected_count:
            raise ValueError(f"Task 2 prediction count mismatch: expected {expected_count}, got {len(t2_preds)}")
        for idx, (p, src) in enumerate(zip(t2_preds, source_records)):
            p_idx = p.get("fact_index", p.get("index"))
            if p_idx != idx:
                raise ValueError(f"Task 2 fact index mismatch at position {idx}: expected {idx}, got {p_idx}")
            if p.get("fact", "").strip() != src["fact"]:
                raise ValueError(f"Task 2 fact text mismatch at index {idx}: expected '{src['fact']}', got '{p.get('fact')}'")
            if p.get("gold_label") != src["gold_label"]:
                raise ValueError(f"Task 2 gold label mismatch at index {idx}: expected '{src['gold_label']}', got '{p.get('gold_label')}'")

    t2_exp_path = os.path.join(outputs_dir, "task2", "experiments.json")
    if not os.path.exists(t2_exp_path):
        t2_exp_path = "outputs/task2/experiments.json"

    t2_exp_data = {}
    if os.path.exists(t2_exp_path):
        with open(t2_exp_path, "r", encoding="utf-8") as f:
            t2_exp_data = json.load(f)

    benchmarks_data = {}
    if os.path.exists(benchmarks_path):
        with open(benchmarks_path, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)

    return {
        "baselines": b_data,
        "task1": t1_data,
        "task2": t2_data,
        "task2_experiments": t2_exp_data,
        "benchmarks": benchmarks_data
    }


def collate_comparison_records(loaded_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collates structured performance records across all four fact-checking paradigms.
    """
    b_always = loaded_data["baselines"]["always_supported"]
    b_rand = loaded_data["baselines"]["random_guess_seed42"]

    t1_win = loaded_data["task1"]["winning_configuration"]
    t1_m = t1_win["metrics"]
    t1_cv = loaded_data["task1"].get("nested_grouped_5fold_cross_validation", {})
    t1_oof = t1_cv.get("overall_oof_accuracy")

    t2_win = loaded_data["task2"]["winning_config"]
    t2_cv = loaded_data["task2"].get("grouped_cv", {})
    t2_oof = t2_cv.get("pooled_oof_accuracy")

    # Resolve Task 2 measured latency from task2_experiments or benchmarks
    t2_bench = (
        loaded_data.get("task2_experiments", {})
        .get("group_d_computational_profiling", {})
        .get("measured_full_dataset_runtimes", {})
        .get("all_retained", {})
    )
    if not t2_bench:
        t2_bench = loaded_data.get("benchmarks", {}).get("full_dataset_benchmarks", {}).get("all_retained", {})

    t2_runtime_val = t2_bench.get("wall_clock_seconds")
    t2_eval_inst = t2_bench.get("evaluated_instances")
    t2_uniq_pairs = t2_bench.get("unique_pairs")

    t2_runtime_str = f"{t2_runtime_val:.2f}s" if isinstance(t2_runtime_val, (int, float)) else "Not Available"
    if isinstance(t2_runtime_val, (int, float)) and t2_runtime_val > 0:
        if t2_uniq_pairs and t2_eval_inst and t2_uniq_pairs != t2_eval_inst:
            tp_pairs = t2_uniq_pairs / t2_runtime_val
            ms_p = (t2_runtime_val / t2_uniq_pairs) * 1000.0
            t2_throughput_str = f"{tp_pairs:.2f} pairs/s ({ms_p:.1f} ms/p)"
        elif t2_uniq_pairs:
            tp_pairs = t2_uniq_pairs / t2_runtime_val
            ms_p = (t2_runtime_val / t2_uniq_pairs) * 1000.0
            t2_throughput_str = f"{tp_pairs:.2f} pairs/s ({ms_p:.1f} ms/p)"
        else:
            t2_tp_val = t2_bench.get("throughput_pairs_per_sec")
            t2_ms_val = t2_bench.get("ms_per_pair")
            t2_throughput_str = f"{t2_tp_val:.2f} pairs/s ({t2_ms_val:.1f} ms/p)" if isinstance(t2_tp_val, (int, float)) and isinstance(t2_ms_val, (int, float)) else "Not Available"
    else:
        t2_throughput_str = "Not Available"

    # Resolve Task 1 measured latency from benchmarks
    t1_bench = loaded_data["task1"].get("runtime_benchmarks", {})
    t1_runtime_val = t1_bench.get("wall_clock_seconds")
    t1_throughput_val = t1_bench.get("throughput_facts_per_sec")
    t1_ms_val = t1_bench.get("ms_per_fact")

    t1_runtime_str = f"{t1_runtime_val:.2f}s" if isinstance(t1_runtime_val, (int, float)) else "Not Available"
    t1_throughput_str = f"{t1_throughput_val:.1f} facts/s ({t1_ms_val:.1f} ms/f)" if isinstance(t1_throughput_val, (int, float)) and isinstance(t1_ms_val, (int, float)) else "Not Available"

    # Resolve Baseline runtimes dynamically
    b_rand_runtime_val = b_rand.get("runtime_seconds")
    b_rand_tp_val = b_rand.get("throughput_facts_per_sec")
    b_rand_ms_val = b_rand.get("ms_per_fact")
    b_rand_runtime_str = f"{b_rand_runtime_val:.4f}s" if isinstance(b_rand_runtime_val, (int, float)) else "Not Available"
    b_rand_throughput_str = f"{b_rand_tp_val:.0f} facts/s ({b_rand_ms_val:.3f} ms/f)" if isinstance(b_rand_tp_val, (int, float)) and isinstance(b_rand_ms_val, (int, float)) else "Not Available"

    b_always_runtime_val = b_always.get("runtime_seconds")
    b_always_tp_val = b_always.get("throughput_facts_per_sec")
    b_always_ms_val = b_always.get("ms_per_fact")
    b_always_runtime_str = f"{b_always_runtime_val:.4f}s" if isinstance(b_always_runtime_val, (int, float)) else "Not Available"
    b_always_throughput_str = f"{b_always_tp_val:.0f} facts/s ({b_always_ms_val:.3f} ms/f)" if isinstance(b_always_tp_val, (int, float)) and isinstance(b_always_ms_val, (int, float)) else "Not Available"

    # Resolve candidate pair counts dynamically
    b_passages = loaded_data.get("baselines", {}).get("dataset_summary", {}).get("total_associated_passages")
    if b_passages is None:
        b_passages = b_always.get("total_associated_passages")
    b_passages_count = b_passages if isinstance(b_passages, int) else None
    b_evidence_str = f"All Passages ({b_passages_count:,})" if b_passages_count is not None else "All Passages"

    t1_pairs = loaded_data.get("task1", {}).get("meta", {}).get("total_candidate_pairs")
    t1_pairs_count = t1_pairs if isinstance(t1_pairs, int) else None
    t1_evidence_str = f"Max Passages ({t1_pairs_count:,})" if t1_pairs_count is not None else "Max Passages"

    t2_pairs = t2_bench.get("unique_pairs")
    t2_pairs_count = t2_pairs if isinstance(t2_pairs, int) else None
    t2_evidence_str = f"Lexically Pruned Sentences ({t2_pairs_count:,})" if t2_pairs_count is not None else "Lexically Pruned Sentences"

    t1_cm = t1_m["confusion_matrix"]
    t1_tp, t1_fn, t1_fp, t1_tn = t1_cm[0][0], t1_cm[0][1], t1_cm[1][0], t1_cm[1][1]

    t2_cm = t2_win["confusion_matrix"]
    t2_tp, t2_fn, t2_fp, t2_tn = t2_cm[0][0], t2_cm[0][1], t2_cm[1][0], t2_cm[1][1]
    t2_ns_prec = t2_win["precision_ns"] * 100.0
    t2_ns_rec = t2_win["recall_ns"] * 100.0

    t1_opt_th = t1_win.get("optimal_threshold", t1_win.get("threshold"))
    t2_opt_th = t2_win.get("optimal_threshold", t2_win.get("threshold"))
    if not isinstance(t1_opt_th, (int, float)) or not isinstance(t2_opt_th, (int, float)):
        raise ValueError("Verified winning thresholds are required for the comparison table.")
    t1_dec_rule = f"Thresholding on Word Recall (θ* = {t1_opt_th:.2f})"
    t2_dec_rule = f"Optimal Entailment Thresholding (τ* = {t2_opt_th:.2f})"

    t1_strength = "Fast CPU inference; strong surface lexical recall and numerical entity alignment."
    t1_vulnerability = f"Lexical overlap deceived by topic words ({t1_fp} FPs); lacks semantic negation awareness."

    t2_strength = f"Deep semantic reasoning; high NS recall ({t2_ns_rec:.2f}%), NS precision ({t2_ns_prec:.2f}%), and low false positives ({t2_fp} FPs)."
    t2_vulnerability = f"Higher computational latency (CPU); false negatives on multi-hop/numeric claims ({t2_fn} FNs)."

    records = [
        {
            "id": "random_guess",
            "name": "Random Guess (seed=42)",
            "paradigm": "Stochastic Baseline",
            "evidence_units": "None (Null)",
            "candidate_pairs": 0,
            "decision_rule": "Uniform Random Bernoulli (p=0.5)",
            "dev_acc": b_rand["accuracy"] * 100.0,
            "dev_correct": b_rand["correct"],
            "dev_total": b_rand["total"],
            "oof_acc": None,
            "s_prec": b_rand["precision_s"] * 100.0,
            "s_rec": b_rand["recall_s"] * 100.0,
            "s_f1": b_rand["f1_s"],
            "ns_prec": b_rand["precision_ns"] * 100.0,
            "ns_rec": b_rand["recall_ns"] * 100.0,
            "ns_f1": b_rand["f1_ns"],
            "macro_f1": b_rand["macro_f1"],
            "confusion_matrix": b_rand["confusion_matrix"],
            "runtime": b_rand_runtime_str,
            "throughput": b_rand_throughput_str,
            "color": "#64748B",
            "primary_strength": "Zero parameter overhead; provides uninformative stochastic lower bound.",
            "primary_vulnerability": "Completely blind to semantic evidence and fact content."
        },
        {
            "id": "always_supported",
            "name": "Always Supported",
            "paradigm": "Majority Class Baseline",
            "evidence_units": "None (Null)",
            "candidate_pairs": 0,
            "decision_rule": "Constant y = 'S'",
            "dev_acc": b_always["accuracy"] * 100.0,
            "dev_correct": b_always["correct"],
            "dev_total": b_always["total"],
            "oof_acc": None,
            "s_prec": b_always["precision_s"] * 100.0,
            "s_rec": b_always["recall_s"] * 100.0,
            "s_f1": b_always["f1_s"],
            "ns_prec": None,
            "ns_rec": 0.0,
            "ns_f1": None,
            "macro_f1": None,
            "confusion_matrix": b_always["confusion_matrix"],
            "runtime": b_always_runtime_str,
            "throughput": b_always_throughput_str,
            "color": "#94A3B8",
            "primary_strength": "Maximizes Supported recall (100.0%); trivial zero-rule baseline without evaluating passages.",
            "primary_vulnerability": "Total failure on Not Supported class (0.0% precision, recall, F1); blind to evidence."
        },
        {
            "id": "task1_word_overlap",
            "name": "Task 1: Word Overlap (P4 Porter-Stemmed)",
            "paradigm": "Lexical Matching Heuristic",
            "evidence_units": t1_evidence_str,
            "candidate_pairs": t1_pairs_count if t1_pairs_count is not None else 0,
            "decision_rule": t1_dec_rule,
            "dev_acc": t1_m["accuracy"] * 100.0,
            "dev_correct": t1_m["correct"],
            "dev_total": t1_m["total"],
            "oof_acc": t1_oof * 100.0 if t1_oof is not None else None,
            "s_prec": t1_m["precision_s"] * 100.0,
            "s_rec": t1_m["recall_s"] * 100.0,
            "s_f1": t1_m["f1_s"],
            "ns_prec": t1_m["precision_ns"] * 100.0,
            "ns_rec": t1_m["recall_ns"] * 100.0,
            "ns_f1": t1_m["f1_ns"],
            "macro_f1": t1_m["macro_f1"],
            "confusion_matrix": t1_m["confusion_matrix"],
            "runtime": t1_runtime_str,
            "throughput": t1_throughput_str,
            "color": "#2563EB",
            "primary_strength": t1_strength,
            "primary_vulnerability": t1_vulnerability
        },
        {
            "id": "task2_deberta_v3",
            "name": "Task 2: DeBERTa-v3 Entailment",
            "paradigm": "Deep Cross-Attention NLI",
            "evidence_units": t2_evidence_str,
            "candidate_pairs": t2_pairs_count if t2_pairs_count is not None else 0,
            "decision_rule": t2_dec_rule,
            "dev_acc": t2_win["accuracy"] * 100.0,
            "dev_correct": t2_win["correct"],
            "dev_total": t2_win["total"],
            "oof_acc": t2_oof * 100.0 if t2_oof is not None else None,
            "s_prec": t2_win["precision_s"] * 100.0,
            "s_rec": t2_win["recall_s"] * 100.0,
            "s_f1": t2_win["f1_s"],
            "ns_prec": t2_win["precision_ns"] * 100.0,
            "ns_rec": t2_win["recall_ns"] * 100.0,
            "ns_f1": t2_win["f1_ns"],
            "macro_f1": t2_win["macro_f1"],
            "confusion_matrix": t2_win["confusion_matrix"],
            "runtime": t2_runtime_str,
            "throughput": t2_throughput_str,
            "color": "#0F766E",
            "primary_strength": t2_strength,
            "primary_vulnerability": t2_vulnerability
        }
    ]

    return records


def generate_overall_model_comparison_table(comparison_records: List[Dict[str, Any]],
                                            output_path: str = "outputs/comparison/figures/overall_model_comparison_table.png") -> str:
    """
    Renders high-DPI comprehensive comparison table across all 4 coursework paradigms,
    split into two complementary stacked panels with sufficient height and exact width allocation
    to eliminate all text clipping or overflow:
      - Panel A: Classification Verification Metrics (Accuracy, Precision, Recall, F1)
      - Panel B: Evidence Granularity, Inference Load, Decision Rule & Latency Profile
    Also exports standalone figures for each panel for modular inclusion in reports.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    out_dir = os.path.dirname(output_path) or "outputs/comparison/figures"
    os.makedirs(out_dir, exist_ok=True)

    name_display = {
        "random_guess": "Random Guess Baseline",
        "always_supported": "Always Supported Baseline",
        "task1_word_overlap": "Task 1: Word Overlap (P4 Stemmed)",
        "task2_deberta_v3": "Task 2: DeBERTa-v3 Entailment"
    }

    # =========================================================================
    # 1. Master Stacked Two-Panel Figure (16.0 x 11.2 inches)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16.0, 11.2), dpi=200, gridspec_kw={"hspace": 0.42})
    ax1.axis("off")
    ax2.axis("off")

    total_facts = comparison_records[0].get("dev_total") if comparison_records else None
    title_total_facts = f"Development N={total_facts} Facts" if total_facts is not None else "Development Set"

    # Panel A: Classification Verification Metrics
    headers_a = [
        "Model System", "Paradigm", "Dev Acc (%)", "5-Fold OOF (%)",
        "S-Prec (%)", "S-Rec (%)", "S-F1", "NS-Prec (%)", "NS-Rec (%)", "NS-F1", "Macro F1"
    ]
    rows_a = []
    for d in comparison_records:
        disp_name = name_display.get(d["id"], d["name"])
        oof_str = f"{d['oof_acc']:.2f}%" if d.get("oof_acc") is not None else "—"
        s_prec_str = f"{d['s_prec']:.1f}%" if d.get('s_prec') is not None else "N/A"
        s_rec_str = f"{d['s_rec']:.1f}%" if d.get('s_rec') is not None else "N/A"
        s_f1_str = f"{d['s_f1']:.4f}" if d.get('s_f1') is not None else "N/A"
        ns_prec_str = f"{d['ns_prec']:.1f}%" if d.get('ns_prec') is not None else "N/A"
        ns_rec_str = f"{d['ns_rec']:.1f}%" if d.get('ns_rec') is not None else "N/A"
        ns_f1_str = f"{d['ns_f1']:.4f}" if d.get('ns_f1') is not None else "N/A"
        macro_f1_str = f"{d['macro_f1']:.4f}" if d.get('macro_f1') is not None else "N/A"
        rows_a.append([
            disp_name,
            d["paradigm"],
            f"{d['dev_acc']:.2f}%",
            oof_str,
            s_prec_str,
            s_rec_str,
            s_f1_str,
            ns_prec_str,
            ns_rec_str,
            ns_f1_str,
            macro_f1_str
        ])

    col_widths_a = [0.18, 0.13, 0.08, 0.09, 0.07, 0.07, 0.07, 0.08, 0.07, 0.08, 0.08]
    table_a = ax1.table(cellText=rows_a, colLabels=headers_a, colWidths=col_widths_a, loc="center", cellLoc="center")
    table_a.auto_set_font_size(False)
    table_a.set_fontsize(8.5)
    table_a.scale(1.0, 2.0)

    for (r_idx, c_idx), cell in table_a.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx == 3:  # Task 1
            cell.set_facecolor("#DBEAFE")
            if c_idx in [0, 2, 3, 10]:
                cell.set_text_props(color="#1E293B", fontweight="bold")
        elif r_idx == 4:  # Task 2
            cell.set_facecolor("#CCFBF1")
            if c_idx in [0, 2, 3, 10]:
                cell.set_text_props(color="#0F766E", fontweight="bold")
        elif r_idx % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
        if c_idx in [0, 1]:
            cell.set_text_props(ha="left")

    ax1.set_title(f"Panel A: Classification Verification Metrics ({title_total_facts})",
                  fontsize=11.0, fontweight="bold", color="#1E293B", pad=12)

    # Panel B: Evidence Granularity, Inference Load & Hardware Latency
    headers_b = [
        "Model System", "Paradigm", "Evidence Granularity", "Candidate Pairs",
        "Decision Rule", "Recorded Runtime\n(CPU)", "Recorded\nThroughput"
    ]
    rows_b = []
    for d in comparison_records:
        disp_name = name_display.get(d["id"], d["name"])
        pairs_str = f"{d['candidate_pairs']:,}" if isinstance(d['candidate_pairs'], int) and d['candidate_pairs'] > 0 else ("0" if d['candidate_pairs'] == 0 else "—")
        ev_wrapped = textwrap.fill(d["evidence_units"], width=22)
        dec_wrapped = textwrap.fill(d["decision_rule"], width=22)
        rows_b.append([
            disp_name,
            d["paradigm"],
            ev_wrapped,
            pairs_str,
            dec_wrapped,
            d["runtime"],
            d["throughput"]
        ])

    col_widths_b = [0.20, 0.13, 0.15, 0.08, 0.18, 0.12, 0.14]
    table_b = ax2.table(cellText=rows_b, colLabels=headers_b, colWidths=col_widths_b, loc="center", cellLoc="center")
    table_b.auto_set_font_size(False)
    table_b.set_fontsize(8.5)
    table_b.scale(1.0, 2.1)

    for (r_idx, c_idx), cell in table_b.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx == 3:  # Task 1
            cell.set_facecolor("#DBEAFE")
            if c_idx in [0, 3, 5, 6]:
                cell.set_text_props(color="#1E293B", fontweight="bold")
        elif r_idx == 4:  # Task 2
            cell.set_facecolor("#CCFBF1")
            if c_idx in [0, 3, 5, 6]:
                cell.set_text_props(color="#0F766E", fontweight="bold")
        elif r_idx % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
        if c_idx in [0, 1, 4]:
            cell.set_text_props(ha="left")

    ax2.set_title("Panel B: Evidence Granularity, Decision Rule & Computational Latency Profile (CPU)",
                  fontsize=11.0, fontweight="bold", color="#1E293B", pad=12)

    plt.suptitle("Comprehensive Comparative Evaluation Across Fact-Checking Paradigms",
                 fontsize=12.5, fontweight="bold", color="#1E293B", y=0.985)
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.02, right=0.98, hspace=0.42)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close()

    # =========================================================================
    # 2. Standalone Panel A Figure: overall_model_comparison_metrics.png
    # =========================================================================
    fig_a, ax_a = plt.subplots(figsize=(14.5, 4.8), dpi=200)
    ax_a.axis("off")
    t_a = ax_a.table(cellText=rows_a, colLabels=headers_a, colWidths=col_widths_a, loc="center", cellLoc="center")
    t_a.auto_set_font_size(False)
    t_a.set_fontsize(8.5)
    t_a.scale(1.0, 2.0)
    for (r_idx, c_idx), cell in t_a.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx == 3:
            cell.set_facecolor("#DBEAFE")
            if c_idx in [0, 2, 3, 10]:
                cell.set_text_props(color="#1E293B", fontweight="bold")
        elif r_idx == 4:
            cell.set_facecolor("#CCFBF1")
            if c_idx in [0, 2, 3, 10]:
                cell.set_text_props(color="#0F766E", fontweight="bold")
        elif r_idx % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
        if c_idx in [0, 1]:
            cell.set_text_props(ha="left")
    title_standalone = f"Development N={total_facts}" if total_facts is not None else "Development Set"
    ax_a.set_title(f"Comparative Evaluation: Classification Verification Metrics ({title_standalone})",
                   fontsize=11.5, fontweight="bold", color="#1E293B", pad=14)
    path_a = os.path.join(out_dir, "overall_model_comparison_metrics.png")
    plt.tight_layout()
    plt.savefig(path_a, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close()

    # =========================================================================
    # 3. Standalone Panel B Figure: overall_model_comparison_profile.png
    # =========================================================================
    fig_b, ax_b = plt.subplots(figsize=(16.0, 5.2), dpi=200)
    ax_b.axis("off")
    t_b = ax_b.table(cellText=rows_b, colLabels=headers_b, colWidths=col_widths_b, loc="center", cellLoc="center")
    t_b.auto_set_font_size(False)
    t_b.set_fontsize(8.5)
    t_b.scale(1.0, 2.1)
    for (r_idx, c_idx), cell in t_b.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx == 3:
            cell.set_facecolor("#DBEAFE")
            if c_idx in [0, 3, 5, 6]:
                cell.set_text_props(color="#1E293B", fontweight="bold")
        elif r_idx == 4:
            cell.set_facecolor("#CCFBF1")
            if c_idx in [0, 3, 5, 6]:
                cell.set_text_props(color="#0F766E", fontweight="bold")
        elif r_idx % 2 == 1:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("#FFFFFF")
        if c_idx in [0, 1, 4]:
            cell.set_text_props(ha="left")
    ax_b.set_title("Comparative Evaluation: Evidence Granularity & Computational Latency Profile (CPU)",
                   fontsize=11.5, fontweight="bold", color="#1E293B", pad=14)
    path_b = os.path.join(out_dir, "overall_model_comparison_profile.png")
    plt.tight_layout()
    plt.savefig(path_b, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close()

    return output_path


def generate_overall_accuracy_progression(comparison_records: List[Dict[str, Any]],
                                          output_path: str = "outputs/comparison/figures/overall_accuracy_progression.png") -> str:
    """
    Renders high-DPI bar chart showing development accuracy progression across all coursework paradigms.
    Includes benchmark target lines (75% for Task 1, 83% for Task 2, 50% Chance level) with clean annotations.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)

    names = [
        d["name"]
        .replace(" (seed=42)", "")
        .replace(" (P4 Porter-Stemmed Number-Aware Word Recall)", "")
        .replace(" (P4 Porter-Stemmed)", "")
        .replace(" (Lemmatized)", "")
        for d in comparison_records
    ]
    accs = [d["dev_acc"] for d in comparison_records]
    colors = ["#64748B", "#64748B", "#2563EB", "#0F766E"]

    x = np.arange(len(names))
    width = 0.52

    bars = ax.bar(x, accs, width=width, color=colors, edgecolor="#1E293B", linewidth=0.8, zorder=3)

    t1_rec = next((d for d in comparison_records if d["id"] == "task1_word_overlap"), None)
    t2_rec = next((d for d in comparison_records if d["id"] == "task2_deberta_v3"), None)
    t1_achieved_str = f" | Achieved: {t1_rec['dev_acc']:.2f}%" if t1_rec else ""
    t2_achieved_str = f" | Achieved: {t2_rec['dev_acc']:.2f}%" if t2_rec else ""

    # Reference target lines
    ax.axhline(75.0, color="#DC2626", linestyle="--", linewidth=1.5, zorder=2,
               label=f"Task 1 Target (≥ 75.0%{t1_achieved_str})")
    ax.axhline(83.0, color="#DC2626", linestyle=":", linewidth=1.5, zorder=2,
               label=f"Task 2 Target (≥ 83.0%{t2_achieved_str})")
    ax.axhline(50.0, color="#94A3B8", linestyle="-.", linewidth=1.2, zorder=2,
               label="Chance Level (50.0%)")

    # Clean, dynamic annotations on top of bars
    for i, bar in enumerate(bars):
        h = bar.get_height()
        d = comparison_records[i]
        ax.annotate(f"{h:.2f}%\n({d['dev_correct']}/{d['dev_total']})",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.0, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Development Accuracy (%)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.set_title("Systematic Accuracy Progression Across Coursework Paradigms",
                 fontsize=12.0, fontweight="bold", color="#1E293B", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.2, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y", zorder=1)
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close()
    return output_path


def save_overall_comparison_json(comparison_records: List[Dict[str, Any]],
                                 t1_data: Dict[str, Any],
                                 t2_data: Dict[str, Any],
                                 output_path: str = "outputs/comparison/overall_results.json") -> str:
    """
    Saves outputs/comparison/overall_results.json containing comprehensive metrics,
    per-system breakdown, agreement analysis, and qualitative trade-offs.
    All summary counts and synthesis statements are dynamically derived from predictions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    t1_preds = t1_data.get("predictions_221") or t1_data.get("predictions", [])
    t2_preds = t2_data.get("predictions", [])
    total_facts = len(t1_preds) if t1_preds else len(t2_preds)

    s_count = sum(1 for p in (t1_preds or t2_preds) if p.get("gold_label") == "S")
    ns_count = sum(1 for p in (t1_preds or t2_preds) if p.get("gold_label") in ("NS", "IR"))

    agreement_count = sum(1 for p1, p2 in zip(t1_preds, t2_preds) if p1.get("predicted_label") == p2.get("predicted_label"))
    t1_err_indices = set(i for i, p in enumerate(t1_preds) if not p.get("is_correct"))
    t2_err_indices = set(i for i, p in enumerate(t2_preds) if not p.get("is_correct"))

    common_errors = sorted(list(t1_err_indices.intersection(t2_err_indices)))
    t1_only_errors = sorted(list(t1_err_indices - t2_err_indices))
    t2_only_errors = sorted(list(t2_err_indices - t1_err_indices))
    both_correct_count = total_facts - len(t1_err_indices | t2_err_indices)
    oracle_union_correct = total_facts - len(common_errors)

    denom = float(total_facts) if total_facts > 0 else 1.0
    agree_pct = round(agreement_count / denom * 100.0, 2)
    both_correct_pct = round(both_correct_count / denom * 100.0, 2)
    oracle_acc_pct = round(oracle_union_correct / denom * 100.0, 2)

    t1_tp = sum(1 for p in t1_preds if p.get("gold_label") == "S" and p.get("predicted_label") == "S")
    t1_fn = sum(1 for p in t1_preds if p.get("gold_label") == "S" and p.get("predicted_label") != "S")
    t1_fp = sum(1 for p in t1_preds if p.get("gold_label") != "S" and p.get("predicted_label") == "S")
    t1_tn = sum(1 for p in t1_preds if p.get("gold_label") != "S" and p.get("predicted_label") != "S")

    t2_tp = sum(1 for p in t2_preds if p.get("gold_label") == "S" and p.get("predicted_label") == "S")
    t2_fn = sum(1 for p in t2_preds if p.get("gold_label") == "S" and p.get("predicted_label") != "S")
    t2_fp = sum(1 for p in t2_preds if p.get("gold_label") != "S" and p.get("predicted_label") == "S")
    t2_tn = sum(1 for p in t2_preds if p.get("gold_label") != "S" and p.get("predicted_label") != "S")

    t2_ns_prec = (t2_tn / (t2_tn + t2_fn) * 100.0) if (t2_tn + t2_fn) > 0 else 0.0
    t2_ns_rec = (t2_tn / (t2_tn + t2_fp) * 100.0) if (t2_tn + t2_fp) > 0 else 0.0

    synthesis_insight = (
        "Task 1 (Word Overlap) and Task 2 (DeBERTa-v3) exhibit complementary error profiles: "
        f"Task 2 achieves exceptional sensitivity on Not Supported claims ({t2_ns_rec:.2f}% NS recall, {t2_ns_prec:.2f}% NS precision, "
        f"only {t2_fp} FPs vs Task 1's {t1_fp} FPs), successfully rejecting lexical distractors. Conversely, Task 1 maintains higher "
        f"recall on multi-hop and numeric Supported claims ({t1_fn} FNs vs Task 2's {t2_fn} FNs). "
        f"Only {len(common_errors)} facts represent common failures across both paradigms, and an oracle union yields {oracle_acc_pct:.2f}% accuracy "
        "as a diagnostic theoretical upper bound (not a trained hybrid model), highlighting strong potential for hybrid lexical-neural fact-checking architectures."
    )

    master_payload = {
        "metadata": {
            "evaluation_dataset": "data/dev_labeled_ChatGPT.jsonl",
            "source_passages": "data/passages_bm25_ChatGPT_humfacts.jsonl",
            "total_facts_evaluated": total_facts,
            "class_balance": {"supported": s_count, "not_supported": ns_count},
            "description": "Master cross-paradigm evaluation and comparative audit for CM3604 Phase 8."
        },
        "model_comparison_records": comparison_records,
        "comparative_error_analysis": {
            "total_facts": total_facts,
            "task1_error_count": len(t1_err_indices),
            "task2_error_count": len(t2_err_indices),
            "prediction_agreement_count": agreement_count,
            "prediction_agreement_percentage": agree_pct,
            "both_correct_count": both_correct_count,
            "both_correct_percentage": both_correct_pct,
            "common_failure_count": len(common_errors),
            "common_failure_indices": common_errors,
            "task1_only_errors_count": len(t1_only_errors),
            "task1_only_error_indices": t1_only_errors,
            "task2_only_errors_count": len(t2_only_errors),
            "task2_only_error_indices": t2_only_errors,
            "oracle_union_correct_count": oracle_union_correct,
            "oracle_union_accuracy_percentage": oracle_acc_pct,
            "oracle_union_diagnostic_note": "Theoretical diagnostic upper bound assuming an oracle selector choosing the correct prediction if either model is correct; not a trained hybrid model.",
            "synthesis_insight": synthesis_insight
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_payload, f, indent=2)

    return output_path


def run_model_comparison(outputs_dir: str = "outputs/comparison",
                         figures_dir: str = "outputs/comparison/figures",
                         cache_dir: str = "outputs/task2/cache") -> None:
    """
    Main entry point for Phase 8: Comprehensive Model Comparison and Evidence Integrity Audit.
    Loads experimental evidence, generates Phase 8 deliverables, and verifies cross-paradigm consistency.
    """
    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)

    # 1. Load system results
    print("\n[Phase 8] Loading and cross-validating system outputs...")
    loaded = load_all_system_results(outputs_dir="outputs", cache_dir=cache_dir)
    print("  Loaded baseline results, Task 1 experiments, Task 2 results, and benchmarks.")

    # 2. Collate comparison records
    print("\n[Phase 8] Collating comparative records across 4 fact-checking paradigms...")
    comparison_records = collate_comparison_records(loaded)
    for r in comparison_records:
        oof_display = f"{r['oof_acc']:<6.2f}%" if r.get('oof_acc') is not None else "N/A   "
        mf1_display = f"{r['macro_f1']:<6.4f}" if r.get('macro_f1') is not None else "N/A   "
        print(f"  {r['name']:<35} | Dev Acc: {r['dev_acc']:<6.2f}% | OOF CV: {oof_display} | Macro-F1: {mf1_display} | Latency: {r['runtime']}")

    # 3. Save outputs/comparison/overall_results.json
    print("\n[Phase 8] Saving comparison JSON...")
    os.makedirs(outputs_dir, exist_ok=True)
    p_comp = save_overall_comparison_json(comparison_records, loaded["task1"], loaded["task2"],
                                          os.path.join(outputs_dir, "overall_results.json"))
    print(f"  Saved {p_comp}")

    # 4. Export outputs/comparison/comparison.csv
    csv_path = os.path.join(outputs_dir, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "evidence_units", "dev_accuracy", "oof_cv_accuracy", "s_f1", "ns_f1", "macro_f1", "latency", "throughput"])
        for r in comparison_records:
            oof_str = f"{r['oof_acc']:.2f}%" if r.get('oof_acc') is not None else "N/A"
            s_f1_str = f"{r['s_f1']:.4f}" if r.get('s_f1') is not None else "N/A"
            ns_f1_str = f"{r['ns_f1']:.4f}" if r.get('ns_f1') is not None else "N/A"
            macro_str = f"{r['macro_f1']:.4f}" if r.get('macro_f1') is not None else "N/A"
            writer.writerow([r["name"], r["evidence_units"], f"{r['dev_acc']:.2f}%", oof_str, s_f1_str, ns_f1_str, macro_str, r["runtime"], r["throughput"]])
    print(f"  Exported comparison spreadsheet to {csv_path}")

    # 5. Generate Figures
    os.makedirs(figures_dir, exist_ok=True)
    print("\n[Phase 8] Generating high-DPI comparison table...")
    fig_table = os.path.join(figures_dir, "overall_model_comparison_table.png")
    generate_overall_model_comparison_table(comparison_records, fig_table)
    print(f"  Saved {fig_table}")

    print("\n[Phase 8] Generating high-DPI accuracy progression chart...")
    fig_prog = os.path.join(figures_dir, "overall_accuracy_progression.png")
    generate_overall_accuracy_progression(comparison_records, fig_prog)
    print(f"  Saved {fig_prog}")

    print("\n" + "=" * 80)
    print("Phase 8 Model Comparison successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    run_model_comparison()

