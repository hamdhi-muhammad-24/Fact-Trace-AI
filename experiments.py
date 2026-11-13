# experiments.py
"""
CM3604 Deep Learning Coursework — Experimental Evaluation Suite
Consolidated Benchmark Orchestration, Parameter Optimizations, Cross-Validation & Visualizations

Modules Covered:
  - Phase 2: Majority & Stochastic Baselines (Always Supported & Monte Carlo Random Guess)
  - Phase 3: Task 1 Heuristic Word Overlap Optimization (120-config grid, P0-P4 ablation, nested grouped CV)
  - Phase 5: Task 2 DeBERTa-v3 Textual Entailment Optimization (evidence formulations, pruning sweeps, threshold tuning)

All experiments run independently of the official submission file `factcheck.py`.
Classifiers and data loaders are imported directly from `factcheck.py`.
"""

import os
import sys
import json
import re
import csv
import math
import copy
import gc
import time
import textwrap
import hashlib
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import torch
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Safe optional Matplotlib import with headless Agg backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Import official production classes and math primitives from factcheck.py
from factcheck import (
    FactExample,
    FactChecker,
    RandomGuessFactChecker,
    AlwaysEntailedFactChecker,
    WordRecallThresholdFactChecker,
    EntailmentFactChecker,
    EntailmentModel,
    load_passages_data,
    load_dev_examples,
    calculate_confusion_matrix,
    calculate_metrics,
    preprocess_tokens,
    compute_similarity_metric,
    compute_claim_passages_score,
    get_english_stopwords,
    get_wordnet_lemmatizer,
    split_into_sentences,
    build_evidence_units,
    get_tokens_for_pruning,
    compute_overlap_recall,
    prune_candidate_units
)


# ==============================================================================
# PHASE 2: BASELINE EXPERIMENTS & METRICS EVALUATION
# ==============================================================================

def evaluate_fact_checker_standalone(examples: List[FactExample], fact_checker: FactChecker) -> Tuple[Dict[str, Any], List[str]]:
    """
    Evaluates any FactChecker instance on development examples using official binary mapping:
    'S' remains 'S', 'NS' remains 'NS', and 'IR' is mapped to 'NS'.
    Returns (metrics_dict, predictions_list).
    """
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    predictions = []
    t_start = time.perf_counter()
    for ex in examples:
        raw_pred = fact_checker.predict(ex.fact, ex.passages)
        if raw_pred not in ("S", "NS"):
            raise ValueError(f"Classifier returned invalid prediction '{raw_pred}'. Must be 'S' or 'NS'.")
        predictions.append(raw_pred)
    elapsed = time.perf_counter() - t_start

    confusion_mat = calculate_confusion_matrix(gold_labels, predictions)
    metrics = calculate_metrics(confusion_mat)
    n_ex = len(examples)
    metrics["runtime_seconds"] = round(elapsed, 4)
    metrics["throughput_facts_per_sec"] = round(n_ex / elapsed, 1) if elapsed > 0 else 0.0
    metrics["ms_per_fact"] = round((elapsed / n_ex) * 1000.0, 2) if n_ex > 0 else 0.0
    return metrics, predictions


def run_monte_carlo_random_baseline(examples: List[FactExample], num_iterations: int = 1000) -> Dict[str, Any]:
    """
    Executes multiple independent seeds of RandomGuessFactChecker to measure true empirical distribution,
    standard deviation, and 95% confidence intervals across chance-level predictions.
    """
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    accs = []
    prec_s_list, rec_s_list, f1_s_list = [], [], []
    prec_ns_list, rec_ns_list, f1_ns_list = [], [], []
    bal_accs = []
    macro_f1s = []

    for seed in range(num_iterations):
        checker = RandomGuessFactChecker(seed=seed)
        preds = [checker.predict(ex.fact, ex.passages) for ex in examples]
        cm = calculate_confusion_matrix(gold_labels, preds)
        m = calculate_metrics(cm)

        accs.append(m["accuracy"])
        bal_accs.append(m["balanced_accuracy"])
        if m["precision_s"] is not None:
            prec_s_list.append(m["precision_s"])
        rec_s_list.append(m["recall_s"])
        if m["f1_s"] is not None:
            f1_s_list.append(m["f1_s"])

        if m["precision_ns"] is not None:
            prec_ns_list.append(m["precision_ns"])
        rec_ns_list.append(m["recall_ns"])
        if m["f1_ns"] is not None:
            f1_ns_list.append(m["f1_ns"])

        if m["macro_f1"] is not None:
            macro_f1s.append(m["macro_f1"])

    def summarize_series(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ci_95_lower": float(np.percentile(arr, 2.5)),
            "ci_95_upper": float(np.percentile(arr, 97.5))
        }

    return {
        "num_iterations": num_iterations,
        "accuracy": summarize_series(accs),
        "balanced_accuracy": summarize_series(bal_accs),
        "precision_s": summarize_series(prec_s_list),
        "recall_s": summarize_series(rec_s_list),
        "f1_s": summarize_series(f1_s_list),
        "precision_ns": summarize_series(prec_ns_list),
        "recall_ns": summarize_series(rec_ns_list),
        "f1_ns": summarize_series(f1_ns_list),
        "macro_f1": summarize_series(macro_f1s)
    }


def generate_baseline_performance_table(always_m: Dict[str, Any], random_single_m: Dict[str, Any],
                                        random_mc: Dict[str, Any],
                                        output_path: str = "outputs/baselines/figures/baselines_performance_table.png") -> str:
    """
    Renders high-DPI image table comparing Always Supported against Random Guess single seed and Monte Carlo distributions.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    def fmt_pct(val: Optional[float], denom: Optional[str] = None) -> str:
        if val is None:
            return "Undefined (0/0)"
        pct = f"{val * 100:.2f}%"
        return f"{pct} ({denom})" if denom else pct

    def fmt_mc(stat: Dict[str, float]) -> str:
        return f"{stat['mean']*100:.2f}% ± {stat['std']*100:.2f}%\n[{stat['ci_95_lower']*100:.2f}%, {stat['ci_95_upper']*100:.2f}%]"

    always_total = always_m.get('total') if always_m.get('total') is not None else (always_m['tp'] + always_m['fn'] + always_m['fp'] + always_m['tn'])
    random_total = random_single_m.get('total') if random_single_m.get('total') is not None else (random_single_m['tp'] + random_single_m['fn'] + random_single_m['fp'] + random_single_m['tn'])
    mc_runs = random_mc.get("num_iterations")
    always_gold_s = always_m['tp'] + always_m['fn']
    always_gold_ns = always_m['tn'] + always_m['fp']

    table_data = [
        [
            "Overall Accuracy",
            f"{always_m['correct']}/{always_total} = {always_m['accuracy']*100:.2f}%",
            f"{random_single_m['correct']}/{random_total} = {random_single_m['accuracy']*100:.2f}%",
            fmt_mc(random_mc["accuracy"]),
            "Task 1: ≥ 75.0%\nTask 2: ≥ 83.0%"
        ],
        [
            "Balanced Accuracy",
            f"{always_m['balanced_accuracy']*100:.2f}%",
            f"{random_single_m['balanced_accuracy']*100:.2f}%",
            fmt_mc(random_mc["balanced_accuracy"]),
            "Unbiased Metric"
        ],
        [
            "Precision (Supported - S)",
            fmt_pct(always_m['precision_s'], f"{always_m['tp']}/{always_total}"),
            fmt_pct(random_single_m['precision_s'], f"{random_single_m['tp']}/{random_single_m['tp']+random_single_m['fp']}"),
            fmt_mc(random_mc["precision_s"]),
            "Target: High Precision"
        ],
        [
            "Recall (Supported - S)",
            fmt_pct(always_m['recall_s'], f"{always_m['tp']}/{always_gold_s}"),
            fmt_pct(random_single_m['recall_s'], f"{random_single_m['tp']}/{always_gold_s}"),
            fmt_mc(random_mc["recall_s"]),
            "Target: High Recall"
        ],
        [
            "F1-Score (Supported - S)",
            f"{always_m['f1_s']:.4f} ({always_m['f1_s']*100:.2f}%)",
            f"{random_single_m['f1_s']:.4f} ({random_single_m['f1_s']*100:.2f}%)",
            f"{random_mc['f1_s']['mean']:.4f} ± {random_mc['f1_s']['std']:.4f}",
            "Harmonic Mean"
        ],
        [
            "Precision (Not Supported - NS)",
            fmt_pct(always_m['precision_ns']),
            fmt_pct(random_single_m['precision_ns'], f"{random_single_m['tn']}/{random_single_m['tn']+random_single_m['fn']}"),
            fmt_mc(random_mc["precision_ns"]),
            "Target: High Precision"
        ],
        [
            "Recall (Not Supported - NS)",
            fmt_pct(always_m['recall_ns'], f"{always_m['tn']}/{always_gold_ns}"),
            fmt_pct(random_single_m['recall_ns'], f"{random_single_m['tn']}/{always_gold_ns}"),
            fmt_mc(random_mc["recall_ns"]),
            "Target: High Recall"
        ],
        [
            "F1-Score (Not Supported - NS)",
            "Undefined (N/A)" if always_m.get('f1_ns') is None else f"{always_m['f1_ns']:.4f}",
            f"{random_single_m['f1_ns']:.4f} ({random_single_m['f1_ns']*100:.2f}%)",
            f"{random_mc['f1_ns']['mean']:.4f} ± {random_mc['f1_ns']['std']:.4f}",
            "Harmonic Mean"
        ],
        [
            "Macro-F1 (Unweighted Average)",
            "Undefined (N/A)" if always_m.get('macro_f1') is None else f"{always_m['macro_f1']:.4f}",
            f"{random_single_m['macro_f1']:.4f} ({random_single_m['macro_f1']*100:.2f}%)",
            f"{random_mc['macro_f1']['mean']:.4f} ± {random_mc['macro_f1']['std']:.4f}",
            "Balanced Quality"
        ],
        [
            "Evaluated Instances (N)",
            f"{always_total} facts" if always_total is not None else "N/A",
            f"{random_total} facts (Seed 42)" if random_total is not None else "N/A",
            (f"{always_total} facts × {mc_runs:,} runs" if always_total is not None and mc_runs is not None
             else (f"{always_total} facts × MC" if always_total is not None else "N/A")),
            f"{always_total} facts (All preserved)" if always_total is not None else "N/A"
        ]
    ]

    mc_col_label = f"Random Guess\n({mc_runs:,}-Run Monte Carlo)" if mc_runs is not None else "Random Guess\n(Monte Carlo)"
    col_labels = [
        "Evaluation Metric / Dimension",
        "Always Supported\n(Majority Baseline)",
        "Random Guess\n(Seed 42 Single Run)",
        mc_col_label,
        "Coursework\nTarget / Role"
    ]
    col_widths = [0.24, 0.18, 0.19, 0.22, 0.17]

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
            if col == 4:
                cell.set_text_props(weight="semibold", color="#DC2626")
            elif col == 1:
                cell.set_text_props(weight="semibold")

    title_n = f", N = {always_total}" if always_total is not None else ""
    plt.title(f"Baseline Benchmark Performance Comparison (FActScore Development Set{title_n})",
              fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_baseline_confusion_matrices(always_m: Dict[str, Any], random_single_m: Dict[str, Any],
                                        output_path: str = "outputs/baselines/figures/baselines_confusion_matrix.png") -> str:
    """
    Renders side-by-side standardized two-tone confusion matrices for Always Supported and Random Guess baselines.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=200)

    categories = ["Supported (S)", "Not Supported (NS)"]
    mask = np.array([[1, 0], [0, 1]])
    cmap = matplotlib.colors.ListedColormap(["#DBEAFE", "#1D4ED8"])

    # Panel A: Always Supported
    cm_always = np.array(always_m["confusion_matrix"])
    always_tot = always_m.get("total", int(np.sum(cm_always)))
    ax1.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title(f"Always Supported (Majority Baseline)\nAccuracy: {always_m['accuracy']*100:.2f}% ({always_m['correct']}/{always_tot})",
                  fontsize=10.5, fontweight="bold", color="#1E293B", pad=12)
    ax1.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("True Label (Ground Truth)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(categories, fontsize=9.0, fontweight="bold", color="#1E293B")
    ax1.set_yticklabels(categories, fontsize=9.0, fontweight="bold", color="#1E293B")

    tags = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            val = cm_always[i, j]
            pct = (val / always_tot * 100.0) if always_tot > 0 else 0.0
            tag = tags[i][j]
            color = "white" if (i == j) else "#1E293B"
            ax1.text(j, i, f"{val}\n({pct:.1f}%)\n[{tag}]", ha="center", va="center",
                     color=color, fontsize=10.5, fontweight="bold")

    # Panel B: Random Guess (Seed 42)
    cm_random = np.array(random_single_m["confusion_matrix"])
    rand_tot = random_single_m.get("total", int(np.sum(cm_random)))
    ax2.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title(f"Random Guess (Seed 42 Baseline)\nAccuracy: {random_single_m['accuracy']*100:.2f}% ({random_single_m['correct']}/{rand_tot})",
                  fontsize=10.5, fontweight="bold", color="#1E293B", pad=12)
    ax2.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("True Label (Ground Truth)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(categories, fontsize=9.0, fontweight="bold", color="#1E293B")
    ax2.set_yticklabels(categories, fontsize=9.0, fontweight="bold", color="#1E293B")

    for i in range(2):
        for j in range(2):
            val = cm_random[i, j]
            pct = (val / rand_tot * 100.0) if rand_tot > 0 else 0.0
            tag = tags[i][j]
            color = "white" if (i == j) else "#1E293B"
            ax2.text(j, i, f"{val}\n({pct:.1f}%)\n[{tag}]", ha="center", va="center",
                     color=color, fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close()
    return output_path


def run_baselines(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
                  dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
                  figures_dir: str = "outputs/baselines/figures",
                  outputs_dir: str = "outputs/baselines") -> None:
    """
    Main entry point for Phase 2 Evaluation Framework and Baseline Benchmarks.
    Evaluates Always Supported and Random Guess (single run and 1,000-run Monte Carlo).
    """
    print("[Phase 2 Baselines] Loading datasets...")
    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)
    print(f"[Phase 2 Baselines] Loaded {len(examples)} development fact examples.")

    # 1. Evaluate Always Supported Baseline
    always_checker = AlwaysEntailedFactChecker()
    always_metrics, always_preds = evaluate_fact_checker_standalone(examples, always_checker)
    print(f"[Phase 2 Baselines] Always Supported Accuracy: {always_metrics['correct']}/{always_metrics['total']} = {always_metrics['accuracy']:.4f} ({always_metrics['accuracy']*100:.2f}%)")

    # 2. Evaluate Random Guess Baseline (Single Seed 42 Run)
    random_single_checker = RandomGuessFactChecker(seed=42)
    random_single_metrics, random_single_preds = evaluate_fact_checker_standalone(examples, random_single_checker)
    print(f"[Phase 2 Baselines] Random Guess (Seed 42) Accuracy: {random_single_metrics['correct']}/{random_single_metrics['total']} = {random_single_metrics['accuracy']:.4f} ({random_single_metrics['accuracy']*100:.2f}%)")

    # 3. Evaluate Random Guess Baseline (1,000-Run Monte Carlo)
    print("[Phase 2 Baselines] Running 1,000 Monte Carlo iterations for Random Guess...")
    random_mc = run_monte_carlo_random_baseline(examples, num_iterations=1000)
    acc_mc = random_mc["accuracy"]
    print(f"[Phase 2 Baselines] Monte Carlo (1,000 runs): Mean={acc_mc['mean']*100:.2f}%, Std={acc_mc['std']*100:.2f}%, Min={acc_mc['min']*100:.2f}%, Max={acc_mc['max']*100:.2f}%")

    # 4. Save JSON results
    os.makedirs(outputs_dir, exist_ok=True)
    baseline_record = {
        "dataset_summary": {
            "total_evaluated_instances": len(examples),
            "total_associated_passages": sum(len(ex.passages) for ex in examples),
            "gold_supported_count": sum(1 for ex in examples if ex.label == "S"),
            "gold_not_supported_count": sum(1 for ex in examples if ex.label in ("NS", "IR"))
        },
        "always_supported": always_metrics,
        "random_guess_seed42": {
            **random_single_metrics,
            "seed": 42
        },
        "random_guess_monte_carlo": random_mc
    }
    json_path = os.path.join(outputs_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(baseline_record, f, indent=2)
    print(f"[Phase 2 Baselines] Saved raw benchmark results to '{json_path}'.")

    # 5. Generate high-DPI figures
    os.makedirs(figures_dir, exist_ok=True)
    table_fig = os.path.join(figures_dir, "baselines_performance_table.png")
    cm_fig = os.path.join(figures_dir, "baselines_confusion_matrix.png")

    generate_baseline_performance_table(always_metrics, random_single_metrics, random_mc, table_fig)
    generate_baseline_confusion_matrices(always_metrics, random_single_metrics, cm_fig)
    print(f"[Phase 2 Baselines] Generated figures:\n  - {table_fig}\n  - {cm_fig}")

    print("[Phase 2 Baselines] Execution completed successfully.")


# ==============================================================================
# PHASE 3: TASK 1 EXPERIMENTS & THRESHOLD OPTIMIZATION
# ==============================================================================

def evaluate_config_thresholds_fast(sub_scores: List[float], sub_gold: List[str], metric_name: str) -> Tuple[float, Dict[str, Any], Tuple[float, float], List[Dict[str, Any]], int]:
    """
    Fast evaluation of all candidate thresholds for a given score vector and gold label subset.
    """
    N = len(sub_gold)
    unique_s = sorted(set(sub_scores))
    midpoints = [(unique_s[i] + unique_s[i+1]) / 2.0 for i in range(len(unique_s)-1)]

    if metric_name in ("jaccard", "dice", "tf_cosine"):
        fine_grid = [x / 10000.0 for x in range(0, 3000, 5)]
    else:
        fine_grid = [x / 100.0 for x in range(0, 101)]

    candidates = sorted(set(unique_s + midpoints + fine_grid))
    min_s, max_s = min(sub_scores), max(sub_scores)
    candidates = [c for c in candidates if (min_s - 0.01) <= c <= (max_s + 0.01)]
    if not candidates:
        candidates = [0.5]

    best_acc = -1.0
    plateau = []
    curve_records = []

    for th in candidates:
        tp = sum(1 for s, g in zip(sub_scores, sub_gold) if s >= th and g == "S")
        fn = sum(1 for s, g in zip(sub_scores, sub_gold) if s < th and g == "S")
        fp = sum(1 for s, g in zip(sub_scores, sub_gold) if s >= th and g == "NS")
        tn = sum(1 for s, g in zip(sub_scores, sub_gold) if s < th and g == "NS")

        acc = (tp + tn) / N
        rec_s = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rec_ns = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        bal = (rec_s + rec_ns) / 2.0

        prec_s = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_s = (2.0 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) > 0 else 0.0

        prec_ns = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_ns = (2.0 * prec_ns * rec_ns / (prec_ns + rec_ns)) if (prec_ns + rec_ns) > 0 else 0.0
        macro = (f1_s + f1_ns) / 2.0

        record = {
            "threshold": th,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "macro_f1": macro,
            "f1_s": f1_s,
            "f1_ns": f1_ns,
            "precision_s": prec_s,
            "recall_s": rec_s,
            "precision_ns": prec_ns,
            "recall_ns": rec_ns,
            "confusion_matrix": [[tp, fn], [fp, tn]],
            "correct": tp + tn,
            "total": N
        }
        curve_records.append(record)

        if acc > best_acc:
            best_acc = acc
            plateau = [record]
        elif acc == best_acc:
            plateau.append(record)

    max_bal = max(r["balanced_accuracy"] for r in plateau)
    plateau = [r for r in plateau if r["balanced_accuracy"] == max_bal]

    max_macro = max(r["macro_f1"] for r in plateau)
    plateau = [r for r in plateau if r["macro_f1"] == max_macro]

    max_f1_s = max(r["f1_s"] for r in plateau)
    plateau = [r for r in plateau if r["f1_s"] == max_f1_s]

    plateau_thresholds = sorted(r["threshold"] for r in plateau)
    median_th = plateau_thresholds[len(plateau_thresholds) // 2]
    best_record = next(r for r in plateau if r["threshold"] == median_th)
    plateau_bounds = (plateau_thresholds[0], plateau_thresholds[-1])

    return best_record["threshold"], best_record, plateau_bounds, curve_records, len(candidates)


def multi_criterion_ranking_key(cfg_result: Dict[str, Any]) -> Tuple[float, float, float, float, float, int]:
    """
    Deterministic multi-criterion ranking key for selecting the winning classifier configuration.
    """
    plat = cfg_result["plateau_bounds"]
    plat_width = plat[1] - plat[0]
    return (
        -cfg_result["accuracy"],
        -cfg_result["balanced_accuracy"],
        -cfg_result["macro_f1"],
        -cfg_result["f1_s"],
        -plat_width,
        cfg_result["cfg"].get("complexity", 999)
    )


def run_nested_grouped_cross_validation(examples: List[FactExample], raw_bios: List[dict],
                                        all_scores_dict: Dict[str, List[float]],
                                        grid_configs: List[Dict[str, Any]],
                                        gold_labels: List[str]) -> Dict[str, Any]:
    """
    Executes genuinely leakage-controlled nested Grouped 5-Fold Cross-Validation.
    """
    idx_to_topic = {}
    cur_idx = 0
    for b in raw_bios:
        topic = b.get("topic")
        for s in b.get("annotations", []):
            for f in s.get("human-atomic-facts", []):
                idx_to_topic[cur_idx] = topic
                cur_idx += 1

    topic_counts = Counter(idx_to_topic.values())
    folds = [[] for _ in range(5)]
    fold_counts = [0] * 5
    folds[0].extend(["Marianne McAndrew", "Mike Trivisonno"])
    fold_counts[0] += topic_counts["Marianne McAndrew"] + topic_counts["Mike Trivisonno"]

    rem = [t for t in topic_counts if t not in ("Marianne McAndrew", "Mike Trivisonno")]
    rem.sort(key=lambda t: -topic_counts[t])
    for t in rem:
        m = fold_counts.index(min(fold_counts))
        folds[m].append(t)
        fold_counts[m] += topic_counts[t]

    oof_preds = [None] * len(examples)
    fold_records = []
    total_inner_evaluations = 0

    for f_idx in range(5):
        val_topics = set(folds[f_idx])
        val_indices = [i for i in range(len(examples)) if idx_to_topic[i] in val_topics]
        train_indices = [i for i in range(len(examples)) if idx_to_topic[i] not in val_topics]

        train_candidates = []
        for cfg in grid_configs:
            key = f"{cfg['prep']}_{cfg['metric']}_{cfg['agg']}_{cfg['penalty_rule']}_{cfg['penalty_factor']}"
            th, m, plat, _, num_evals = evaluate_config_thresholds_fast(
                [all_scores_dict[key][i] for i in train_indices],
                [gold_labels[i] for i in train_indices],
                cfg["metric"]
            )
            total_inner_evaluations += num_evals
            train_candidates.append({
                "cfg": cfg,
                "key": key,
                "threshold": th,
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "f1_s": m["f1_s"],
                "plateau_bounds": plat,
                "train_metrics": m
            })

        train_candidates.sort(key=multi_criterion_ranking_key)
        fold_winner = train_candidates[0]
        w_key = fold_winner["key"]
        w_cfg = fold_winner["cfg"]
        w_th = fold_winner["threshold"]
        w_train_m = fold_winner["train_metrics"]

        val_scores = [all_scores_dict[w_key][i] for i in val_indices]
        val_gold = [gold_labels[i] for i in val_indices]
        val_preds = ["S" if s >= w_th else "NS" for s in val_scores]

        for idx, p in zip(val_indices, val_preds):
            oof_preds[idx] = p

        cm_val = calculate_confusion_matrix(val_gold, val_preds)
        m_val = calculate_metrics(cm_val)

        fold_records.append({
            "fold": f_idx,
            "val_topics": sorted(list(val_topics)),
            "val_examples_count": len(val_indices),
            "train_examples_count": len(train_indices),
            "selected_config": {
                "prep": w_cfg["prep"],
                "metric": w_cfg["metric"],
                "agg": w_cfg["agg"],
                "penalty_rule": w_cfg["penalty_rule"],
                "penalty_factor": w_cfg["penalty_factor"]
            },
            "chosen_threshold": w_th,
            "train_accuracy": w_train_m["accuracy"],
            "train_balanced_accuracy": w_train_m["balanced_accuracy"],
            "train_macro_f1": w_train_m["macro_f1"],
            "val_accuracy": m_val["accuracy"],
            "val_balanced_accuracy": m_val["balanced_accuracy"],
            "val_precision_s": m_val["precision_s"],
            "val_recall_s": m_val["recall_s"],
            "val_f1_s": m_val["f1_s"],
            "val_precision_ns": m_val["precision_ns"],
            "val_recall_ns": m_val["recall_ns"],
            "val_f1_ns": m_val["f1_ns"],
            "val_macro_f1": m_val["macro_f1"],
            "val_confusion_matrix": cm_val
        })

    cm_oof = calculate_confusion_matrix(gold_labels, oof_preds)
    m_oof = calculate_metrics(cm_oof)
    val_accs = [f["val_accuracy"] for f in fold_records]

    return {
        "folds": fold_records,
        "mean_val_accuracy": float(np.mean(val_accs)),
        "std_val_accuracy": float(np.std(val_accs)),
        "min_val_accuracy": float(np.min(val_accs)),
        "max_val_accuracy": float(np.max(val_accs)),
        "overall_oof_accuracy": m_oof["accuracy"],
        "overall_oof_balanced_accuracy": m_oof["balanced_accuracy"],
        "overall_oof_macro_f1": m_oof["macro_f1"],
        "overall_oof_metrics": m_oof,
        "oof_predictions": oof_preds,
        "total_inner_threshold_evaluations": total_inner_evaluations
    }


def generate_task1_preprocessing_comparison(prep_results: Dict[str, Dict[str, Any]],
                                            baseline_always_acc: float,
                                            output_path: str = "outputs/task1/figures/task1_preprocessing_comparison.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    modes = ["P0\nRaw", "P1\nStopwords", "P2\nStemming", "P3\nLemmatization", "P4\nNumber-Aware"]
    keys = ["P0", "P1", "P2", "P3", "P4"]

    accs = [prep_results[k]["accuracy"] * 100 for k in keys]
    bal_accs = [prep_results[k]["balanced_accuracy"] * 100 for k in keys]
    macro_f1s = [prep_results[k]["macro_f1"] * 100 for k in keys]

    x = np.arange(len(modes))
    width = 0.26

    rects1 = ax.bar(x - width, accs, width, label="Overall Accuracy", color="#2563EB", edgecolor="black")
    rects2 = ax.bar(x, bal_accs, width, label="Balanced Accuracy", color="#0F766E", edgecolor="black")
    rects3 = ax.bar(x + width, macro_f1s, width, label="Macro-F1", color="#64748B", edgecolor="black")

    ax.axhline(75.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 75.0%)")
    ax.axhline(baseline_always_acc * 100, color="#94A3B8", linestyle=":", linewidth=1.4,
               label=f"Always Supported Baseline ({baseline_always_acc*100:.2f}%)")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 1 Preprocessing Ablation Comparison (P0 - P4 under Max Word Recall)",
                 fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_metric_comparison(metric_results: Dict[str, Dict[str, Any]],
                                     baseline_always_acc: float,
                                     output_path: str = "outputs/task1/figures/task1_metric_comparison.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    metric_names = ["Word Recall\n(Asymmetric)", "Jaccard\n(Symmetric)", "Dice\n(Symmetric)", "TF Cosine\n(Vector Angle)"]
    keys = ["recall", "jaccard", "dice", "tf_cosine"]

    accs = [metric_results[k]["accuracy"] * 100 for k in keys]
    bal_accs = [metric_results[k]["balanced_accuracy"] * 100 for k in keys]
    ths = [metric_results[k]["threshold"] for k in keys]

    x = np.arange(len(metric_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, accs, width, label="Overall Accuracy", color="#2563EB", edgecolor="black")
    rects2 = ax.bar(x + width/2, bal_accs, width, label="Balanced Accuracy", color="#0F766E", edgecolor="black")

    ax.axhline(75.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 75.0%)")
    ax.axhline(baseline_always_acc * 100, color="#94A3B8", linestyle=":", linewidth=1.4,
               label=f"Always Supported Baseline ({baseline_always_acc*100:.2f}%)")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 1 Lexical Similarity Metric Comparison (P4 Number-Aware, Max Aggregation)",
                 fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    for i, rect in enumerate(rects1):
        h = rect.get_height()
        th_val = ths[i]
        th_str = f"θ*={th_val:.2f}" if th_val >= 0.1 else f"θ*={th_val:.4f}"
        ax.annotate(f"{h:.1f}%\n({th_str})", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_threshold_curve(curve_data: List[Dict[str, Any]],
                                   optimal_th: float,
                                   plateau_bounds: Tuple[float, float],
                                   opt_record: Dict[str, Any],
                                   winning_title: str,
                                   output_path: str = "outputs/task1/figures/task1_threshold_curve.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    plot_pts = [p for p in curve_data if 0.10 <= p["threshold"] <= 0.95]
    if not plot_pts:
        plot_pts = curve_data

    ths = [p["threshold"] for p in plot_pts]
    accs = [p["accuracy"] * 100 for p in plot_pts]
    bal_accs = [p["balanced_accuracy"] * 100 for p in plot_pts]
    f1_s = [p["f1_s"] * 100 if p["f1_s"] is not None else 0.0 for p in plot_pts]
    f1_ns = [p["f1_ns"] * 100 if p["f1_ns"] is not None else 0.0 for p in plot_pts]

    ax.plot(ths, accs, label="Overall Accuracy", color="#2563EB", linewidth=2.2)
    ax.plot(ths, bal_accs, label="Balanced Accuracy", color="#0F766E", linewidth=1.8, linestyle="-.")
    ax.plot(ths, f1_s, label="F1-Score (Supported - S)", color="#1E293B", linewidth=1.5, linestyle=":")
    ax.plot(ths, f1_ns, label="F1-Score (Not Supported - NS)", color="#64748B", linewidth=1.5, linestyle="--")

    ax.axhline(75.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 75.0%)")

    p_min, p_max = plateau_bounds
    ax.axvspan(p_min, p_max, color="#DBEAFE", alpha=0.55, label=f"Peak Accuracy Plateau [{p_min:.2f} - {p_max:.2f}]")
    ax.axvline(optimal_th, color="#1E293B", linestyle="-", linewidth=1.8, label=f"Selected θ* = {optimal_th:.2f}")

    correct_cnt = opt_record["correct"]
    total_cnt = opt_record["total"]
    peak_acc_pct = opt_record["accuracy"] * 100

    ax.plot(optimal_th, peak_acc_pct, marker="o", markersize=7, color="#1E293B")
    ax.annotate(f"Peak Accuracy: {peak_acc_pct:.2f}%\n({correct_cnt}/{total_cnt} correct)",
                xy=(optimal_th, peak_acc_pct),
                xytext=(optimal_th - 0.22, peak_acc_pct - 8),
                arrowprops=dict(facecolor="#1E293B", shrink=0.08, width=1, headwidth=5),
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#1E293B"))

    ax.set_xlabel("Decision Threshold (θ)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title(f"Task 1 Decision Threshold Optimization Curve ({winning_title})",
                 fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlim(0.10, 0.95)
    ax.set_ylim(35, 100)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_score_separability(scores_s: List[float], scores_ns: List[float],
                                      optimal_th: float,
                                      output_path: str = "outputs/task1/figures/task1_score_separability.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=200)

    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(scores_s, bins=bins, alpha=0.75, color="#2563EB", edgecolor="#1E293B",
            label=f"Supported Claims (S, N={len(scores_s)})", zorder=3)
    ax.hist(scores_ns, bins=bins, alpha=0.65, color="#64748B", edgecolor="#1E293B",
            label=f"Not Supported Claims (NS, N={len(scores_ns)})", zorder=2)

    ax.axvline(optimal_th, color="#DC2626", linestyle="--", linewidth=1.8, zorder=4,
               label=f"Optimal Decision Boundary (θ* = {optimal_th:.2f})")

    ax.set_xlabel("Porter-Stemmed Word Overlap Recall Score", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Claim Count (Frequency)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 1: Overlap Score Separability Between Supported and Not Supported Claims",
                 fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y", zorder=1)
    ax.legend(loc="upper center", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_validation_summary(cv_results: Dict[str, Any],
                                      output_path: str = "outputs/task1/figures/task1_validation_summary.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    folds = cv_results["folds"]
    total_n = sum(f["val_examples_count"] for f in folds)
    labels = [f"Fold {f['fold']}\n(N={f['val_examples_count']})" for f in folds] + [f"Overall OOF\n(N={total_n})"]

    train_accs = [f["train_accuracy"] * 100 for f in folds] + [float(np.mean([f["train_accuracy"] * 100 for f in folds]))]
    val_accs = [f["val_accuracy"] * 100 for f in folds] + [cv_results["overall_oof_accuracy"] * 100]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, train_accs, width, label="Training Portion Accuracy", color="#0F766E", edgecolor="black")
    rects2 = ax.bar(x + width/2, val_accs, width, label="Out-of-Fold Validation Accuracy", color="#2563EB", edgecolor="black")

    ax.axhline(75.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 75.0%)")

    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title(f"Task 1 Nested Grouped 5-Fold Cross-Validation Summary (Mean OOF: {cv_results['mean_val_accuracy']*100:.2f}% ± {cv_results['std_val_accuracy']*100:.2f}%)",
                 fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_ylim(60, 100)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_final_confusion_matrix(cm_data: List[List[int]],
                                          optimal_th: float,
                                          winning_title: str,
                                          output_path: str = "outputs/task1/figures/task1_final_confusion_matrix.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)

    cm = np.array(cm_data)
    total = int(cm.sum())
    correct = int(cm[0, 0] + cm[1, 1])
    accuracy = correct / total if total > 0 else 0.0
    categories = ["Supported (S)", "Not Supported (NS)"]

    mask = np.array([[1, 0], [0, 1]])
    cmap = matplotlib.colors.ListedColormap(["#DBEAFE", "#1D4ED8"])

    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"Task 1 Final Confusion Matrix ({winning_title} @ θ*={optimal_th:.2f})\nAccuracy: {correct}/{total} = {accuracy*100:.2f}%",
                 fontsize=10.5, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylabel("True Label (Ground Truth)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_yticklabels(categories, fontsize=9.5, fontweight="bold", color="#1E293B")

    tags = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100.0 if total > 0 else 0.0
            tag = tags[i][j]
            color = "white" if (i == j) else "#1E293B"
            ax.text(j, i, f"{val}\n({pct:.1f}%)\n[{tag}]", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close()
    return output_path


def generate_task1_final_results_table(baseline_always: Dict[str, Any],
                                       baseline_rand: Dict[str, Any],
                                       prep_records: Dict[str, Dict[str, Any]],
                                       final_m: Dict[str, Any],
                                       cv_oof_acc: float,
                                       winning_title: str,
                                       output_path: str = "outputs/task1/figures/task1_final_results_table.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    def fmt(v: Optional[float]) -> str:
        return f"{v*100:.2f}%" if v is not None else "Undefined"

    def fmt_f1(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "Undefined"

    table_data = [
        ["Overall Accuracy", fmt(baseline_always.get("accuracy")), fmt(baseline_rand.get("accuracy")), fmt(prep_records["P0"].get("accuracy")), fmt(prep_records["P2"].get("accuracy")), fmt(final_m.get("accuracy")), "≥ 75.0%"],
        ["Balanced Accuracy", fmt(baseline_always.get("balanced_accuracy")), fmt(baseline_rand.get("balanced_accuracy")), fmt(prep_records["P0"].get("balanced_accuracy")), fmt(prep_records["P2"].get("balanced_accuracy")), fmt(final_m.get("balanced_accuracy")), "Unbiased Metric"],
        ["Precision (Supported - S)", fmt(baseline_always.get("precision_s")), fmt(baseline_rand.get("precision_s")), fmt(prep_records["P0"].get("precision_s")), fmt(prep_records["P2"].get("precision_s")), fmt(final_m.get("precision_s")), "High Precision"],
        ["Recall (Supported - S)", fmt(baseline_always.get("recall_s")), fmt(baseline_rand.get("recall_s")), fmt(prep_records["P0"].get("recall_s")), fmt(prep_records["P2"].get("recall_s")), fmt(final_m.get("recall_s")), "High Recall"],
        ["F1-Score (Supported - S)", fmt_f1(baseline_always.get("f1_s")), fmt_f1(baseline_rand.get("f1_s")), fmt_f1(prep_records["P0"].get("f1_s")), fmt_f1(prep_records["P2"].get("f1_s")), fmt_f1(final_m.get("f1_s")), "Harmonic Mean"],
        ["Precision (Not Supported - NS)", fmt(baseline_always.get("precision_ns")), fmt(baseline_rand.get("precision_ns")), fmt(prep_records["P0"].get("precision_ns")), fmt(prep_records["P2"].get("precision_ns")), fmt(final_m.get("precision_ns")), "High Precision"],
        ["Recall (Not Supported - NS)", fmt(baseline_always.get("recall_ns")), fmt(baseline_rand.get("recall_ns")), fmt(prep_records["P0"].get("recall_ns")), fmt(prep_records["P2"].get("recall_ns")), fmt(final_m.get("recall_ns")), "High Recall"],
        ["F1-Score (Not Supported - NS)", fmt_f1(baseline_always.get("f1_ns")), fmt_f1(baseline_rand.get("f1_ns")), fmt_f1(prep_records["P0"].get("f1_ns")), fmt_f1(prep_records["P2"].get("f1_ns")), fmt_f1(final_m.get("f1_ns")), "Harmonic Mean"],
        ["Macro-F1 (Unweighted)", fmt_f1(baseline_always.get("macro_f1")), fmt_f1(baseline_rand.get("macro_f1")), fmt_f1(prep_records["P0"].get("macro_f1")), fmt_f1(prep_records["P2"].get("macro_f1")), fmt_f1(final_m.get("macro_f1")), "Balanced Quality"],
        ["Grouped 5-Fold OOF Accuracy", "N/A", "N/A", "N/A", "N/A", f"{cv_oof_acc*100:.2f}%", "Generalization"]
    ]

    col_labels = [
        "Evaluation Metric",
        "Always Supp.\n(Baseline)",
        "Random Guess\n(Seed 42)",
        "P0 Raw\n(Ablation)",
        "P2 Stemmed\n(Ablation)",
        f"Task 1 Final\n({winning_title})",
        "Coursework\nTarget"
    ]
    col_widths = [0.24, 0.12, 0.12, 0.12, 0.12, 0.16, 0.12]

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.52)

    for (row, col), cell in table.get_celld().items():
        if col == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
            if col == 5:
                cell.set_facecolor("#DBEAFE")
                cell.set_text_props(weight="bold", color="#1E293B")
            elif col == 6:
                cell.set_text_props(weight="semibold", color="#DC2626")

    plt.title(f"Task 1 Final Benchmark Comparison (FActScore Development Set, N = {final_m['total']})",
              fontsize=11, fontweight="bold", color="#1E293B", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def run_task1(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
              dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
              figures_dir: str = "outputs/task1/figures",
              outputs_dir: str = "outputs/task1") -> None:
    """
    Main entry point for Phase 3 Task 1 Word Overlap Fact-Checking experiment suite.
    """
    start_time = time.time()
    print("[Phase 3 Task 1] Loading datasets...")
    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    print(f"[Phase 3 Task 1] Loaded {len(examples)} development fact examples across {len(raw_bios)} topics.")

    # 1. Baseline benchmarks loading / dynamic recalculation
    baseline_path = os.path.join(os.path.dirname(outputs_dir), "baselines", "results.json")
    if not os.path.exists(baseline_path):
        baseline_path = "outputs/baselines/results.json"

    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        b_always = b_data.get("always_supported", {})
        b_rand = b_data.get("random_guess_seed42", {})
    else:
        print("[Phase 3 Task 1] baseline results not found; calculating baseline metrics dynamically...")
        always_checker = AlwaysEntailedFactChecker()
        always_preds = [always_checker.predict(ex.fact, ex.passages) for ex in examples]
        cm_always = calculate_confusion_matrix(gold_labels, always_preds)
        b_always = calculate_metrics(cm_always)

        rand_checker = RandomGuessFactChecker(seed=42)
        rand_preds = [rand_checker.predict(ex.fact, ex.passages) for ex in examples]
        cm_rand = calculate_confusion_matrix(gold_labels, rand_preds)
        b_rand = calculate_metrics(cm_rand)

    baseline_always_acc = float(b_always.get("accuracy", sum(1 for y in gold_labels if y == "S") / len(examples)))

    # 2. Precomputing tokens
    stemmer = PorterStemmer()
    lemmatizer = get_wordnet_lemmatizer()
    stops = get_english_stopwords()

    print("[Phase 3 Task 1] Precomputing claim and evidence token representations...")
    t_pre0 = time.time()
    modes = ["P0", "P1", "P2", "P3"]
    claim_toks = {m: [preprocess_tokens(ex.fact, m, stemmer, lemmatizer, stops) for ex in examples] for m in modes}
    claim_nums = [set(re.findall(r'\b\d+\b', ex.fact)) for ex in examples]

    pass_toks = {m: [[preprocess_tokens(p.get("text", ""), m, stemmer, lemmatizer, stops) for p in ex.passages] for ex in examples] for m in modes}
    pass_nums = [[set(re.findall(r'\b\d+\b', p.get("text", ""))) for p in ex.passages] for ex in examples]

    pass_concat_toks = {m: [preprocess_tokens(" ".join(p.get("text", "") for p in ex.passages), m, stemmer, lemmatizer, stops) for ex in examples] for m in modes}
    pass_concat_nums = [set(re.findall(r'\b\d+\b', " ".join(p.get("text", "") for p in ex.passages))) for ex in examples]
    print(f"[Phase 3 Task 1] Token precomputation completed in {time.time() - t_pre0:.2f} seconds.")

    def get_scores_for_config(prep: str, metric: str, agg: str, num_penalty_rule: Optional[str] = None, penalty_factor: float = 0.0) -> List[float]:
        base_prep = "P2" if prep == "P4" else prep
        scores = []
        for i in range(len(examples)):
            c_t = claim_toks[base_prep][i]
            c_n = claim_nums[i]
            if agg == "concat":
                p_t = pass_concat_toks[base_prep][i]
                s = compute_similarity_metric(c_t, p_t, metric)
                if prep == "P4" and c_n:
                    p_n = pass_concat_nums[i]
                    if num_penalty_rule == "disjoint" and c_n.isdisjoint(p_n):
                        s *= penalty_factor
                    elif num_penalty_rule == "subset" and not c_n.issubset(p_n):
                        s *= penalty_factor
                scores.append(s)
            else:
                p_ts = pass_toks[base_prep][i]
                p_ns = pass_nums[i]
                per_p = []
                for p_t, p_n in zip(p_ts, p_ns):
                    s = compute_similarity_metric(c_t, p_t, metric)
                    if prep == "P4" and c_n:
                        if num_penalty_rule == "disjoint" and c_n.isdisjoint(p_n):
                            s *= penalty_factor
                        elif num_penalty_rule == "subset" and not c_n.issubset(p_n):
                            s *= penalty_factor
                    per_p.append(s)
                if not per_p:
                    scores.append(0.0)
                elif agg == "max":
                    scores.append(max(per_p))
                elif agg == "mean":
                    scores.append(sum(per_p) / len(per_p))
        return scores

    # 3. Construct 120 configurations
    grid_configs = []
    comp_idx = 0
    for p in ["P0", "P1", "P2", "P3"]:
        for m in ["recall", "jaccard", "dice", "tf_cosine"]:
            for a in ["max", "concat", "mean"]:
                grid_configs.append({
                    "prep": p, "metric": m, "agg": a,
                    "penalty_rule": None, "penalty_factor": 0.0,
                    "complexity": comp_idx
                })
                comp_idx += 1

    for pen in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for m in ["recall", "jaccard", "dice", "tf_cosine"]:
            for a in ["max", "concat", "mean"]:
                grid_configs.append({
                    "prep": "P4", "metric": m, "agg": a,
                    "penalty_rule": "disjoint", "penalty_factor": pen,
                    "complexity": comp_idx
                })
                comp_idx += 1

    for m in ["recall", "jaccard", "dice", "tf_cosine"]:
        for a in ["max", "concat", "mean"]:
            grid_configs.append({
                "prep": "P4", "metric": m, "agg": a,
                "penalty_rule": "subset", "penalty_factor": 0.0,
                "complexity": comp_idx
            })
            comp_idx += 1

    print(f"\n--- [Grid Search] Precomputing score matrices for all {len(grid_configs)} configurations ---")
    t_score0 = time.time()
    all_scores_dict = {}
    for cfg in grid_configs:
        key = f"{cfg['prep']}_{cfg['metric']}_{cfg['agg']}_{cfg['penalty_rule']}_{cfg['penalty_factor']}"
        all_scores_dict[key] = get_scores_for_config(
            cfg["prep"], cfg["metric"], cfg["agg"], cfg["penalty_rule"], cfg["penalty_factor"]
        )
    print(f"[Phase 3 Task 1] All {len(grid_configs)} score vectors computed in {time.time() - t_score0:.2f} seconds.")

    # 4. Evaluate grid search
    t_grid0 = time.time()
    full_dev_results = []
    total_evals_dev = 0
    for cfg in grid_configs:
        key = f"{cfg['prep']}_{cfg['metric']}_{cfg['agg']}_{cfg['penalty_rule']}_{cfg['penalty_factor']}"
        th, m, plat, curve, num_cands = evaluate_config_thresholds_fast(all_scores_dict[key], gold_labels, cfg["metric"])
        total_evals_dev += num_cands
        full_dev_results.append({
            "cfg": cfg,
            "key": key,
            "threshold": th,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "f1_s": m["f1_s"],
            "plateau_bounds": plat,
            "train_metrics": m,
            "curve_records": curve,
            "num_candidates": num_cands
        })

    full_dev_results.sort(key=multi_criterion_ranking_key)
    winning_record = full_dev_results[0]
    w_cfg = winning_record["cfg"]
    winning_prep = w_cfg["prep"]
    winning_metric = w_cfg["metric"]
    winning_agg = w_cfg["agg"]
    winning_penalty = w_cfg["penalty_factor"]
    winning_rule = w_cfg["penalty_rule"]
    winning_key = winning_record["key"]
    opt_th = winning_record["threshold"]
    final_m = winning_record["train_metrics"]
    min_plat_th, max_plat_th = winning_record["plateau_bounds"]
    full_curve = winning_record["curve_records"]
    winning_scores = all_scores_dict[winning_key]
    final_preds = ["S" if s >= opt_th else "NS" for s in winning_scores]
    winning_title = f"{winning_prep} {winning_metric.upper()} {winning_agg.upper()}"

    print(f"[Phase 3 Task 1] Winning configuration: {winning_title} @ θ* = {opt_th:.2f}")
    print(f"  Accuracy: {final_m['correct']}/{final_m['total']} = {final_m['accuracy']*100:.2f}%")

    res_by_key = {r["key"]: r for r in full_dev_results}

    prep_results = {}
    for m in ["P0", "P1", "P2", "P3", "P4"]:
        k = f"{m}_recall_max_disjoint_0.0" if m == "P4" else f"{m}_recall_max_None_0.0"
        r = res_by_key[k]
        prep_results[m] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    metric_results = {}
    for m in ["recall", "jaccard", "dice", "tf_cosine"]:
        k = f"P4_{m}_max_disjoint_0.0"
        r = res_by_key[k]
        metric_results[m] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    agg_results = {}
    for a in ["max", "mean", "concat"]:
        k = f"P4_recall_{a}_disjoint_0.0"
        r = res_by_key[k]
        agg_results[a] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    penalty_results = {}
    for pen in [0.0, 0.25, 0.50, 0.75, 1.0]:
        k = f"P4_recall_max_disjoint_{pen}"
        r = res_by_key[k]
        penalty_results[str(pen)] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    r_disjoint = res_by_key["P4_recall_max_disjoint_0.0"]
    r_subset = res_by_key["P4_recall_max_subset_0.0"]
    rule_results = {
        "disjoint": {**r_disjoint["train_metrics"], "threshold": r_disjoint["threshold"], "plateau_bounds": r_disjoint["plateau_bounds"]},
        "subset": {**r_subset["train_metrics"], "threshold": r_subset["threshold"], "plateau_bounds": r_subset["plateau_bounds"]}
    }

    # 5. Numerical consistency audit
    p2_key = "P2_recall_max_None_0.0"
    p2_scores = all_scores_dict[p2_key]
    p2_th = prep_results["P2"]["threshold"]
    p2_preds = ["S" if s >= p2_th else "NS" for s in p2_scores]

    claims_with_nums = []
    fp_to_tn_transitions = []
    partial_match_cases = []

    for i, ex in enumerate(examples):
        c_n = claim_nums[i]
        if c_n:
            all_p_nums = set().union(*pass_nums[i]) if pass_nums[i] else set()
            is_disjoint = c_n.isdisjoint(all_p_nums)
            is_subset = c_n.issubset(all_p_nums)

            audit_item = {
                "index": i,
                "fact": ex.fact,
                "claim_numbers": sorted(list(c_n)),
                "passage_numbers_union": sorted(list(all_p_nums)),
                "gold_label": gold_labels[i],
                "p2_raw_score": p2_scores[i],
                "p2_predicted": p2_preds[i],
                "p4_score": winning_scores[i],
                "p4_predicted": final_preds[i],
                "is_disjoint": bool(is_disjoint),
                "is_subset": bool(is_subset)
            }
            claims_with_nums.append(audit_item)

            if p2_preds[i] == "S" and final_preds[i] == "NS" and gold_labels[i] == "NS":
                fp_to_tn_transitions.append(audit_item)

            if (not is_disjoint) and (not is_subset):
                partial_match_cases.append(audit_item)

    # 6. Leak-Free Nested Grouped 5-Fold Cross-Validation
    cv_results = run_nested_grouped_cross_validation(examples, raw_bios, all_scores_dict, grid_configs, gold_labels)

    # 7. Predictions list
    idx_to_topic = {}
    cur_idx = 0
    for b in raw_bios:
        topic = b.get("topic")
        for s in b.get("annotations", []):
            for f in s.get("human-atomic-facts", []):
                idx_to_topic[cur_idx] = topic
                cur_idx += 1

    prediction_records = []
    for i, ex in enumerate(examples):
        prediction_records.append({
            "index": i,
            "topic": idx_to_topic.get(i, ""),
            "fact": ex.fact,
            "gold_label": gold_labels[i],
            "predicted_label": final_preds[i],
            "score": winning_scores[i],
            "is_correct": bool(final_preds[i] == gold_labels[i]),
            "contains_numbers": bool(len(claim_nums[i]) > 0),
            "claim_numbers": sorted(list(claim_nums[i]))
        })

    # Benchmark runtime
    bench_checker = WordRecallThresholdFactChecker(threshold=opt_th)
    t_bench0 = time.perf_counter()
    bench_preds = [bench_checker.predict(ex.fact, ex.passages) for ex in examples]
    bench_elapsed = time.perf_counter() - t_bench0
    n_ex = len(examples)
    bench_tp = n_ex / bench_elapsed if bench_elapsed > 0 else 0.0
    bench_ms = (bench_elapsed / n_ex) * 1000.0 if n_ex > 0 else 0.0

    elapsed_time = time.time() - start_time

    # 8. Save structured JSON experiment records
    os.makedirs(outputs_dir, exist_ok=True)
    task1_record = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_evaluated_instances": len(examples),
            "total_candidate_pairs": sum(len(ex.passages) for ex in examples),
            "total_biography_topics": len(raw_bios),
            "search_space_configurations": len(grid_configs),
            "dev_threshold_evaluations": total_evals_dev,
            "nested_cv_inner_threshold_evaluations": cv_results["total_inner_threshold_evaluations"],
            "total_threshold_evaluations": total_evals_dev + cv_results["total_inner_threshold_evaluations"],
            "runtime_seconds": round(elapsed_time, 2)
        },
        "runtime_benchmarks": {
            "wall_clock_seconds": round(bench_elapsed, 4),
            "ms_per_fact": round(bench_ms, 2),
            "throughput_facts_per_sec": round(bench_tp, 1),
            "provenance": f"Live measured evaluation of winning WordRecallThresholdFactChecker on development set ({len(examples)} facts, CPU)"
        },
        "winning_configuration": {
            "preprocessing": winning_prep,
            "metric": winning_metric,
            "aggregation": winning_agg,
            "number_rule": winning_rule,
            "number_penalty": winning_penalty,
            "optimal_threshold": opt_th,
            "plateau_range": [min_plat_th, max_plat_th],
            "plateau_width": round(max_plat_th - min_plat_th, 4),
            "selection_rationale": f"Ranked #1 out of {len(grid_configs)} configurations via deterministic multi-criterion ranking.",
            "metrics": final_m
        },
        "top_10_grid_configurations": [
            {
                "rank": idx + 1,
                "config": r["cfg"],
                "optimal_threshold": r["threshold"],
                "plateau_bounds": r["plateau_bounds"],
                "accuracy": r["accuracy"],
                "balanced_accuracy": r["balanced_accuracy"],
                "macro_f1": r["macro_f1"],
                "f1_s": r["f1_s"],
                "metrics": r["train_metrics"]
            }
            for idx, r in enumerate(full_dev_results[:10])
        ],
        "preprocessing_ablation": prep_results,
        "similarity_metric_comparison": metric_results,
        "evidence_aggregation_comparison": agg_results,
        "numerical_penalty_ablation": penalty_results,
        "numerical_rule_comparison": rule_results,
        "numerical_consistency_audit": {
            "total_claims_with_numbers": len(claims_with_nums),
            "fp_to_tn_transitions_count": len(fp_to_tn_transitions),
            "fp_to_tn_transitions": fp_to_tn_transitions,
            "partial_match_cases_count": len(partial_match_cases),
            "partial_match_cases": partial_match_cases,
            "all_numerical_claims": claims_with_nums
        },
        "nested_grouped_5fold_cross_validation": {
            "folds": cv_results["folds"],
            "mean_val_accuracy": cv_results["mean_val_accuracy"],
            "std_val_accuracy": cv_results["std_val_accuracy"],
            "min_val_accuracy": cv_results["min_val_accuracy"],
            "max_val_accuracy": cv_results["max_val_accuracy"],
            "overall_oof_accuracy": cv_results["overall_oof_accuracy"],
            "overall_oof_balanced_accuracy": cv_results["overall_oof_balanced_accuracy"],
            "overall_oof_macro_f1": cv_results["overall_oof_macro_f1"],
            "overall_oof_metrics": cv_results["overall_oof_metrics"]
        },
        "threshold_curve": full_curve,
        "predictions_221": prediction_records
    }

    json_path = os.path.join(outputs_dir, "experiments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(task1_record, f, indent=2)

    results_path = os.path.join(outputs_dir, "results.json")
    results_record = {
        "winning_configuration": task1_record["winning_configuration"],
        "nested_grouped_5fold_cross_validation": task1_record["nested_grouped_5fold_cross_validation"],
        "predictions_221": prediction_records
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_record, f, indent=2)

    # 9. CSV Exports
    csv_preds_path = os.path.join(outputs_dir, "predictions.csv")
    with open(csv_preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "topic", "fact", "gold_label", "predicted_label", "score", "is_correct"])
        for r in prediction_records:
            writer.writerow([r["index"], r["topic"], r["fact"], r["gold_label"], r["predicted_label"], f"{r['score']:.4f}", r["is_correct"]])

    csv_comp_path = os.path.join(outputs_dir, "comparisons.csv")
    with open(csv_comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["preprocessing", "accuracy", "balanced_accuracy", "macro_f1", "threshold"])
        for k, v in prep_results.items():
            writer.writerow([k, f"{v['accuracy']*100:.2f}%", f"{v['balanced_accuracy']*100:.2f}%", f"{v['macro_f1']:.4f}", f"{v['threshold']:.2f}"])

    # 10. Generate high-DPI figures
    os.makedirs(figures_dir, exist_ok=True)
    fig1 = os.path.join(figures_dir, "task1_preprocessing_comparison.png")
    fig2 = os.path.join(figures_dir, "task1_metric_comparison.png")
    fig3 = os.path.join(figures_dir, "task1_threshold_curve.png")
    fig4 = os.path.join(figures_dir, "task1_validation_summary.png")
    fig5 = os.path.join(figures_dir, "task1_final_confusion_matrix.png")
    fig6 = os.path.join(figures_dir, "task1_final_results_table.png")
    fig7 = os.path.join(figures_dir, "task1_score_separability.png")

    generate_task1_preprocessing_comparison(prep_results, baseline_always_acc, fig1)
    generate_task1_metric_comparison(metric_results, baseline_always_acc, fig2)
    generate_task1_threshold_curve(full_curve, opt_th, (min_plat_th, max_plat_th), final_m, winning_title, fig3)
    generate_task1_validation_summary(cv_results, fig4)
    generate_task1_final_confusion_matrix(final_m["confusion_matrix"], opt_th, winning_title, fig5)
    generate_task1_final_results_table(b_always, b_rand, prep_results, final_m, cv_results["overall_oof_accuracy"], winning_title, fig6)

    scores_s = [winning_scores[i] for i in range(len(examples)) if gold_labels[i] == "S"]
    scores_ns = [winning_scores[i] for i in range(len(examples)) if gold_labels[i] == "NS"]
    generate_task1_score_separability(scores_s, scores_ns, opt_th, fig7)

    print(f"[Phase 3 Task 1] Execution completed successfully in {elapsed_time:.2f}s.")


# ==============================================================================
# PHASE 5: TASK 2 EXPERIMENTS & PRUNING ABLATIONS
# ==============================================================================

def compute_nli_pairs_fingerprint(pairs: List[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for p, hyp in pairs:
        h.update(p.encode("utf-8"))
        h.update(b"|||")
        h.update(hyp.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_nli_binding_fingerprint(pairs: Optional[List[Tuple[str, str]]], data: Any) -> str:
    h = hashlib.sha256()
    if isinstance(data, dict):
        for k in sorted(data.keys()):
            h.update(k.encode("utf-8"))
            h.update(b"|||")
            v = data[k]
            h.update(f"{v[0]:.8f},{v[1]:.8f},{v[2]:.8f}\n".encode("utf-8"))
    elif isinstance(data, list):
        if pairs is None or len(pairs) != len(data):
            raise ValueError(f"Binding fingerprint requires matching pairs vs data")
        for (p, hyp), prob in zip(pairs, data):
            h.update(p.encode("utf-8"))
            h.update(b"|||")
            h.update(hyp.encode("utf-8"))
            h.update(b"|||")
            h.update(f"{prob[0]:.8f},{prob[1]:.8f},{prob[2]:.8f}\n".encode("utf-8"))
    return h.hexdigest()


def validate_nli_cache(cache_path: str,
                       expected_count: Optional[int] = None,
                       manifest: Optional[Dict[str, Any]] = None,
                       cache_key: Optional[str] = None,
                       expected_pairs: Optional[List[Tuple[str, str]]] = None,
                       expected_dict_keys: Optional[Set[str]] = None,
                       require_manifest: bool = True,
                       expected_model_name: Optional[str] = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                       dataset_path: Optional[str] = "data/dev_labeled_ChatGPT.jsonl",
                       passages_path: Optional[str] = "data/passages_bm25_ChatGPT_humfacts.jsonl") -> Tuple[bool, str]:
    """
    Cryptographic and mathematical validation of NLI probability cache files.
    Enforces strict structural, distributional, provenance, and input-binding guarantees:
      1. File existence and JSON parse integrity.
      2. Manifest verification (model identity, tokenizer settings, dataset SHA-256 bindings).
      3. Candidate count and vector dimensionality (strictly 3 classes: Entailment, Neutral, Contradiction).
      4. Numerical validity: finite floats, in [0.0, 1.0], normalized sum to 1.0 +/- tolerance.
      5. Ordered premise-hypothesis pair fingerprint matching (pairs_sha256).
      6. Pair-probability binding fingerprint matching (binding_sha256).
      7. Payload byte-stream SHA-256 integrity (file_sha256).
    """
    if not os.path.exists(cache_path):
        return False, f"Cache file does not exist: {cache_path}"

    # 1. Payload file SHA-256 computation
    h_file = hashlib.sha256()
    try:
        with open(cache_path, "rb") as f:
            while chunk := f.read(65536):
                h_file.update(chunk)
    except Exception as e:
        return False, f"Error reading cache file '{cache_path}': {e}"
    actual_file_sha = h_file.hexdigest()

    # 2. JSON Parsing
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON parse error: {e}"

    # 3. Manifest Verification
    if require_manifest:
        if manifest is None:
            return False, f"Strict verification failure: cache manifest is required for '{cache_path}', but was not provided."
        if not isinstance(manifest, dict):
            return False, "Cache manifest is not a valid dictionary."
        if expected_model_name and manifest.get("model_name") != expected_model_name:
            return False, f"Model identity mismatch: expected '{expected_model_name}', got '{manifest.get('model_name')}'"

        tok_settings = manifest.get("tokenizer_settings")
        if not tok_settings or not isinstance(tok_settings, dict):
            return False, "Manifest missing required 'tokenizer_settings' configuration metadata."

        if not manifest.get("raw_dataset_sha256"):
            return False, "Manifest missing required 'raw_dataset_sha256' field."
        if not manifest.get("passages_sha256"):
            return False, "Manifest missing required 'passages_sha256' field."

        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                d_sha = hashlib.sha256(f.read()).hexdigest()
            if manifest.get("raw_dataset_sha256") != d_sha:
                return False, f"Dataset SHA-256 mismatch: manifest specifies '{manifest.get('raw_dataset_sha256')}', but '{dataset_path}' has '{d_sha}'"

        if passages_path and os.path.exists(passages_path):
            with open(passages_path, "rb") as f:
                p_sha = hashlib.sha256(f.read()).hexdigest()
            if manifest.get("passages_sha256") != p_sha:
                return False, f"Passages SHA-256 mismatch: manifest specifies '{manifest.get('passages_sha256')}', but '{passages_path}' has '{p_sha}'"

        if cache_key:
            if "caches" not in manifest or cache_key not in manifest["caches"]:
                return False, f"Cache key '{cache_key}' not registered in manifest."
            m_info = manifest["caches"][cache_key]
            if not m_info.get("file_sha256"):
                return False, f"Manifest entry for '{cache_key}' missing required 'file_sha256' field."
            if m_info["file_sha256"] != actual_file_sha:
                return False, f"Payload file SHA-256 mismatch for '{cache_key}': expected {m_info['file_sha256']}, computed {actual_file_sha}"
            if not m_info.get("binding_sha256"):
                return False, f"Manifest entry for '{cache_key}' missing required 'binding_sha256' field."
            if isinstance(data, list):
                if not m_info.get("pairs_sha256"):
                    return False, f"Manifest entry for list cache '{cache_key}' missing required 'pairs_sha256' field."

    # 4. Container & Mathematical Vector Validation
    if isinstance(data, list):
        if expected_count is not None and len(data) != expected_count:
            return False, f"Cache item count mismatch for '{cache_path}': expected {expected_count}, got {len(data)}"
        for idx, vec in enumerate(data):
            if not isinstance(vec, (list, tuple)):
                return False, f"Invalid probability container at index {idx}: expected list/tuple, got {type(vec)}"
            if len(vec) != 3:
                return False, f"Vector dimensionality error at index {idx}: expected 3 classes, got {len(vec)}"
            for c_idx, val in enumerate(vec):
                if not isinstance(val, (int, float)):
                    return False, f"Non-numeric probability at index {idx}[{c_idx}]: {type(val)}"
                if math.isnan(val) or math.isinf(val):
                    return False, f"Invalid probability (NaN or Inf) at index {idx}[{c_idx}]: {val}"
                if val < -1e-5 or val > 1.0 + 1e-5:
                    return False, f"Probability out of bounds [0, 1] at index {idx}[{c_idx}]: {val}"
            vec_sum = sum(vec)
            if abs(vec_sum - 1.0) > 0.01:
                return False, f"Probabilities do not sum to 1.0 within tolerance at index {idx}: sum = {vec_sum:.6f}"

        # 5. Cryptographic input-binding verification
        if expected_pairs is not None:
            if len(expected_pairs) != len(data):
                return False, f"Expected pairs count ({len(expected_pairs)}) does not match cache count ({len(data)})"
            if manifest and cache_key and cache_key in manifest.get("caches", {}):
                m_info = manifest["caches"][cache_key]
                if "pairs_sha256" in m_info:
                    p_sha = compute_nli_pairs_fingerprint(expected_pairs)
                    if p_sha != m_info["pairs_sha256"]:
                        return False, f"Pairs fingerprint mismatch for '{cache_key}': expected {m_info['pairs_sha256']}, computed {p_sha}"
                if "binding_sha256" in m_info:
                    b_sha = compute_nli_binding_fingerprint(expected_pairs, data)
                    if b_sha != m_info["binding_sha256"]:
                        return False, f"Binding fingerprint mismatch for '{cache_key}': expected {m_info['binding_sha256']}, computed {b_sha}"

        return True, f"Cryptographically verified list cache ({len(data)} items, SHA={actual_file_sha[:12]}...)"

    elif isinstance(data, dict):
        if expected_dict_keys is not None and set(data.keys()) != expected_dict_keys:
            return False, "Unpruned cache keys do not match current source-generated evidence pairs."
        if expected_count is not None and len(data) != expected_count:
            return False, f"Cache dictionary key count mismatch: expected {expected_count}, got {len(data)}"
        for k, vec in data.items():
            if not isinstance(k, str) or "|||" not in k:
                return False, f"Malformed cache dictionary key '{k[:40]}...': must be string containing '|||'"
            if not isinstance(vec, (list, tuple)):
                return False, f"Invalid probability container for key '{k[:40]}...': expected list/tuple, got {type(vec)}"
            if len(vec) != 3:
                return False, f"Vector dimensionality error for key '{k[:40]}...': expected 3 classes, got {len(vec)}"
            for c_idx, val in enumerate(vec):
                if not isinstance(val, (int, float)):
                    return False, f"Non-numeric probability for key '{k[:40]}...'[{c_idx}]: {type(val)}"
                if math.isnan(val) or math.isinf(val):
                    return False, f"Invalid probability (NaN or Inf) for key '{k[:40]}...'[{c_idx}]: {val}"
                if val < -1e-5 or val > 1.0 + 1e-5:
                    return False, f"Probability out of bounds [0, 1] for key '{k[:40]}...'[{c_idx}]: {val}"
            vec_sum = sum(vec)
            if abs(vec_sum - 1.0) > 0.01:
                return False, f"Probabilities do not sum to 1.0 for key '{k[:40]}...': sum = {vec_sum:.6f}"

        if manifest and cache_key and cache_key in manifest.get("caches", {}):
            m_info = manifest["caches"][cache_key]
            if "binding_sha256" in m_info:
                b_sha = compute_nli_binding_fingerprint(None, data)
                if b_sha != m_info["binding_sha256"]:
                    return False, f"Binding fingerprint mismatch for '{cache_key}': expected {m_info['binding_sha256']}, computed {b_sha}"

        return True, f"Cryptographically verified dict cache ({len(data)} keys, SHA={actual_file_sha[:12]}...)"

    return False, f"Unexpected cache container type: {type(data)}"


def run_cache_validation_negative_tests(cache_dir: str = "outputs/task2/cache") -> Dict[str, Any]:
    """
    Executes comprehensive negative tests to verify that the hardened NLI cache validator
    strictly rejects corrupted, tampered, missing, or malformed cache structures and metadata.
    Tested failure conditions:
      1. Missing manifest
      2. Missing probability values (fewer than 3 dimensions)
      3. Invalid probability values (out of bounds [0, 1])
      4. NaN or infinity values
      5. Swapped probability vectors (order corruption)
      6. Modified evidence pairs (altered input text)
      7. Changed model configuration
      8. Altered cache payload (file hash corruption)
      9. Mismatched dataset hash
      10. Missing raw_dataset_sha256 in manifest
      11. Missing passages_sha256 in manifest
      12. Missing file_sha256 in cache entry
      13. Missing pairs_sha256 in list cache entry
      14. Missing binding_sha256 in cache entry
      15. Missing tokenizer_settings in manifest
    """
    import tempfile

    test_results = {}
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(manifest_path):
        return {"all_passed": False, "error": f"Manifest not found at '{manifest_path}'"}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    win_cache_path = os.path.join(cache_dir, "pair_probs_window2.json")
    if not os.path.exists(win_cache_path):
        return {"all_passed": False, "error": f"Cache file not found at '{win_cache_path}'"}

    with open(win_cache_path, "r", encoding="utf-8") as f:
        genuine_data = json.load(f)

    # 1. Missing manifest test
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=None, require_manifest=True)
    test_results["test_missing_manifest"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 2. Missing probability values (truncated 2D vector)
    bad_dim_data = [[0.5, 0.5]] + genuine_data[1:]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(bad_dim_data, tf)
        tf_name = tf.name
    try:
        ok, msg = validate_nli_cache(tf_name, expected_count=600, manifest=manifest, cache_key="window2", require_manifest=False)
        test_results["test_missing_probability_values"] = {
            "expected_rejection": True,
            "rejected": not ok,
            "message": msg,
            "rejection_reason": msg
        }
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)

    # 3. Invalid probability values (out of [0, 1])
    bad_val_data = [[1.5, -0.2, -0.3]] + genuine_data[1:]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(bad_val_data, tf)
        tf_name = tf.name
    try:
        ok, msg = validate_nli_cache(tf_name, expected_count=600, manifest=manifest, cache_key="window2", require_manifest=False)
        test_results["test_invalid_probability_values"] = {
            "expected_rejection": True,
            "rejected": not ok,
            "message": msg,
            "rejection_reason": msg
        }
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)

    # 4. NaN or infinity
    bad_nan_data = [[float("nan"), 0.5, 0.5]] + genuine_data[1:]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(bad_nan_data, tf)
        tf_name = tf.name
    try:
        ok, msg = validate_nli_cache(tf_name, expected_count=600, manifest=manifest, cache_key="window2", require_manifest=False)
        test_results["test_nan_or_infinity"] = {
            "expected_rejection": True,
            "rejected": not ok,
            "message": msg,
            "rejection_reason": msg
        }
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)

    # 5. Swapped probability vectors (order corruption detected by binding fingerprint)
    dummy_pairs = [(f"Premise sentence {i}", f"Hypothesis fact {i}") for i in range(len(genuine_data))]
    swapped_data = [genuine_data[1], genuine_data[0]] + genuine_data[2:]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(swapped_data, tf)
        tf_name = tf.name
    try:
        swapped_file_sha = hashlib.sha256(open(tf_name, "rb").read()).hexdigest()
        swapped_manifest = copy.deepcopy(manifest)
        swapped_manifest["caches"]["window2"]["file_sha256"] = swapped_file_sha
        swapped_manifest["caches"]["window2"]["pairs_sha256"] = compute_nli_pairs_fingerprint(dummy_pairs)
        swapped_manifest["caches"]["window2"]["binding_sha256"] = compute_nli_binding_fingerprint(dummy_pairs, genuine_data)

        ok, msg = validate_nli_cache(tf_name, expected_count=600, manifest=swapped_manifest, cache_key="window2",
                                     expected_pairs=dummy_pairs, require_manifest=True)
        test_results["test_swapped_probability_vectors"] = {
            "expected_rejection": True,
            "rejected": not ok,
            "message": msg,
            "rejection_reason": msg
        }
    finally:
        if os.path.exists(tf_name):
            os.remove(tf_name)

    # 6. Modified evidence pairs (altered input text detected by pairs fingerprint)
    pairs_manifest = copy.deepcopy(manifest)
    pairs_manifest["caches"]["window2"]["pairs_sha256"] = compute_nli_pairs_fingerprint(dummy_pairs)
    dummy_pairs_mod = [("Tampered premise sentence", dummy_pairs[0][1])] + dummy_pairs[1:]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=pairs_manifest, cache_key="window2",
                                 expected_pairs=dummy_pairs_mod, require_manifest=True)
    test_results["test_modified_evidence_pairs"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 7. Changed model configuration in manifest
    bad_model_manifest = copy.deepcopy(manifest)
    bad_model_manifest["model_name"] = "bert-base-uncased"
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=bad_model_manifest, cache_key="window2", require_manifest=True)
    test_results["test_changed_model_configuration"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 8. Altered cache payload (payload hash mismatch)
    bad_sha_manifest = copy.deepcopy(manifest)
    bad_sha_manifest["caches"]["window2"]["file_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=bad_sha_manifest, cache_key="window2", require_manifest=True)
    test_results["test_altered_cache_payload"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 9. Mismatched dataset hash
    bad_ds_manifest = copy.deepcopy(manifest)
    bad_ds_manifest["raw_dataset_sha256"] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=bad_ds_manifest, cache_key="window2",
                                 dataset_path="data/dev_labeled_ChatGPT.jsonl", require_manifest=True)
    test_results["test_mismatched_dataset_hash"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 10. Missing raw_dataset_sha256 in manifest
    manifest_no_raw_ds = copy.deepcopy(manifest)
    del manifest_no_raw_ds["raw_dataset_sha256"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_raw_ds, cache_key="window2", require_manifest=True)
    test_results["test_missing_raw_dataset_sha256"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 11. Missing passages_sha256 in manifest
    manifest_no_passages = copy.deepcopy(manifest)
    del manifest_no_passages["passages_sha256"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_passages, cache_key="window2", require_manifest=True)
    test_results["test_missing_passages_sha256"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 12. Missing file_sha256 in cache entry
    manifest_no_file_sha = copy.deepcopy(manifest)
    del manifest_no_file_sha["caches"]["window2"]["file_sha256"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_file_sha, cache_key="window2", require_manifest=True)
    test_results["test_missing_file_sha256"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 13. Missing pairs_sha256 in list cache entry
    manifest_no_pairs_sha = copy.deepcopy(manifest)
    del manifest_no_pairs_sha["caches"]["window2"]["pairs_sha256"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_pairs_sha, cache_key="window2", require_manifest=True)
    test_results["test_missing_pairs_sha256"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 14. Missing binding_sha256 in cache entry
    manifest_no_binding = copy.deepcopy(manifest)
    del manifest_no_binding["caches"]["window2"]["binding_sha256"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_binding, cache_key="window2", require_manifest=True)
    test_results["test_missing_binding_sha256"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    # 15. Missing tokenizer_settings in manifest
    manifest_no_tok = copy.deepcopy(manifest)
    del manifest_no_tok["tokenizer_settings"]
    ok, msg = validate_nli_cache(win_cache_path, expected_count=600, manifest=manifest_no_tok, cache_key="window2", require_manifest=True)
    test_results["test_missing_tokenizer_settings"] = {
        "expected_rejection": True,
        "rejected": not ok,
        "message": msg,
        "rejection_reason": msg
    }

    all_passed = all(t["rejected"] for t in test_results.values())
    return {
        "all_passed": all_passed,
        "total_tests": len(test_results),
        "passed_count": sum(1 for t in test_results.values() if t["rejected"]),
        "tests": test_results
    }


def run_threshold_only_cv(max_scores: List[float], labels: List[str],
                          folds: List[List[str]], idx_to_topic: Dict[int, str]) -> Dict[str, Any]:
    fold_records = []
    oof_preds = [None] * len(labels)

    for f_idx in range(len(folds)):
        val_topics = set(folds[f_idx])
        val_indices = [i for i in range(len(labels)) if idx_to_topic[i] in val_topics]
        train_indices = [i for i in range(len(labels)) if idx_to_topic[i] not in val_topics]

        best_tr_acc, best_th = -1, None
        for th in np.linspace(0.05, 0.95, 91):
            tr_correct = sum(1 for i in train_indices if (labels[i] == "S") == (max_scores[i] >= th))
            tr_acc = tr_correct / len(train_indices)
            if tr_acc > best_tr_acc:
                best_tr_acc = tr_acc
                best_th = float(th)

        val_correct = sum(1 for i in val_indices if (labels[i] == "S") == (max_scores[i] >= best_th))
        val_acc = val_correct / len(val_indices)
        for i in val_indices:
            oof_preds[i] = "S" if max_scores[i] >= best_th else "NS"

        cm_val = calculate_confusion_matrix([labels[i] for i in val_indices], [oof_preds[i] for i in val_indices])
        m_val = calculate_metrics(cm_val)

        fold_records.append({
            "fold": f_idx,
            "val_topics": sorted(list(val_topics)),
            "val_count": len(val_indices),
            "train_count": len(train_indices),
            "train_accuracy": float(best_tr_acc),
            "val_accuracy": float(val_acc),
            "val_correct": val_correct,
            "val_total": len(val_indices),
            "selected_threshold": best_th,
            "metrics": m_val
        })

    cm_oof = calculate_confusion_matrix(labels, oof_preds)
    m_oof = calculate_metrics(cm_oof)

    mean_val = float(np.mean([f["val_accuracy"] for f in fold_records]))
    std_val = float(np.std([f["val_accuracy"] for f in fold_records]))
    mean_train = float(np.mean([f["train_accuracy"] for f in fold_records]))
    gap = mean_train - mean_val

    return {
        "folds": fold_records,
        "mean_train_accuracy": mean_train,
        "mean_val_accuracy": mean_val,
        "std_val_accuracy": std_val,
        "overfitting_gap": gap,
        "pooled_oof_accuracy": m_oof["accuracy"],
        "pooled_oof_correct": m_oof["correct"],
        "pooled_oof_total": m_oof["total"],
        "pooled_oof_metrics": m_oof,
        "oof_predictions": oof_preds
    }


def generate_task2_strategy_comparison(strategy_records: List[Dict[str, Any]], target_acc: float = 0.83,
                                       output_path: str = "outputs/task2/figures/task2_strategy_comparison.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    names = [s["name"] for s in strategy_records]
    accs = [s["accuracy"] * 100 for s in strategy_records]
    s_f1s = [s["f1_s"] * 100 for s in strategy_records]
    ns_f1s = [s["f1_ns"] * 100 for s in strategy_records]
    macro_f1s = [s["macro_f1"] * 100 for s in strategy_records]

    x = np.arange(len(names))
    width = 0.20

    r1 = ax.bar(x - 1.5 * width, accs, width, label="Overall Accuracy", color="#0F766E", edgecolor="#1E293B", linewidth=0.8)
    r2 = ax.bar(x - 0.5 * width, s_f1s, width, label="Supported F1 (S)", color="#2563EB", edgecolor="#1E293B", linewidth=0.8)
    r3 = ax.bar(x + 0.5 * width, ns_f1s, width, label="Not Supported F1 (NS)", color="#64748B", edgecolor="#1E293B", linewidth=0.8)
    r4 = ax.bar(x + 1.5 * width, macro_f1s, width, label="Macro F1", color="#1E293B", edgecolor="#1E293B", linewidth=0.8)

    ax.axhline(target_acc * 100, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Coursework Target (≥ {target_acc*100:.1f}%)")

    for rects in [r1, r2, r3, r4]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 2: Binary Decision Strategy Optimization Comparison (DeBERTa-v3)", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_pruning_tradeoff(pruning_records: List[Dict[str, Any]],
                                    output_path: str = "outputs/task2/figures/task2_pruning_tradeoff.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    top3_conservative = [r for r in pruning_records if r.get("top_k") == 3 and r.get("fallback") == "conservative"]
    top3_conservative.sort(key=lambda x: x["gamma"])

    gammas = [r["gamma"] for r in top3_conservative]
    accs = [r["best_prob_accuracy"] * 100 for r in top3_conservative]
    pairs = [r["pairs"] for r in top3_conservative]

    color1 = "#2563EB"
    ax1.set_xlabel("Lexical Pruning Overlap Threshold (γ)", fontsize=10, fontweight="bold", color="#1E293B")
    ax1.set_ylabel("Development Accuracy (%)", color=color1, fontsize=10, fontweight="bold")
    line1 = ax1.plot(gammas, accs, marker="o", color=color1, linewidth=2.2, label="Accuracy (Top-3 Conservative)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(70, 90)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color2 = "#0F766E"
    ax2.set_ylabel("Evaluated Candidate Pairs (Inference Load)", color=color2, fontsize=10, fontweight="bold")
    line2 = ax2.plot(gammas, pairs, marker="s", color=color2, linewidth=2.0, linestyle="--", label="Candidate Pairs Evaluated")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(pairs) * 1.25 if pairs else 1000)

    for g, a, p in zip(gammas, accs, pairs):
        ax1.annotate(f"{a:.2f}%", (g, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, fontweight="bold", color=color1)
        ax2.annotate(f"{p} pairs", (g, p), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.95)

    plt.title("Task 2: Lexical Evidence Pruning Trade-off (Accuracy vs Computational Load)", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_runtime_comparison(runtime_plot_data: List[Dict[str, Any]],
                                      output_path: str = "outputs/task2/figures/task2_runtime_comparison.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    labels = [item["label"] for item in runtime_plot_data]
    seconds = [item["seconds"] for item in runtime_plot_data]

    palette = ["#64748B", "#0F766E", "#2563EB", "#94A3B8", "#1E293B", "#38BDF8", "#475569"]
    bar_colors = [palette[i % len(palette)] for i in range(len(labels))]

    bars = ax.bar(range(len(labels)), seconds, color=bar_colors, edgecolor="#1E293B", linewidth=0.8, width=0.55)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}s", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Execution Time (Seconds, CPU)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 2: DeBERTa-v3 CPU Inference Execution Latency Benchmarks", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, max(seconds) * 1.18 if seconds else 700)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_threshold_curve(th_curve_data: List[Dict[str, Any]], opt_th: float,
                                   output_path: str = "outputs/task2/figures/task2_threshold_calibration_curve.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    plot_pts = [p for p in th_curve_data if 0.10 <= p["threshold"] <= 0.90]
    if not plot_pts:
        plot_pts = th_curve_data

    ths = [p["threshold"] for p in plot_pts]
    accs = [p["accuracy"] * 100 for p in plot_pts]
    bal_accs = [p["balanced_accuracy"] * 100 for p in plot_pts]
    f1_s = [p["f1_s"] * 100 if p["f1_s"] is not None else 0.0 for p in plot_pts]
    f1_ns = [p["f1_ns"] * 100 if p["f1_ns"] is not None else 0.0 for p in plot_pts]

    ax.plot(ths, accs, label="Overall Accuracy", color="#2563EB", linewidth=2.2)
    ax.plot(ths, bal_accs, label="Balanced Accuracy", color="#0F766E", linewidth=1.8, linestyle="-.")
    ax.plot(ths, f1_s, label="F1-Score (Supported - S)", color="#1E293B", linewidth=1.5, linestyle=":")
    ax.plot(ths, f1_ns, label="F1-Score (Not Supported - NS)", color="#64748B", linewidth=1.5, linestyle="--")

    ax.axhline(83.0, color="#DC2626", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 83.0%)")
    ax.axvline(opt_th, color="#1E293B", linestyle="-", linewidth=1.8, label=f"Optimal τ* = {opt_th:.2f}")

    opt_records = [p for p in plot_pts if abs(p["threshold"] - opt_th) < 1e-4]
    if opt_records:
        peak_acc = opt_records[0]["accuracy"] * 100
        ax.plot(opt_th, peak_acc, marker="o", markersize=7, color="#1E293B")
        ax.annotate(f"Optimal τ* = {opt_th:.2f}\nPeak Acc: {peak_acc:.2f}%",
                    xy=(opt_th, peak_acc), xytext=(opt_th - 0.22, peak_acc - 10),
                    arrowprops=dict(facecolor="#1E293B", shrink=0.08, width=1, headwidth=5),
                    fontsize=8.5, fontweight="bold", color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#1E293B"))

    ax.set_xlabel("Entailment Decision Threshold (τ)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 2: Decision Threshold Optimization Curve (DeBERTa-v3 All-Retained)", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(35, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_validation_summary(cv_all_retained: Dict[str, Any], cv_top3: Dict[str, Any],
                                      cv_unpruned: Dict[str, Any],
                                      output_path: str = "outputs/task2/figures/task2_validation_summary.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)

    folds = cv_all_retained["folds"]
    total_oof_n = sum(f.get("val_count", 0) for f in folds)
    labels = [f"Fold {f['fold']}\n(N={f['val_count']})" for f in folds] + [f"Pooled OOF\n(N={total_oof_n})"]

    all_val = [f["val_accuracy"] * 100 for f in folds] + [cv_all_retained["pooled_oof_accuracy"] * 100]
    top3_val = [f["val_accuracy"] * 100 for f in cv_top3["folds"]] + [cv_top3["pooled_oof_accuracy"] * 100]
    unp_val = [f["val_accuracy"] * 100 for f in cv_unpruned["folds"]] + [cv_unpruned["pooled_oof_accuracy"] * 100]

    x = np.arange(len(labels))
    width = 0.26

    r1 = ax.bar(x - width, all_val, width, label=f"All-Retained Val (Mean {cv_all_retained['mean_val_accuracy']*100:.1f}%)", color="#0F766E", edgecolor="#1E293B", linewidth=0.8)
    r2 = ax.bar(x, top3_val, width, label=f"Top-3 Val (Mean {cv_top3['mean_val_accuracy']*100:.1f}%)", color="#2563EB", edgecolor="#1E293B", linewidth=0.8)
    r3 = ax.bar(x + width, unp_val, width, label=f"Unpruned Val (Mean {cv_unpruned['mean_val_accuracy']*100:.1f}%)", color="#64748B", edgecolor="#1E293B", linewidth=0.8)

    ax.axhline(83.0, color="#DC2626", linestyle="--", linewidth=1.4, label="Coursework Target (≥ 83.0%)")

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1E293B")

    ax.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Task 2: Grouped 5-Fold Biography Cross-Validation Stability Across Configurations", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_ylim(50, 105)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_confusion_matrix(cm: List[List[int]], opt_th: float,
                                    output_path: str = "outputs/task2/figures/task2_final_confusion_matrix.png") -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)

    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    categories = ["Supported (S)", "Not Supported (NS)"]

    mask = np.array([[1, 0], [0, 1]])
    cmap = matplotlib.colors.ListedColormap(["#DBEAFE", "#1D4ED8"])

    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
    title_txt = f"Task 2 Final Confusion Matrix (DeBERTa-v3 @ τ*={opt_th:.2f})\nAccuracy: {tp+tn}/{total} = {acc*100:.2f}%"
    ax.set_title(title_txt, fontsize=10.5, fontweight="bold", color="#1E293B", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylabel("True Label (Ground Truth)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_yticklabels(categories, fontsize=9.5, fontweight="bold", color="#1E293B")

    tags = [["TP", "FN"], ["FP", "TN"]]
    cm_arr = np.array([[tp, fn], [fp, tn]])
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            pct = val / total * 100.0 if total > 0 else 0.0
            tag = tags[i][j]
            color = "white" if mask[i, j] == 1 else "#1E293B"
            ax.text(j, i, f"[{tag}]\n{val}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task2_final_results_table(b_always: Dict[str, Any], b_rand: Dict[str, Any],
                                       t1_m: Dict[str, Any], t1_oof_acc: float,
                                       t2_m: Dict[str, Any], t2_oof_acc: float,
                                       f_passage: Dict[str, Any], f_unpruned: Dict[str, Any],
                                       f_win: Dict[str, Any], f_title: Dict[str, Any],
                                       unpruned_oof_acc: Optional[float] = None,
                                       output_path: str = "outputs/task2/figures/task2_final_results_table.png",
                                       t1_pairs_count: Optional[int] = None,
                                       t2_pairs_count: Optional[int] = None) -> str:
    if not _HAS_MATPLOTLIB:
        return ""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 5.4), dpi=200)
    ax.axis("off")

    headers = ["Model / Configuration", "Evidence Units", "Candidate Pairs", "Dev Acc (%)", "OOF CV Acc (%)", "S-F1", "NS-F1", "Macro-F1"]

    b_always_acc = b_always.get("accuracy", 0.0) * 100
    b_always_sf1 = f"{b_always.get('f1_s', 0.0):.4f}" if b_always.get("f1_s") is not None else "N/A"
    b_always_nsf1 = f"{b_always.get('f1_ns', 0.0):.4f}" if b_always.get("f1_ns") is not None else "N/A"
    b_always_mf1 = f"{b_always.get('macro_f1', 0.0):.4f}" if b_always.get("macro_f1") is not None else "N/A"

    b_rand_acc = f"{b_rand.get('accuracy', 0.0) * 100:.2f}%" if b_rand and b_rand.get("accuracy") is not None else "N/A"
    b_rand_sf1 = f"{b_rand.get('f1_s', 0.0):.4f}" if b_rand and b_rand.get("f1_s") is not None else "N/A"
    b_rand_nsf1 = f"{b_rand.get('f1_ns', 0.0):.4f}" if b_rand and b_rand.get("f1_ns") is not None else "N/A"
    b_rand_mf1 = f"{b_rand.get('macro_f1', 0.0):.4f}" if b_rand and b_rand.get("macro_f1") is not None else "N/A"

    t1_pairs_str = f"{t1_pairs_count:,}" if isinstance(t1_pairs_count, int) else "N/A"
    pass_pairs_str = f"{f_passage.get('pairs_evaluated'):,}" if isinstance(f_passage.get('pairs_evaluated'), int) else str(f_passage.get('pairs_evaluated', "N/A"))
    unp_pairs_str = f"{f_unpruned.get('pairs_evaluated'):,}" if isinstance(f_unpruned.get('pairs_evaluated'), int) else str(f_unpruned.get('pairs_evaluated', "N/A"))
    win_pairs_str = f"{f_win.get('pairs_evaluated'):,}" if isinstance(f_win.get('pairs_evaluated'), int) else str(f_win.get('pairs_evaluated', "N/A"))
    title_pairs_str = f"{f_title.get('pairs_evaluated'):,}" if isinstance(f_title.get('pairs_evaluated'), int) else str(f_title.get('pairs_evaluated', "N/A"))
    t2_pairs_str = f"{t2_pairs_count:,}" if isinstance(t2_pairs_count, int) else "N/A"

    t1_acc_str = f"{t1_m.get('accuracy', 0.0)*100:.2f}%" if t1_m and t1_m.get('accuracy') is not None else "N/A"
    t1_oof_str = f"{t1_oof_acc*100:.2f}%" if t1_oof_acc is not None else "N/A"
    t1_sf1_str = f"{t1_m.get('f1_s', 0.0):.4f}" if t1_m and t1_m.get('f1_s') is not None else "N/A"
    t1_nsf1_str = f"{t1_m.get('f1_ns', 0.0):.4f}" if t1_m and t1_m.get('f1_ns') is not None else "N/A"
    t1_mf1_str = f"{t1_m.get('macro_f1', 0.0):.4f}" if t1_m and t1_m.get('macro_f1') is not None else "N/A"

    unp_oof_str = f"{unpruned_oof_acc*100:.2f}%" if unpruned_oof_acc is not None else "N/A"

    fmt_acc = lambda d: f"{d['accuracy']*100:.2f}%" if d and d.get('accuracy') is not None else "N/A"
    fmt_f1_val = lambda d, k: f"{d[k]:.4f}" if d and d.get(k) is not None else "N/A"

    rows = [
        ["Baseline: Always Supported", "None (Null)", "0", f"{b_always_acc:.2f}%" if isinstance(b_always_acc, (int, float)) else "N/A", "N/A", b_always_sf1, b_always_nsf1, b_always_mf1],
        ["Baseline: Random Guess (seed=42)", "N/A", "N/A", b_rand_acc, "N/A", b_rand_sf1, b_rand_nsf1, b_rand_mf1],
        ["Task 1: Word Overlap (Porter-Stemmed)", "All Passages", t1_pairs_str, t1_acc_str, t1_oof_str, t1_sf1_str, t1_nsf1_str, t1_mf1_str],
        ["T2: Full Passage (DeBERTa-v3)", "Passage", pass_pairs_str, fmt_acc(f_passage), "N/A", fmt_f1_val(f_passage, 'f1_s'), fmt_f1_val(f_passage, 'f1_ns'), fmt_f1_val(f_passage, 'macro_f1')],
        ["T2: Unpruned Sentences (DeBERTa-v3)", "Sentence", unp_pairs_str, fmt_acc(f_unpruned), unp_oof_str, fmt_f1_val(f_unpruned, 'f1_s'), fmt_f1_val(f_unpruned, 'f1_ns'), fmt_f1_val(f_unpruned, 'macro_f1')],
        ["T2: 2-Sentence Window (DeBERTa-v3)", "Window-2", win_pairs_str, fmt_acc(f_win), "N/A", fmt_f1_val(f_win, 'f1_s'), fmt_f1_val(f_win, 'f1_ns'), fmt_f1_val(f_win, 'macro_f1')],
        ["T2: Title-Sentence (DeBERTa-v3)", "Title:Sent", title_pairs_str, fmt_acc(f_title), "N/A", fmt_f1_val(f_title, 'f1_s'), fmt_f1_val(f_title, 'f1_ns'), fmt_f1_val(f_title, 'macro_f1')],
        ["T2 Winning: All-Retained (γ=0.05)", "Pruned Sent", t2_pairs_str, fmt_acc(t2_m), f"{t2_oof_acc*100:.2f}%" if t2_oof_acc is not None else "N/A", fmt_f1_val(t2_m, 'f1_s'), fmt_f1_val(t2_m, 'f1_ns'), fmt_f1_val(t2_m, 'macro_f1')]
    ]

    col_widths = [0.28, 0.13, 0.11, 0.10, 0.11, 0.09, 0.09, 0.09]
    table = ax.table(cellText=rows, colLabels=headers, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)

    for (r_idx, c_idx), cell in table.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx == len(rows):
            cell.set_facecolor("#DBEAFE")
            cell.set_text_props(color="#1E293B", fontweight="bold")
        elif r_idx % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        if c_idx == 0 and r_idx > 0:
            cell.set_text_props(ha="left")

    plt.title("Task 2: Textual Entailment Comprehensive Performance & Ablation Summary", fontsize=11, fontweight="bold", pad=12, color="#1E293B")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def ensure_task2_inference_caches(cache_dir: str, model_name: str, dev_path: str,
                                  passages_path: str, facts: List[str],
                                  cache_pairs: Dict[str, List[Tuple[str, str]]],
                                  profile_top3_pairs: List[Tuple[str, str]],
                                  force_rebuild: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    specs = {
        "sweep1519": ("pair_probs_sweep1519.json", 256, 8),
        "unpruned_full": ("pair_probs_unpruned_full.json", 256, 8),
        "passage": ("pair_probs_passage.json", 512, 4),
        "window2": ("pair_probs_window2.json", 256, 8),
        "title_sentence": ("pair_probs_title_sentence.json", 256, 8),
    }
    os.makedirs(cache_dir, exist_ok=True)
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    benchmark_path = os.path.join(cache_dir, "inference_benchmarks.json")
    has_manifest = os.path.isfile(manifest_path)
    existing = [name for name, (filename, _, _) in specs.items()
                if os.path.isfile(os.path.join(cache_dir, filename))]
    if has_manifest and not force_rebuild:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = None
        if existing and not force_rebuild:
            raise RuntimeError(
                "NLI cache manifest is missing but probability files exist. "
                "Restore the matching verified manifest, or use --rebuild-cache for real inference."
            )

    if manifest is not None and not force_rebuild:
        settings = manifest.get("tokenizer_settings", {})
        required_settings = {"padding": True, "truncation": True,
                             "max_lengths": {"sentence_level": 256, "passage_level": 512}}
        if manifest.get("model_name") != model_name or settings != required_settings:
            raise RuntimeError("Cached NLI model/tokenizer settings are incompatible. Use --rebuild-cache to rerun inference.")
        if len(facts) != manifest.get("total_facts"):
            raise RuntimeError("Cached fact count differs from source data. Use --rebuild-cache to rerun inference.")
        for key in existing:
            filename, _, _ = specs[key]
            pairs = cache_pairs[key]
            expected = None if key == "unpruned_full" else pairs
            dict_keys = {f"{a}|||{b}" for a, b in pairs} if key == "unpruned_full" else None
            ok, reason = validate_nli_cache(
                os.path.join(cache_dir, filename), expected_count=len(set(pairs)) if key == "unpruned_full" else len(pairs),
                manifest=manifest, cache_key=key, expected_pairs=expected,
                expected_dict_keys=dict_keys, require_manifest=True,
                expected_model_name=model_name, dataset_path=dev_path, passages_path=passages_path
            )
            if not ok:
                raise RuntimeError(f"Existing NLI cache '{key}' failed integrity checking: {reason}. "
                                   "Restore matching originals or explicitly use --rebuild-cache.")

    missing = list(specs) if force_rebuild else [key for key in specs if key not in existing]
    if missing and existing and not force_rebuild:
        print("[Task 2 Cache] Partial cache set detected; regenerating all five together to avoid mixing model snapshots.")
        missing = list(specs)
        force_rebuild = True
    if not missing:
        print("[Task 2 Cache] Reusing all five strictly verified inference caches.")
        bench = {}
        if os.path.isfile(benchmark_path):
            with open(benchmark_path, encoding="utf-8") as f:
                bench = json.load(f)
        return manifest, bench

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "Missing NLI caches require fresh pretrained DeBERTa inference. "
            "Install requirements.txt in Python 3.10, or restore the matching verified caches."
        ) from exc

    from factcheck import EntailmentModel
    import tempfile

    print(f"[Task 2 Cache] {len(missing)} missing/rebuild cache(s); computing genuine DeBERTa probabilities.")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    wrapper = EntailmentModel(model, tok, cuda=False)

    fresh_manifest = copy.deepcopy(manifest) if manifest is not None and not force_rebuild else {
        "model_name": model_name,
        "device": "CPU",
        "torch_version": str(torch.__version__),
        "source_dataset": dev_path,
        "raw_dataset_sha256": hashlib.sha256(open(dev_path, "rb").read()).hexdigest(),
        "source_passages": passages_path,
        "passages_sha256": hashlib.sha256(open(passages_path, "rb").read()).hexdigest(),
        "total_facts": len(facts),
        "facts_sha256": hashlib.sha256("".join(f + "\n" for f in facts).encode("utf-8")).hexdigest(),
        "tokenizer_settings": {"padding": True, "truncation": True,
                               "max_lengths": {"sentence_level": 256, "passage_level": 512}},
        "caches": {},
    }
    if not os.path.isfile(benchmark_path) or force_rebuild:
        bench = {"metadata": {"device": "CPU", "model_name": model_name,
                               "torch_version": str(torch.__version__),
                               "description": "Measured live CPU inference runs; unavailable timings are omitted."},
                 "batch_size_pilot_benchmarks": [], "full_dataset_benchmarks": {}}
    else:
        with open(benchmark_path, encoding="utf-8") as f:
            bench = json.load(f)
        bench.setdefault("full_dataset_benchmarks", {})

    with tempfile.TemporaryDirectory(prefix=".nli_rebuild_", dir=cache_dir) as temporary:
        rebuilt = {}
        for key in missing:
            filename, max_length, batch_size = specs[key]
            pairs = cache_pairs[key]
            if key == "unpruned_full":
                pairs = list(dict.fromkeys(pairs))
                keys = [f"{a}|||{b}" for a, b in pairs]
                if len(set(keys)) != len(keys):
                    raise ValueError("Unpruned evidence keys collide; cannot safely create a dictionary cache.")
            print(f"[Task 2 Cache] Live inference for {key}: {len(pairs)} pairs, max_length={max_length}.")
            start = time.perf_counter()
            values = wrapper.check_entailment_batch(
                [a for a, _ in pairs], [b for _, b in pairs],
                batch_size=batch_size, max_length=max_length
            )
            elapsed = time.perf_counter() - start
            if len(values) != len(pairs):
                raise RuntimeError(f"Model returned {len(values)} outputs for {len(pairs)} inputs ({key}).")
            data = {k: list(v) for k, v in zip(keys, values)} if key == "unpruned_full" else [list(v) for v in values]
            temp_path = os.path.join(temporary, filename)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            info = {
                "filename": filename, "formulation": key,
                "evaluated_instances": len(pairs), "unique_pairs": len(pairs),
                "max_length": max_length,
                "file_sha256": hashlib.sha256(open(temp_path, "rb").read()).hexdigest(),
                "binding_sha256": compute_nli_binding_fingerprint(None if key == "unpruned_full" else pairs, data),
                "provenance": f"Fresh CPU forward passes (B={batch_size}, max_length={max_length})",
            }
            if key != "unpruned_full":
                info["pairs_sha256"] = compute_nli_pairs_fingerprint(pairs)
            fresh_manifest["caches"][key] = info
            rebuilt[key] = temp_path
            if key == "sweep1519":
                bench_key = "all_retained"
            elif key == "unpruned_full":
                bench_key = "unpruned_full"
            else:
                bench_key = None
            if bench_key:
                bench["full_dataset_benchmarks"][bench_key] = {
                    "evaluated_instances": len(pairs), "unique_pairs": len(pairs),
                    "batch_size": batch_size, "wall_clock_seconds": elapsed,
                    "throughput_pairs_per_sec": len(pairs) / elapsed if elapsed else None,
                    "ms_per_pair": elapsed * 1000 / len(pairs) if pairs else None,
                    "provenance": "Measured live CPU forward passes in this cache rebuild",
                }
            print(f"[Task 2 Cache] {key}: {elapsed:.2f} s measured; {len(pairs)} model outputs.")

        if "top3_pruned" not in bench["full_dataset_benchmarks"] and profile_top3_pairs:
            start = time.perf_counter()
            measured = wrapper.check_entailment_batch(
                [a for a, _ in profile_top3_pairs], [b for _, b in profile_top3_pairs],
                batch_size=8, max_length=256
            )
            elapsed = time.perf_counter() - start
            if len(measured) != len(profile_top3_pairs):
                raise RuntimeError("Top-three benchmark did not return one probability per pair.")
            bench["full_dataset_benchmarks"]["top3_pruned"] = {
                "evaluated_instances": len(profile_top3_pairs),
                "unique_pairs": len(set(profile_top3_pairs)),
                "batch_size": 8, "wall_clock_seconds": elapsed,
                "throughput_pairs_per_sec": len(profile_top3_pairs) / elapsed if elapsed else None,
                "ms_per_pair": elapsed * 1000 / len(profile_top3_pairs),
                "provenance": "Measured live CPU forward passes in this cache rebuild",
            }
        if not bench.get("batch_size_pilot_benchmarks"):
            pilot_pairs = cache_pairs["unpruned_full"][:75]
            for batch_size in (1, 4, 8, 16):
                start = time.perf_counter()
                measured = wrapper.check_entailment_batch(
                    [a for a, _ in pilot_pairs], [b for _, b in pilot_pairs],
                    batch_size=batch_size, max_length=256
                )
                elapsed = time.perf_counter() - start
                if len(measured) != len(pilot_pairs):
                    raise RuntimeError("Batch-size pilot inference returned an incorrect count.")
                bench["batch_size_pilot_benchmarks"].append({
                    "batch_size": batch_size, "pairs": len(pilot_pairs),
                    "seconds": elapsed,
                    "ms_per_pair": elapsed * 1000 / len(pilot_pairs),
                    "provenance": f"Measured live CPU inference on a {len(pilot_pairs)}-pair pilot",
                })

        manifest_tmp = os.path.join(temporary, "cache_manifest.json")
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(fresh_manifest, f, indent=2)
        bench_tmp = os.path.join(temporary, "inference_benchmarks.json")
        with open(bench_tmp, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=2)
        for key, tmp in rebuilt.items():
            os.replace(tmp, os.path.join(cache_dir, specs[key][0]))
        os.replace(manifest_tmp, manifest_path)
        os.replace(bench_tmp, benchmark_path)

    del wrapper, model, tok
    gc.collect()
    print("[Task 2 Cache] Saved model-generated probabilities and hash-bound manifest.")
    return fresh_manifest, bench


def run_task2(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
              dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
              figures_dir: str = "outputs/task2/figures",
              outputs_dir: str = "outputs/task2",
              cache_dir: str = "outputs/task2/cache",
              model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
              rebuild_cache: bool = False) -> None:
    """
    Main entry point for Phase 5 Task 2 Textual Entailment Fact-Checking with DeBERTa-v3.
    """
    start_time = time.time()
    print("=" * 80)
    print("Entailment Experiments")
    print(f"Model: {model_name} (DeBERTa-v3-base)")
    print("=" * 80)

    os.makedirs(cache_dir, exist_ok=True)

    print("\n[Phase 5 Task 2] Loading datasets...")
    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    facts = [ex.fact for ex in examples]
    total_facts = len(facts)
    print(f"  Loaded {total_facts} development facts across {len(raw_bios)} biography topics.")

    # Topic-to-Index
    idx_to_topic = {}
    cur_idx = 0
    for b in raw_bios:
        topic = b.get("topic")
        for s in b.get("annotations", []):
            for f in s.get("human-atomic-facts", []):
                idx_to_topic[cur_idx] = topic
                cur_idx += 1

    topic_counts = Counter(idx_to_topic.values())
    folds = [[] for _ in range(5)]
    fold_counts = [0] * 5
    folds[0].extend(["Marianne McAndrew", "Mike Trivisonno"])
    fold_counts[0] += topic_counts["Marianne McAndrew"] + topic_counts["Mike Trivisonno"]

    rem = [t for t in topic_counts if t not in ("Marianne McAndrew", "Mike Trivisonno")]
    rem.sort(key=lambda t: -topic_counts[t])
    for t in rem:
        m = fold_counts.index(min(fold_counts))
        folds[m].append(t)
        fold_counts[m] += topic_counts[t]


    # 1. Candidate extraction & sentence pruning representations
    all_scored_by_ex = []
    all_raw_pairs_by_ex = []
    unique_pairs_gamma05 = []
    pair_to_id_gamma05 = {}

    for ex in examples:
        raw_sents = []
        for p in ex.passages:
            for s in split_into_sentences(p.get("text", "")):
                if s.strip():
                    raw_sents.append(s.strip())
        fact_pairs = [(sent, ex.fact) for sent in raw_sents]
        all_raw_pairs_by_ex.append(fact_pairs)

        fact_tokens = get_tokens_for_pruning(ex.fact) or ex.fact.lower().split()
        scored = []
        for s_unit in raw_sents:
            u_toks = set(get_tokens_for_pruning(s_unit))
            rec = compute_overlap_recall(fact_tokens, u_toks)
            scored.append((s_unit, rec))
        scored.sort(key=lambda x: x[1], reverse=True)
        all_scored_by_ex.append(scored)

        keep = [s_unit for s_unit, rec in scored if rec >= 0.05]
        if not keep and scored:
            keep = [scored[0][0]]
        for s_unit in keep:
            key = (s_unit, ex.fact)
            if key not in pair_to_id_gamma05:
                pair_to_id_gamma05[key] = len(unique_pairs_gamma05)
                unique_pairs_gamma05.append(key)

    total_raw_sentences = sum(len(p) for p in all_raw_pairs_by_ex)
    total_unique_unpruned = len({pair for pairs in all_raw_pairs_by_ex for pair in pairs})
    total_unique_gamma05 = len(unique_pairs_gamma05)

    # 2. Passage pairs formulation
    passage_pairs = []
    ex_passage_indices = []
    for ex in examples:
        start_idx = len(passage_pairs)
        for p in ex.passages:
            t = p.get("text", "").strip()
            if t:
                passage_pairs.append((t, ex.fact))
        ex_passage_indices.append((start_idx, len(passage_pairs)))

    # 3. 2-Sentence window formulation
    win_pairs = []
    ex_win_indices = []
    for ex in examples:
        start_idx = len(win_pairs)
        win_sents = []
        for p in ex.passages:
            sents = split_into_sentences(p.get("text", ""))
            if len(sents) <= 1:
                win_sents.extend(sents)
            else:
                win_sents.extend([f"{sents[i]} {sents[i+1]}" for i in range(len(sents) - 1)])
        fact_tokens = get_tokens_for_pruning(ex.fact) or ex.fact.lower().split()
        scored = [(w, compute_overlap_recall(fact_tokens, set(get_tokens_for_pruning(w)))) for w in win_sents]
        scored.sort(key=lambda x: x[1], reverse=True)
        passed = [w for w, rec in scored if rec >= 0.10][:3]
        if not passed and scored:
            passed = [scored[0][0]]
        for w in passed:
            win_pairs.append((w, ex.fact))
        ex_win_indices.append((start_idx, len(win_pairs)))

    # 4. Title-sentence formulation
    title_pairs = []
    ex_title_indices = []
    for ex in examples:
        start_idx = len(title_pairs)
        raw_units = []
        for p in ex.passages:
            title = p.get("title", "").strip()
            prefix = f"{title}: " if title else ""
            for s in split_into_sentences(p.get("text", "")):
                if s.strip():
                    raw_units.append(f"{prefix}{s.strip()}")
        fact_tokens = get_tokens_for_pruning(ex.fact) or ex.fact.lower().split()
        scored = [(u, compute_overlap_recall(fact_tokens, set(get_tokens_for_pruning(u)))) for u in raw_units]
        scored.sort(key=lambda x: x[1], reverse=True)
        passed = [u for u, rec in scored if rec >= 0.10][:3]
        if not passed and scored:
            passed = [scored[0][0]]
        for u in passed:
            title_pairs.append((u, ex.fact))
        ex_title_indices.append((start_idx, len(title_pairs)))

    all_unique_pairs = list(dict.fromkeys(pair for group in all_raw_pairs_by_ex for pair in group))
    cache_pairs = {
        "sweep1519": unique_pairs_gamma05,
        "unpruned_full": all_unique_pairs,
        "passage": passage_pairs,
        "window2": win_pairs,
        "title_sentence": title_pairs,
    }
    top3_profile_pairs = []
    for ex, scored in zip(examples, all_scored_by_ex):
        selected = [u for u, rec in scored if rec >= 0.10][:3]
        if not selected and scored:
            selected = [scored[0][0]]
        top3_profile_pairs.extend((u, ex.fact) for u in selected)

    manifest, benchmarks_data = ensure_task2_inference_caches(
        cache_dir, model_name, dev_path, passages_path, facts, cache_pairs,
        profile_top3_pairs=top3_profile_pairs, force_rebuild=rebuild_cache
    )

    # =========================================================================
    # STRICT CRYPTOGRAPHIC & MATHEMATICAL VALIDATION OF ALL 5 CACHES
    # =========================================================================
    print("\n[Phase 5 Task 2] Validating cryptographic provenance of all NLI inference caches...")

    cache_sweep_path = os.path.join(cache_dir, "pair_probs_sweep1519.json")
    full_cache_path = os.path.join(cache_dir, "pair_probs_unpruned_full.json")
    cache_passage_path = os.path.join(cache_dir, "pair_probs_passage.json")
    cache_win_path = os.path.join(cache_dir, "pair_probs_window2.json")
    cache_title_path = os.path.join(cache_dir, "pair_probs_title_sentence.json")

    # Validate 1: sweep1519
    ok, msg = validate_nli_cache(
        cache_sweep_path, expected_count=len(unique_pairs_gamma05),
        manifest=manifest, cache_key="sweep1519",
        expected_pairs=unique_pairs_gamma05, require_manifest=True,
        dataset_path=dev_path, passages_path=passages_path
    )
    if not ok:
        raise RuntimeError(f"Task 2 cache validation failed for sweep1519: {msg}")
    print(f"  [Task 2 Cache Verified] sweep1519: {msg}")

    # Validate 2: unpruned_full
    ok, msg = validate_nli_cache(
        full_cache_path, expected_count=len(all_unique_pairs),
        manifest=manifest, cache_key="unpruned_full",
        expected_pairs=None, expected_dict_keys={f"{a}|||{b}" for a, b in all_unique_pairs},
        require_manifest=True,
        dataset_path=dev_path, passages_path=passages_path
    )
    if not ok:
        raise RuntimeError(f"Task 2 cache validation failed for unpruned_full: {msg}")
    print(f"  [Task 2 Cache Verified] unpruned_full: {msg}")

    # Validate 3: passage
    ok, msg = validate_nli_cache(
        cache_passage_path, expected_count=len(passage_pairs),
        manifest=manifest, cache_key="passage",
        expected_pairs=passage_pairs, require_manifest=True,
        dataset_path=dev_path, passages_path=passages_path
    )
    if not ok:
        raise RuntimeError(f"Task 2 cache validation failed for passage: {msg}")
    print(f"  [Task 2 Cache Verified] passage: {msg}")

    # Validate 4: window2
    ok, msg = validate_nli_cache(
        cache_win_path, expected_count=len(win_pairs),
        manifest=manifest, cache_key="window2",
        expected_pairs=win_pairs, require_manifest=True,
        dataset_path=dev_path, passages_path=passages_path
    )
    if not ok:
        raise RuntimeError(f"Task 2 cache validation failed for window2: {msg}")
    print(f"  [Task 2 Cache Verified] window2: {msg}")

    # Validate 5: title_sentence
    ok, msg = validate_nli_cache(
        cache_title_path, expected_count=len(title_pairs),
        manifest=manifest, cache_key="title_sentence",
        expected_pairs=title_pairs, require_manifest=True,
        dataset_path=dev_path, passages_path=passages_path
    )
    if not ok:
        raise RuntimeError(f"Task 2 cache validation failed for title_sentence: {msg}")
    print(f"  [Task 2 Cache Verified] title_sentence: {msg}")

    # Load validated caches
    with open(cache_sweep_path, "r", encoding="utf-8") as f:
        pair_probs_sweep = json.load(f)
    pair_to_prob = {unique_pairs_gamma05[i]: pair_probs_sweep[i] for i in range(len(unique_pairs_gamma05))}

    with open(full_cache_path, "r", encoding="utf-8") as f:
        unpruned_probs_dict = json.load(f)

    with open(cache_passage_path, "r", encoding="utf-8") as f:
        pair_probs_passage = json.load(f)

    with open(cache_win_path, "r", encoding="utf-8") as f:
        pair_probs_win = json.load(f)

    with open(cache_title_path, "r", encoding="utf-8") as f:
        pair_probs_title = json.load(f)

    # Formulations sweep helper
    def evaluate_formulation_sweep(ex_probs_map, name, pairs_count, max_tokens):
        best_acc, best_th, best_m = -1, None, None
        for th in np.linspace(0.05, 0.95, 91):
            preds = []
            for probs in ex_probs_map:
                max_p = max(p[0] for p in probs) if probs else 0.0
                preds.append("S" if max_p >= th else "NS")
            cm = calculate_confusion_matrix(gold_labels, preds)
            m = calculate_metrics(cm)
            if m["accuracy"] > best_acc or (m["accuracy"] == best_acc and m["macro_f1"] > (best_m["macro_f1"] if best_m else 0)):
                best_acc = m["accuracy"]
                best_th = float(th)
                best_m = m
        return {
            "formulation": name,
            "pairs_evaluated": pairs_count,
            "max_token_len": max_tokens,
            "accuracy": best_acc,
            "threshold": best_th,
            "precision_s": best_m["precision_s"],
            "recall_s": best_m["recall_s"],
            "f1_s": best_m["f1_s"],
            "precision_ns": best_m["precision_ns"],
            "recall_ns": best_m["recall_ns"],
            "f1_ns": best_m["f1_ns"],
            "macro_f1": best_m["macro_f1"],
            "confusion_matrix": best_m["confusion_matrix"],
            "correct": best_m["correct"],
            "total": best_m["total"]
        }

    # Formulation 1: Full Passages
    passage_ex_probs = [pair_probs_passage[s_idx:e_idx] for s_idx, e_idx in ex_passage_indices]
    f_passage = evaluate_formulation_sweep(passage_ex_probs, "Full Passage (Baseline)", len(passage_pairs), 512)

    # Formulation 2: Unpruned sentences
    unpruned_ex_probs = []
    for pairs in all_raw_pairs_by_ex:
        p_list = [unpruned_probs_dict[f"{s}|||{h}"] for s, h in pairs if f"{s}|||{h}" in unpruned_probs_dict]
        unpruned_ex_probs.append(p_list)
    f_unpruned = evaluate_formulation_sweep(unpruned_ex_probs, "Sentence (Unpruned Baseline)", len(unpruned_probs_dict), 256)

    # Formulation 3: All-Retained (gamma=0.05)
    all_retained_ex_probs = []
    total_all_ret_instances = 0
    for ex_idx, ex in enumerate(examples):
        scored = all_scored_by_ex[ex_idx]
        passed = [s for s, rec in scored if rec >= 0.05]
        if not passed and scored:
            passed = [scored[0][0]]
        total_all_ret_instances += len(passed)
        p_list = [pair_to_prob[(s, ex.fact)] for s in passed if (s, ex.fact) in pair_to_prob]
        all_retained_ex_probs.append(p_list)
    f_all_retained = evaluate_formulation_sweep(all_retained_ex_probs, "Sentence (All-Retained, γ=0.05)", total_all_ret_instances, 256)

    # Formulation 4: Top-3 (gamma=0.10)
    top3_ex_probs = []
    total_top3_instances = 0
    for ex_idx, ex in enumerate(examples):
        scored = all_scored_by_ex[ex_idx]
        passed = [s for s, rec in scored if rec >= 0.10][:3]
        if not passed and scored:
            passed = [scored[0][0]]
        total_top3_instances += len(passed)
        p_list = [pair_to_prob[(s, ex.fact)] for s in passed if (s, ex.fact) in pair_to_prob]
        top3_ex_probs.append(p_list)
    f_top3 = evaluate_formulation_sweep(top3_ex_probs, "Sentence (Top-3, γ=0.10)", total_top3_instances, 256)

    # Formulation 5: Window-2
    win_ex_probs = [pair_probs_win[s_idx:e_idx] for s_idx, e_idx in ex_win_indices]
    f_win = evaluate_formulation_sweep(win_ex_probs, "2-Sentence Window (γ=0.10, k=3)", len(win_pairs), 256)

    # Formulation 6: Title-Sentence
    title_ex_probs = [pair_probs_title[s_idx:e_idx] for s_idx, e_idx in ex_title_indices]
    f_title = evaluate_formulation_sweep(title_ex_probs, "Title-Sentence (γ=0.10, k=3)", len(title_pairs), 256)
    title_ex_probs = [pair_probs_title[s_idx:e_idx] for s_idx, e_idx in ex_title_indices]
    f_title = evaluate_formulation_sweep(title_ex_probs, "Title-Sentence (γ=0.10, k=3)", len(title_pairs), 256)

    formulation_summary = [f_passage, f_unpruned, f_all_retained, f_top3, f_win, f_title]

    # Decision strategies
    argmax_preds_all = []
    for p_list in all_retained_ex_probs:
        is_s = any(p[0] > p[1] and p[0] > p[2] for p in p_list) if p_list else False
        argmax_preds_all.append("S" if is_s else "NS")
    cm_argmax_all = calculate_confusion_matrix(gold_labels, argmax_preds_all)
    m_argmax_all = calculate_metrics(cm_argmax_all)

    th_curve_data = []
    opt_th = None
    best_p_acc = -1
    best_p_m = None
    for th in np.linspace(0.05, 0.95, 91):
        preds = []
        for p_list in all_retained_ex_probs:
            max_ent = max(p[0] for p in p_list) if p_list else 0.0
            preds.append("S" if max_ent >= th else "NS")
        cm = calculate_confusion_matrix(gold_labels, preds)
        m = calculate_metrics(cm)
        th_curve_data.append({
            "threshold": float(th),
            "accuracy": m["accuracy"],
            "f1_s": m["f1_s"],
            "f1_ns": m["f1_ns"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_accuracy"]
        })
        if m["accuracy"] > best_p_acc or (m["accuracy"] == best_p_acc and m["macro_f1"] > (best_p_m["macro_f1"] if best_p_m else 0)):
            best_p_acc = m["accuracy"]
            opt_th = float(th)
            best_p_m = m

    best_m_acc = -1
    best_m_th = None
    best_m_m = None
    for th in np.linspace(-0.80, 0.80, 161):
        preds = []
        for p_list in all_retained_ex_probs:
            max_mar = max(p[0] - p[2] for p in p_list) if p_list else -1.0
            preds.append("S" if max_mar >= th else "NS")
        cm = calculate_confusion_matrix(gold_labels, preds)
        m = calculate_metrics(cm)
        if m["accuracy"] > best_m_acc or (m["accuracy"] == best_m_acc and m["macro_f1"] > (best_m_m["macro_f1"] if best_m_m else 0)):
            best_m_acc = m["accuracy"]
            best_m_th = float(th)
            best_m_m = m

    strategy_records = [
        {"name": "Discrete Argmax\n(p_ent > p_neu, p_contra)", "accuracy": m_argmax_all["accuracy"], "f1_s": m_argmax_all["f1_s"], "f1_ns": m_argmax_all["f1_ns"], "macro_f1": m_argmax_all["macro_f1"], "metrics": m_argmax_all},
        {"name": f"Probability Threshold\n(p_ent ≥ {opt_th:.2f})", "accuracy": best_p_acc, "f1_s": best_p_m["f1_s"], "f1_ns": best_p_m["f1_ns"], "macro_f1": best_p_m["macro_f1"], "metrics": best_p_m},
        {"name": f"Margin Threshold\n(p_ent - p_contra ≥ {best_m_th:.2f})", "accuracy": best_m_acc, "f1_s": best_m_m["f1_s"], "f1_ns": best_m_m["f1_ns"], "macro_f1": best_m_m["macro_f1"], "metrics": best_m_m}
    ]

    # Pruning sweep (60 configs)
    gammas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    top_ks = [1, 2, 3, 5, None]
    fallbacks = ["conservative", "strict"]

    pruning_records = []
    config_scores_map = {}

    for fb in fallbacks:
        for g in gammas:
            for k in top_ks:
                cfg_key = f"g={g:.2f}_k={k if k is not None else 'all'}_{fb}"
                ex_probs_list = []
                max_ent_scores = []
                total_eval_pairs = 0

                for ex_idx in range(len(examples)):
                    scored = all_scored_by_ex[ex_idx]
                    passed = [s for s, rec in scored if rec >= g]
                    if passed:
                        if k is not None:
                            passed = passed[:k]
                    elif fb == "conservative" and scored:
                        passed = [scored[0][0]]
                    else:
                        passed = []

                    total_eval_pairs += len(passed)
                    p_list = [pair_to_prob[(s, facts[ex_idx])] for s in passed if (s, facts[ex_idx]) in pair_to_prob]
                    ex_probs_list.append(p_list)
                    max_p = max(p[0] for p in p_list) if p_list else 0.0
                    max_ent_scores.append(max_p)

                config_scores_map[cfg_key] = max_ent_scores

                argmax_preds = []
                for p_list in ex_probs_list:
                    is_s = any(p[0] > p[1] and p[0] > p[2] for p in p_list) if p_list else False
                    argmax_preds.append("S" if is_s else "NS")
                cm_arg = calculate_confusion_matrix(gold_labels, argmax_preds)
                m_arg = calculate_metrics(cm_arg)

                best_p_acc_g, best_p_th_g = -1, None
                for th in np.linspace(0.05, 0.95, 91):
                    preds = ["S" if sc >= th else "NS" for sc in max_ent_scores]
                    cm = calculate_confusion_matrix(gold_labels, preds)
                    m = calculate_metrics(cm)
                    if m["accuracy"] > best_p_acc_g:
                        best_p_acc_g = m["accuracy"]
                        best_p_th_g = float(th)

                best_m_acc_g, best_m_th_g = -1, None
                for th in np.linspace(-0.80, 0.80, 161):
                    preds = []
                    for p_list in ex_probs_list:
                        max_mar = max(p[0] - p[2] for p in p_list) if p_list else -1.0
                        preds.append("S" if max_mar >= th else "NS")
                    cm = calculate_confusion_matrix(gold_labels, preds)
                    m = calculate_metrics(cm)
                    if m["accuracy"] > best_m_acc_g:
                        best_m_acc_g = m["accuracy"]
                        best_m_th_g = float(th)

                pruning_records.append({
                    "config_key": cfg_key,
                    "gamma": g,
                    "top_k": k,
                    "fallback": fb,
                    "pairs": total_eval_pairs,
                    "argmax_accuracy": m_arg["accuracy"],
                    "best_prob_threshold": best_p_th_g,
                    "best_prob_accuracy": best_p_acc_g,
                    "best_margin_threshold": best_m_th_g,
                    "best_margin_accuracy": best_m_acc_g
                })

    # Cross-validation
    all_ret_scores = [max(p[0] for p in p_list) if p_list else 0.0 for p_list in all_retained_ex_probs]
    top3_scores = [max(p[0] for p in p_list) if p_list else 0.0 for p_list in top3_ex_probs]
    unpruned_scores = [max(p[0] for p in p_list) if p_list else 0.0 for p_list in unpruned_ex_probs]

    cv_all_retained = run_threshold_only_cv(all_ret_scores, gold_labels, folds, idx_to_topic)
    cv_top3 = run_threshold_only_cv(top3_scores, gold_labels, folds, idx_to_topic)
    cv_unpruned = run_threshold_only_cv(unpruned_scores, gold_labels, folds, idx_to_topic)

    # Nested CV
    nested_oof_preds = [None] * total_facts
    nested_fold_records = []

    for f_idx in range(len(folds)):
        val_topics = set(folds[f_idx])
        val_indices = [i for i in range(total_facts) if idx_to_topic[i] in val_topics]
        train_indices = [i for i in range(total_facts) if idx_to_topic[i] not in val_topics]

        best_cfg_key = None
        best_th = None
        best_tr_acc = -1

        for r in pruning_records:
            c_key = r["config_key"]
            scores = config_scores_map[c_key]
            for th in np.linspace(0.10, 0.90, 81):
                tr_acc = sum(1 for i in train_indices if (gold_labels[i] == "S") == (scores[i] >= th)) / len(train_indices)
                if tr_acc > best_tr_acc:
                    best_tr_acc = tr_acc
                    best_cfg_key = c_key
                    best_th = float(th)

        val_scores = config_scores_map[best_cfg_key]
        val_correct = sum(1 for i in val_indices if (gold_labels[i] == "S") == (val_scores[i] >= best_th))
        val_acc = val_correct / len(val_indices)
        for i in val_indices:
            nested_oof_preds[i] = "S" if val_scores[i] >= best_th else "NS"

        cm_val_n = calculate_confusion_matrix([gold_labels[i] for i in val_indices], [nested_oof_preds[i] for i in val_indices])
        m_val_n = calculate_metrics(cm_val_n)

        nested_fold_records.append({
            "fold": f_idx,
            "val_topics": sorted(list(val_topics)),
            "val_count": len(val_indices),
            "train_count": len(train_indices),
            "selected_config": best_cfg_key,
            "selected_threshold": best_th,
            "train_accuracy": float(best_tr_acc),
            "val_accuracy": float(val_acc),
            "val_correct": val_correct,
            "val_total": len(val_indices),
            "metrics": m_val_n
        })

    cm_oof_nested = calculate_confusion_matrix(gold_labels, nested_oof_preds)
    m_oof_nested = calculate_metrics(cm_oof_nested)
    mean_val_nested = float(np.mean([f["val_accuracy"] for f in nested_fold_records]))
    std_val_nested = float(np.std([f["val_accuracy"] for f in nested_fold_records]))

    # Winning predictions
    winning_preds = []
    winning_prediction_records = []

    for i, ex in enumerate(examples):
        scored = all_scored_by_ex[i]
        passed = [s for s, rec in scored if rec >= 0.05]
        if not passed and scored:
            passed = [scored[0][0]]

        p_list = all_retained_ex_probs[i]
        max_idx = int(np.argmax([p[0] for p in p_list])) if p_list else 0
        best_p_ent = float(p_list[max_idx][0]) if p_list else 0.0
        best_premise = passed[max_idx] if max_idx < len(passed) else ""

        pred = "S" if best_p_ent >= opt_th else "NS"
        winning_preds.append(pred)

        winning_prediction_records.append({
            "fact_index": i,
            "fact": ex.fact,
            "topic": idx_to_topic.get(i, ""),
            "gold_label": gold_labels[i],
            "predicted_label": pred,
            "is_correct": bool(pred == gold_labels[i]),
            "max_entailment_probability": best_p_ent,
            "threshold_used": opt_th,
            "retained_candidate_sentences_count": len(passed),
            "selected_premise_sentence": best_premise
        })

    # Baselines & Task 1 for comparison
    s_cnt = sum(1 for y in gold_labels if y == "S")
    b_always = {
        "accuracy": s_cnt / total_facts,
        "f1_s": (2 * s_cnt) / (total_facts + s_cnt),
        "f1_ns": None,
        "macro_f1": None
    }
    rnd_checker = RandomGuessFactChecker(seed=42)
    b_rand, _ = evaluate_fact_checker_standalone(examples, rnd_checker)

    task1_path = os.path.join(os.path.dirname(outputs_dir), "task1", "experiments.json")
    if not os.path.exists(task1_path):
        task1_path = os.path.join("outputs", "task1", "experiments.json")
    if not os.path.exists(task1_path):
        task1_path = os.path.join("outputs", "task1", "results.json")

    t1_m = None
    t1_oof_acc = None
    if os.path.exists(task1_path):
        with open(task1_path, "r", encoding="utf-8") as f:
            t1_json = json.load(f)
            if "winning_configuration" in t1_json:
                t1_m = t1_json["winning_configuration"].get("metrics")
            if "nested_grouped_5fold_cross_validation" in t1_json:
                t1_oof_acc = t1_json["nested_grouped_5fold_cross_validation"].get("overall_oof_accuracy")

    full_bench = benchmarks_data.get("full_dataset_benchmarks", {})
    t3_bench = full_bench.get("top3_pruned", {})
    all_bench = full_bench.get("all_retained", {})
    unp_bench = full_bench.get("unpruned_full", {})

    task2_experiments_payload = {
        "metadata": {
            "model_name": model_name,
            "device": "CPU",
            "torch_version": torch.__version__,
            "total_facts_evaluated": total_facts,
            "total_biography_topics": len(raw_bios),
            "total_raw_extracted_sentence_instances": total_raw_sentences,
            "unique_inference_pairs_evaluated": len(unpruned_probs_dict) if unpruned_probs_dict else total_raw_sentences,
            "all_retained_evaluated_instances": total_all_ret_instances,
            "all_retained_unique_pairs": total_unique_gamma05,
            "top3_evaluated_instances": total_top3_instances,
            "top3_unique_pairs": total_top3_instances
        },
        "group_a_formulations": formulation_summary,
        "group_b_decision_strategies": strategy_records,
        "group_c_lexical_pruning_sweep": pruning_records,
        "group_d_computational_profiling": {
            "pilot_batch_size_benchmarks": benchmarks_data.get("batch_size_pilot_benchmarks", []),
            "measured_full_dataset_runtimes": {
                "top3_pruned": t3_bench,
                "all_retained": all_bench,
                "unpruned_full": unp_bench
            }
        },
        "candidate_selection_comparison": {
            "all_retained_gamma005": {
                "dev_accuracy": f_all_retained["accuracy"],
                "threshold_only_cv_pooled_oof_accuracy": cv_all_retained["pooled_oof_accuracy"],
                "threshold_only_cv_mean_val_accuracy": cv_all_retained["mean_val_accuracy"],
                "threshold_only_cv_std_val_accuracy": cv_all_retained["std_val_accuracy"],
                "threshold_only_cv_overfitting_gap": cv_all_retained["overfitting_gap"],
                "retained_candidate_instances": total_all_ret_instances,
                "unique_inference_pairs": total_unique_gamma05,
                "measured_runtime_seconds": all_bench.get("wall_clock_seconds", "Not Available")
            },
            "top3_gamma010": {
                "dev_accuracy": f_top3["accuracy"],
                "threshold_only_cv_pooled_oof_accuracy": cv_top3["pooled_oof_accuracy"],
                "threshold_only_cv_mean_val_accuracy": cv_top3["mean_val_accuracy"],
                "threshold_only_cv_std_val_accuracy": cv_top3["std_val_accuracy"],
                "threshold_only_cv_overfitting_gap": cv_top3["overfitting_gap"],
                "retained_candidate_instances": total_top3_instances,
                "unique_inference_pairs": total_top3_instances,
                "measured_runtime_seconds": t3_bench.get("wall_clock_seconds", "Not Available")
            },
            "unpruned_baseline": {
                "dev_accuracy": f_unpruned["accuracy"],
                "threshold_only_cv_pooled_oof_accuracy": cv_unpruned["pooled_oof_accuracy"],
                "threshold_only_cv_mean_val_accuracy": cv_unpruned["mean_val_accuracy"],
                "threshold_only_cv_std_val_accuracy": cv_unpruned["std_val_accuracy"],
                "threshold_only_cv_overfitting_gap": cv_unpruned["overfitting_gap"],
                "retained_candidate_instances": total_raw_sentences,
                "unique_inference_pairs": len(unpruned_probs_dict) if unpruned_probs_dict else total_unique_unpruned,
                "measured_runtime_seconds": unp_bench.get("wall_clock_seconds", "Not Available")
            }
        },
        "winning_classifier": {
            "formulation": "sentence",
            "pruning_gamma": 0.05,
            "top_k": None,
            "fallback": "conservative",
            "decision_rule": "threshold",
            "optimal_threshold": opt_th,
            "dev_metrics": best_p_m,
            "threshold_only_cv_results": cv_all_retained
        },
        "grouped_5fold_cross_validation": {
            "description": "Threshold-only grouped 5-fold cross-validation on fixed winning All-Retained configuration (gamma=0.05, k=all)",
            "folds": cv_all_retained["folds"],
            "mean_val_accuracy": cv_all_retained["mean_val_accuracy"],
            "std_val_accuracy": cv_all_retained["std_val_accuracy"],
            "mean_train_accuracy": cv_all_retained["mean_train_accuracy"],
            "overfitting_gap": cv_all_retained["overfitting_gap"],
            "overall_oof_accuracy": cv_all_retained["pooled_oof_accuracy"],
            "overall_oof_metrics": cv_all_retained["pooled_oof_metrics"]
        },
        "nested_grouped_5fold_cross_validation": {
            "description": "Searches 60 lexical-pruning configurations strictly within single-sentence formulation.",
            "folds": nested_fold_records,
            "mean_val_accuracy": mean_val_nested,
            "std_val_accuracy": std_val_nested,
            "overall_oof_accuracy": m_oof_nested["accuracy"],
            "overall_oof_metrics": m_oof_nested
        },
        "threshold_calibration_curve": th_curve_data,
        "predictions": winning_prediction_records
    }

    t2_exp_path = os.path.join(outputs_dir, "experiments.json")
    with open(t2_exp_path, "w", encoding="utf-8") as f:
        json.dump(task2_experiments_payload, f, indent=2)

    task2_results_payload = {
        "model": model_name,
        "winning_config": {
            "formulation": "sentence",
            "pruning_gamma": 0.05,
            "top_k": None,
            "fallback": "conservative",
            "decision_rule": "threshold",
            "optimal_threshold": opt_th,
            "accuracy": best_p_m["accuracy"],
            "correct": best_p_m["correct"],
            "total": best_p_m["total"],
            "precision_s": best_p_m["precision_s"],
            "recall_s": best_p_m["recall_s"],
            "f1_s": best_p_m["f1_s"],
            "precision_ns": best_p_m["precision_ns"],
            "recall_ns": best_p_m["recall_ns"],
            "f1_ns": best_p_m["f1_ns"],
            "balanced_accuracy": best_p_m["balanced_accuracy"],
            "macro_f1": best_p_m["macro_f1"],
            "confusion_matrix": best_p_m["confusion_matrix"]
        },
        "candidate_comparison": task2_experiments_payload["candidate_selection_comparison"],
        "grouped_cv": cv_all_retained,
        "nested_grouped_cv": {
            "mean_val_accuracy": mean_val_nested,
            "std_val_accuracy": std_val_nested,
            "overall_oof_accuracy": m_oof_nested["accuracy"],
            "overall_oof_metrics": m_oof_nested
        },
        "predictions": winning_prediction_records
    }

    t2_res_path = os.path.join(outputs_dir, "results.json")
    with open(t2_res_path, "w", encoding="utf-8") as f:
        json.dump(task2_results_payload, f, indent=2)

    # CSV Exports
    csv_preds_path = os.path.join(outputs_dir, "predictions.csv")
    with open(csv_preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fact_index", "topic", "fact", "gold_label", "predicted_label", "is_correct", "max_entailment_probability", "threshold_used", "retained_candidate_sentences_count", "selected_premise_sentence"])
        for r in winning_prediction_records:
            writer.writerow([r["fact_index"], r["topic"], r["fact"], r["gold_label"], r["predicted_label"], r["is_correct"], f"{r['max_entailment_probability']:.4f}", f"{r['threshold_used']:.2f}", r["retained_candidate_sentences_count"], r["selected_premise_sentence"]])

    csv_comp_path = os.path.join(outputs_dir, "comparisons.csv")
    with open(csv_comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["formulation", "pairs_evaluated", "accuracy", "f1_s", "f1_ns", "macro_f1"])
        for form in formulation_summary:
            writer.writerow([form["formulation"], form["pairs_evaluated"], f"{form['accuracy']*100:.2f}%", f"{form['f1_s']:.4f}", f"{form['f1_ns']:.4f}", f"{form['macro_f1']:.4f}"])

    # High-DPI Figures
    os.makedirs(figures_dir, exist_ok=True)
    fig1 = os.path.join(figures_dir, "task2_strategy_comparison.png")
    fig2 = os.path.join(figures_dir, "task2_pruning_tradeoff.png")
    fig3 = os.path.join(figures_dir, "task2_runtime_comparison.png")
    fig4 = os.path.join(figures_dir, "task2_threshold_calibration_curve.png")
    fig5 = os.path.join(figures_dir, "task2_validation_summary.png")
    fig6 = os.path.join(figures_dir, "task2_final_confusion_matrix.png")
    fig7 = os.path.join(figures_dir, "task2_final_results_table.png")

    generate_task2_strategy_comparison(strategy_records, target_acc=0.83, output_path=fig1)
    generate_task2_pruning_tradeoff(pruning_records, output_path=fig2)

    runtime_plot_data = [
        {"label": f"B={b['batch_size']}\nPilot ({b['pairs']}p)", "seconds": b["seconds"]}
        for b in benchmarks_data.get("batch_size_pilot_benchmarks", [])
        if isinstance(b.get("seconds"), (int, float))
    ]
    if isinstance(t3_bench.get("wall_clock_seconds"), (int, float)):
        runtime_plot_data.append({"label": f"Top-3 ({total_top3_instances}p)\nB=8 Measured", "seconds": t3_bench["wall_clock_seconds"]})
    if isinstance(all_bench.get("wall_clock_seconds"), (int, float)):
        runtime_plot_data.append({"label": f"All-Ret ({total_all_ret_instances}p)\nB=8 Measured", "seconds": all_bench["wall_clock_seconds"]})
    if isinstance(unp_bench.get("wall_clock_seconds"), (int, float)):
        runtime_plot_data.append({"label": f"Unpruned ({total_raw_sentences}p)\nB=8 Measured", "seconds": unp_bench["wall_clock_seconds"]})
    generate_task2_runtime_comparison(runtime_plot_data, output_path=fig3)

    generate_task2_threshold_curve(th_curve_data, opt_th=opt_th, output_path=fig4)
    generate_task2_validation_summary(cv_all_retained, cv_top3, cv_unpruned, output_path=fig5)
    generate_task2_confusion_matrix(best_p_m["confusion_matrix"], opt_th=opt_th, output_path=fig6)
    generate_task2_final_results_table(
        b_always=b_always, b_rand=b_rand,
        t1_m=t1_m, t1_oof_acc=t1_oof_acc,
        t2_m=best_p_m, t2_oof_acc=cv_all_retained["pooled_oof_accuracy"],
        f_passage=f_passage, f_unpruned=f_unpruned,
        f_win=f_win, f_title=f_title,
        unpruned_oof_acc=cv_unpruned.get("pooled_oof_accuracy"),
        output_path=fig7,
        t1_pairs_count=sum(len(ex.passages) for ex in examples),
        t2_pairs_count=total_unique_gamma05
    )

    elapsed = time.time() - start_time
    print(f"\n[Phase 5 Task 2] Completed in {elapsed:.2f}s.")
    print("=" * 80)


# ==============================================================================
# CLI EXECUTION DISPATCHER
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Fact-Trace-AI Coursework Experimental Suite")
    parser.add_argument("--baselines", action="store_true", help="Run Phase 2 Baseline Benchmarks")
    parser.add_argument("--task1", action="store_true", help="Run Phase 3 Task 1 Word Overlap Experiments")
    parser.add_argument("--task2", action="store_true", help="Run Phase 5 Task 2 DeBERTa-v3 Textual Entailment Experiments")
    parser.add_argument("--all", action="store_true", help="Run complete experimental suite (Baselines + Task 1 + Task 2)")
    parser.add_argument("--test-cache", action="store_true", help="Run 15 cache integrity negative tests")
    parser.add_argument("--rebuild-cache", action="store_true", help="Run fresh DeBERTa inference to replace NLI caches")

    args = parser.parse_args()

    if args.test_cache:
        print("[Cache Integrity Test] Running negative validation tests...")
        res = run_cache_validation_negative_tests()
        total_t = res.get('total_tests', 0)
        passed_t = res.get('passed_count', 0)
        print(f"Total tests: {total_t}")
        print(f"Passed (rejected invalid cache): {passed_t}")
        for t_name, t_info in res.get("tests", {}).items():
            status = "PASSED (REJECTED)" if t_info.get("rejected") else "FAILED"
            reason = t_info.get("rejection_reason") or t_info.get("message")
            print(f"  - {t_name}: {status} -> {reason}")
        if res.get("all_passed"):
            print(f"[SUCCESS] All {total_t} cache negative tests passed successfully.")
        else:
            print("[FAILURE] Some cache negative tests failed.")
            sys.exit(1)
    elif args.all:
        print("[Experimental Suite] Running all benchmarks...")
        run_baselines()
        run_task1()
        run_task2(rebuild_cache=args.rebuild_cache)
    elif args.baselines:
        run_baselines()
    elif args.task1:
        run_task1()
    elif args.task2:
        run_task2(rebuild_cache=args.rebuild_cache)
    else:
        print("Usage:")
        print("  python experiments.py --baselines   (Phase 2 Baselines)")
        print("  python experiments.py --task1       (Phase 3 Task 1 Word Overlap)")
        print("  python experiments.py --task2       (Auto-generate missing caches with live inference)")
        print("  python experiments.py --task2 --rebuild-cache  (Force fresh NLI inference)")
        print("  python experiments.py --all         (Run all benchmark experiments)")
        print("  python experiments.py --test-cache  (Verify negative cache validation tests)")

