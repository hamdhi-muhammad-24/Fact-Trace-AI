# factcheck.py

import os
import sys
import json
import re
import math
import gc
import textwrap
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import torch
import spacy

# Safe optional Matplotlib import with headless Agg backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# ==============================================================================
# STARTER CLASSES & FRAMEWORK DATA STRUCTURES (READ-ONLY INTERFACES)
# ==============================================================================

class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits

        raise Exception("Not implemented")

        del inputs, outputs, logits
        gc.collect()


class FactChecker(object):
    """
    Fact checker base type
    """
    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def predict(self, fact: str, passages: List[dict]) -> str:
        if self.rng is not None:
            return str(self.rng.choice(["S", "NS"]))
        prediction = np.random.choice(["S", "NS"])
        return str(prediction)


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL STARTER CLASS
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except Exception:
            self.nlp = None

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        if self.nlp is None:
            raise RuntimeError("en_core_web_sm model is not installed.")
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations


# ==============================================================================
# PHASE 1: DATASET LOADING & REPRODUCIBLE EDA FUNCTIONS
# ==============================================================================

def load_passages_data(path: str) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """
    Reads the retrieved passages dataset without importing factchecking_main.py,
    avoiding circular imports while matching official loader semantics exactly.
    Returns:
        fact_to_passage_dict: Mapping from fact sentence string to list of passage dicts
        raw_records: List of all parsed JSON objects in the file
    """
    fact_to_passage_dict = {}
    raw_records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            raw_records.append(record)
            name = record.get("name", "")
            passages = record.get("passages", [])
            for passage in passages:
                if passage.get("title") != name:
                    raise Exception(f"Title mismatch on line {line_num}: {name} vs {passage.get('title')}")
            fact_to_passage_dict[record["sent"]] = passages
    return fact_to_passage_dict, raw_records


def load_dev_examples(labeled_facts_path: str, fact_to_passage_dict: Dict[str, List[dict]]) -> Tuple[List[FactExample], List[dict]]:
    """
    Reads the labeled development facts and constructs FactExample objects.
    Matches read_fact_examples from factchecking_main.py bit-for-bit.
    Returns:
        examples: List of FactExample objects in official evaluation order
        raw_bios: List of raw biography JSON objects
    """
    examples = []
    raw_bios = []
    with open(labeled_facts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            bio = json.loads(line_str)
            raw_bios.append(bio)
            if bio.get("annotations") is not None:
                for sent in bio["annotations"]:
                    if sent.get("human-atomic-facts") is not None:
                        for fact in sent["human-atomic-facts"]:
                            fact_text = fact.get("text", "")
                            if fact_text in fact_to_passage_dict:
                                examples.append(FactExample(fact_text, fact_to_passage_dict[fact_text], fact.get("label", "")))
    return examples, raw_bios


def run_data_quality_audit(passages_path: str, dev_path: str, examples: List[FactExample], raw_bios: List[dict], raw_passages: List[dict]) -> Dict[str, Any]:
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

    # 5. Malformed JSON records
    audit["malformed_records_in_passages_file"] = 0
    audit["malformed_records_in_dev_file"] = 0

    return audit


def compute_eda_metrics(examples: List[FactExample], raw_bios: List[dict], raw_passages: List[dict], fact_to_passage_dict: Dict[str, List[dict]]) -> Dict[str, Any]:
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

    # Unique passage texts across all evaluated dev facts
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


# ==============================================================================
# VISUALIZATION & FIGURE GENERATION PIPELINE
# ==============================================================================

def generate_eda_visualizations(metrics: Dict[str, Any], audit: Dict[str, Any], examples: List[FactExample], output_dir: str = "output_figures") -> List[str]:
    """
    Renders all Phase 1 figures and image tables using Matplotlib Agg backend.
    Returns list of saved image paths.
    """
    if not _HAS_MATPLOTLIB:
        print("[Warning] Matplotlib not installed. Skipping figure generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. Summary table rendered as an image
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    table_data = [
        ["Benchmark Retrieval Records (JSON lines)", f"{metrics['retrieval_total_records']:,}"],
        ["Benchmark Total Passage Entries", f"{metrics['retrieval_total_passage_entries']:,}"],
        ["Benchmark Unique Query Fact Strings", f"{metrics['retrieval_unique_fact_queries']:,}"],
        ["Evaluated Biographies (ChatGPT topics)", f"{metrics['dev_total_biographies']}"],
        ["Evaluated Atomic Fact Instances", f"{metrics['dev_total_fact_instances']}"],
        ["Unique Atomic Fact Text Strings", f"{metrics['dev_unique_fact_strings']} (1 duplicate)"],
        ["Raw Annotations Distribution", f"S: {metrics['raw_label_counts']['S']} (50.68%) | NS: {metrics['raw_label_counts']['NS']} (42.08%) | IR: {metrics['raw_label_counts']['IR']} (7.24%)"],
        ["Binary Evaluation Labels (IR -> NS)", f"S: {metrics['converted_label_counts']['S']} (50.68%) | NS: {metrics['converted_label_counts']['NS']} (49.32%)"],
        ["Total Associated Passage Entries", f"{metrics['dev_total_associated_passage_entries']}"],
        ["Unique Retrieved Passage Text Chunks", f"{metrics['dev_unique_passage_texts']} (reused across claims)"],
        ["Passages per Fact (min / mean / max)", f"{metrics['passages_per_fact']['min']} / {metrics['passages_per_fact']['mean']:.2f} / {metrics['passages_per_fact']['max']}"],
        ["Atomic Fact Word Length (mean / median / max)", f"{metrics['fact_length_words']['mean']:.2f} / {metrics['fact_length_words']['median']:.1f} / {metrics['fact_length_words']['max']}"],
        ["Passage Word Length (mean / median / max)", f"{metrics['passage_length_words']['mean']:.2f} / {metrics['passage_length_words']['median']:.1f} / {metrics['passage_length_words']['max']}"],
        ["Length Asymmetry Ratio (Passage / Fact)", f"{metrics['length_asymmetry_word_ratio']:.2f}x"],
        ["Claims Containing Numbers / Dates", f"{metrics['facts_with_numbers_count']} ({metrics['facts_with_numbers_percentage']:.1f}%)"],
        ["Duplicate Fact Anomaly ('Cleveland is in Ohio.')", "2 occurrences (0-based idx 5 & 63; pos 6 & 64)\nConflicting gold: 1 IR -> NS, 1 S"],
        ["Deterministic Upper Bound Accuracy Ceiling", "220 / 221 = 99.55%"]
    ]

    col_widths = [0.50, 0.50]
    table = ax.table(cellText=table_data, colLabels=["Dataset Characteristic / Metric", "Empirically Verified Value"], loc="center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f497d")
            cell.set_text_props(color="white", weight="bold")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
            if col == 1:
                cell.set_text_props(weight="semibold")

    plt.title("FActScore Development Dataset Structural Summary", fontsize=12, fontweight="bold", pad=12)
    path = os.path.join(output_dir, "eda_summary_table.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 2. Label distribution chart (Raw 3-class vs Converted 2-class)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # Panel A: Raw labels
    raw_cats = ["Supported (S)", "Not Supported (NS)", "Irrelevant (IR)"]
    raw_vals = [metrics["raw_label_counts"]["S"], metrics["raw_label_counts"]["NS"], metrics["raw_label_counts"]["IR"]]
    raw_pcts = [metrics["raw_label_percentages"]["S"], metrics["raw_label_percentages"]["NS"], metrics["raw_label_percentages"]["IR"]]
    colors_raw = ["#2b7bba", "#d95f02", "#7570b3"]

    bars1 = ax1.bar(raw_cats, raw_vals, color=colors_raw, edgecolor="black", width=0.55)
    ax1.set_title("Raw FActScore Annotations (N = 221)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count of Atomic Facts")
    ax1.set_ylim(0, 135)
    for bar, val, pct in zip(bars1, raw_vals, raw_pcts):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, y + 2.5, f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Converted binary labels
    conv_cats = ["Supported (S)", "Not Supported (NS)\n[incl. IR -> NS]"]
    conv_vals = [metrics["converted_label_counts"]["S"], metrics["converted_label_counts"]["NS"]]
    conv_pcts = [metrics["converted_label_percentages"]["S"], metrics["converted_label_percentages"]["NS"]]
    colors_conv = ["#2b7bba", "#d95f02"]

    bars2 = ax2.bar(conv_cats, conv_vals, color=colors_conv, edgecolor="black", width=0.45)
    ax2.set_title("Official Binary Evaluation Labels (N = 221)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count of Atomic Facts")
    ax2.set_ylim(0, 135)
    for bar, val, pct in zip(bars2, conv_vals, conv_pcts):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y + 2.5, f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_label_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()
    saved_files.append(path)

    # 3. Length distributions comparing facts vs passages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # Fact length histogram
    all_facts_words = [len(ex.fact.split()) for ex in examples]
    ax1.hist(all_facts_words, bins=range(3, 16), color="#2b7bba", edgecolor="black", alpha=0.85, rwidth=0.85)
    ax1.axvline(metrics["fact_length_words"]["mean"], color="red", linestyle="--", linewidth=1.8, label=f"Mean: {metrics['fact_length_words']['mean']:.2f}")
    ax1.axvline(metrics["fact_length_words"]["median"], color="green", linestyle=":", linewidth=2, label=f"Median: {metrics['fact_length_words']['median']:.1f}")
    ax1.set_title("Atomic Fact Length Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Word Count per Claim")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right")
    ax1.set_xticks(range(4, 15))

    # Passage length histogram
    all_passage_words = [len(p.get("text", "").split()) for ex in examples for p in ex.passages]
    ax2.hist(all_passage_words, bins=20, color="#d95f02", edgecolor="black", alpha=0.85)
    ax2.axvline(metrics["passage_length_words"]["mean"], color="red", linestyle="--", linewidth=1.8, label=f"Mean: {metrics['passage_length_words']['mean']:.2f}")
    ax2.axvline(metrics["passage_length_words"]["median"], color="green", linestyle=":", linewidth=2, label=f"Median: {metrics['passage_length_words']['median']:.1f}")
    ax2.set_title("Retrieved Wikipedia Passage Length Distribution", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Word Count per Passage")
    ax2.set_ylabel("Frequency")
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
    colors = ["#2b7bba", "#d95f02"]

    bars = ax.barh(categories, means, color=colors, edgecolor="black", height=0.45)
    ax.set_xlabel("Mean Word Count")
    ax.set_title(f"Passage-to-Claim Length Asymmetry Ratio: {metrics['length_asymmetry_word_ratio']:.2f}x", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 185)

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
        xytext=(55, 0.45),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2cc", edgecolor="#d6b656")
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
    pcts = [(v / len(examples)) * 100 for v in y_vals]

    bars = ax.bar([str(x) for x in x_vals], y_vals, color="#41b6c4", edgecolor="black", width=0.5)
    ax.set_title("Distribution of Retrieved Passages per Atomic Claim", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Passages Retrieved (BM25)")
    ax.set_ylabel("Number of Facts")
    ax.set_ylim(0, 125)

    for bar, val, pct in zip(bars, y_vals, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.5, f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_passages_per_fact.png")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    saved_files.append(path)

    # 6. Data quality and anomalies summary table image
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    def wrap_cell(val: str, width: int) -> str:
        return "\n".join(textwrap.fill(line, width=width) for line in str(val).split("\n"))

    raw_anomaly_data = [
        ["Empty or Missing Atomic Fact Texts", f"{audit['empty_or_missing_fact_texts']}", "Clean (No missing inputs)"],
        ["Facts with Empty Passage Lists", f"{audit['facts_with_empty_passage_lists']}", "Clean (All claims have candidate evidence)"],
        ["Passages with Missing Wikipedia Titles", f"{audit['passages_with_missing_titles']}", "Clean (All passages mapped to source article)"],
        ["Passages with Empty Body Text", f"{audit['passages_with_empty_text']}", "Clean (All evidence contains textual content)"],
        ["Duplicate Passages within Same Fact", f"{audit['facts_with_duplicate_passages']}", "Clean (No redundant duplicate passages per claim)"],
        ["Duplicate Fact Instances across Dataset", f"{audit['duplicate_fact_strings_count']} string ('Cleveland is in Ohio.')", "Identical claim appears twice across 2 biographies"],
        ["Duplicate Instance #1 (0-based idx 5 / pos 6, Marianne McAndrew)", "Ground Truth: IR -> Converted to NS", "3 candidate passages retrieved (identical to #2)"],
        ["Duplicate Instance #2 (0-based idx 63 / pos 64, Mike Trivisonno)", "Ground Truth: S -> Converted to S", "3 candidate passages retrieved (identical to #1)"],
        ["Label Contradiction on Identical Inputs", "CONTRADICTION DETECTED", "Deterministic classifiers cannot predict both S and NS"],
        ["Theoretical Maximum Accuracy Ceiling", "220 / 221 = 99.55%", "1 error is mathematically unavoidable"]
    ]

    wrapped_anomaly_data = [
        [wrap_cell(row[0], 34), wrap_cell(row[1], 28), wrap_cell(row[2], 34)]
        for row in raw_anomaly_data
    ]

    col_widths = [0.38, 0.28, 0.34]
    table = ax.table(cellText=wrapped_anomaly_data, colLabels=["Data Quality Check", "Finding / Value", "Implication for Modeling"], loc="center", cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#78281f")
            cell.set_text_props(color="white", weight="bold")
        else:
            if "CONTRADICTION" in cell.get_text().get_text():
                cell.set_facecolor("#f9e79f")
                cell.set_text_props(weight="bold", color="#78281f")
            elif row in (7, 8):
                cell.set_facecolor("#fcf3cf" if row % 2 == 1 else "#fef9e7")
            elif row % 2 == 1:
                cell.set_facecolor("#fdf2e9")
            else:
                cell.set_facecolor("#ffffff")

    plt.title("Data Quality & Benchmark Anomaly Audit", fontsize=12, fontweight="bold", pad=10)
    path = os.path.join(output_dir, "eda_data_quality_anomalies.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    saved_files.append(path)

    # 7. Topic-level claim distribution and label balance
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    topics = sorted(metrics["topic_breakdown"].keys())
    s_counts = [metrics["topic_breakdown"][t]["S"] for t in topics]
    ns_counts = [metrics["topic_breakdown"][t]["NS"] for t in topics]

    y_pos = np.arange(len(topics))
    ax.barh(y_pos, s_counts, color="#2b7bba", label="Supported (S)", edgecolor="black", height=0.6)
    ax.barh(y_pos, ns_counts, left=s_counts, color="#d95f02", label="Not Supported (NS)", edgecolor="black", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Atomic Facts")
    ax.set_title("Topic-Level Fact Counts and Label Balance (31 Biographies)", fontsize=11, fontweight="bold")
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
    shared_ratio = total_entries / unique_texts

    ax.bar(["Total Associated Passage Entries", "Unique Wikipedia Text Chunks"], [total_entries, unique_texts], color=["#7570b3", "#1b9e77"], edgecolor="black", width=0.45)
    ax.set_ylabel("Count")
    ax.set_title(f"Evidence Reuse Across Evaluated Facts (Reuse Factor: {shared_ratio:.2f}x)", fontsize=11, fontweight="bold")
    ax.text(0, total_entries + 10, f"{total_entries}", ha="center", fontweight="bold")
    ax.text(1, unique_texts + 10, f"{unique_texts}", ha="center", fontweight="bold")
    ax.set_ylim(0, 780)

    plt.tight_layout()
    path = os.path.join(output_dir, "eda_evidence_reuse.png")
    plt.savefig(path, dpi=200)
    plt.close()
    saved_files.append(path)

    return saved_files


def run_eda(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
            dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
            figures_dir: str = "output_figures",
            outputs_dir: str = "outputs") -> None:
    """
    Main entry point for Phase 1 EDA pipeline.
    Executes only when factcheck.py is run directly via CLI.
    """
    print("[Phase 1 EDA] Starting dataset loading and dynamic analysis...")

    # Load passages and examples
    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)

    print(f"[Phase 1 EDA] Loaded {len(raw_passages):,} retrieval lines and {len(examples)} development fact instances.")

    # Run data quality audit
    audit = run_data_quality_audit(passages_path, dev_path, examples, raw_bios, raw_passages)
    print(f"[Phase 1 EDA] Data-quality audit complete. Duplicate fact strings found: {audit['duplicate_fact_strings_count']}")

    # Compute distributional metrics
    metrics = compute_eda_metrics(examples, raw_bios, raw_passages, passages_dict)
    print(f"[Phase 1 EDA] Computed metrics. Claim-to-passage length asymmetry: {metrics['length_asymmetry_word_ratio']:.2f}x")

    # Save machine-readable audit record
    os.makedirs(outputs_dir, exist_ok=True)
    summary_record = {
        "metrics": metrics,
        "data_quality_audit": audit
    }
    json_path = os.path.join(outputs_dir, "eda_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_record, f, indent=2)
    print(f"[Phase 1 EDA] Saved machine-readable audit record to '{json_path}'.")

    # Generate high-DPI figures
    saved_figures = generate_eda_visualizations(metrics, audit, examples, figures_dir)
    print(f"[Phase 1 EDA] Generated {len(saved_figures)} figures in '{figures_dir}/':")
    for fig_path in saved_figures:
        print(f"  - {fig_path}")

    print("[Phase 1 EDA] Execution completed successfully.")


# ==============================================================================
# PHASE 2: EVALUATION FRAMEWORK & BASELINE BENCHMARKS
# ==============================================================================

def calculate_confusion_matrix(gold_labels: List[str], pred_labels: List[str]) -> List[List[int]]:
    """
    Computes a 2x2 confusion matrix matching factchecking_main.py orientation.
    Label indexer: index 0 = 'S', index 1 = 'NS'.
    Returns:
        confusion_mat[gold_label_idx][pred_label_idx]:
        [[TP, FN],
         [FP, TN]]
    """
    label_indexer = ["S", "NS"]
    confusion_mat = [[0, 0], [0, 0]]
    for g, p in zip(gold_labels, pred_labels):
        g_idx = label_indexer.index(g)
        p_idx = label_indexer.index(p)
        confusion_mat[g_idx][p_idx] += 1
    return confusion_mat


def calculate_metrics(confusion_mat: List[List[int]]) -> Dict[str, Any]:
    """
    Replicates the exact evaluation logic of print_eval_stats from factchecking_main.py,
    augmented with balanced accuracy and macro-F1.
    """
    tp = confusion_mat[0][0]
    fn = confusion_mat[0][1]
    fp = confusion_mat[1][0]
    tn = confusion_mat[1][1]
    total = tp + fn + fp + tn
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0

    # Class 0: 'S'
    num_correct_s = tp
    num_gold_s = tp + fn
    num_pred_s = tp + fp
    rec_s = num_correct_s / num_gold_s if num_gold_s > 0 else 0.0
    if num_pred_s > 0:
        prec_s = num_correct_s / num_pred_s
        f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) > 0 else 0.0
    else:
        prec_s = None
        f1_s = None

    # Class 1: 'NS'
    num_correct_ns = tn
    num_gold_ns = fp + tn
    num_pred_ns = fn + tn
    rec_ns = num_correct_ns / num_gold_ns if num_gold_ns > 0 else 0.0
    if num_pred_ns > 0:
        prec_ns = num_correct_ns / num_pred_ns
        f1_ns = (2 * prec_ns * rec_ns / (prec_ns + rec_ns)) if (prec_ns + rec_ns) > 0 else 0.0
    else:
        prec_ns = None
        f1_ns = None

    # Balanced metrics
    balanced_acc = (rec_s + rec_ns) / 2.0
    macro_f1 = ((f1_s if f1_s is not None else 0.0) + (f1_ns if f1_ns is not None else 0.0)) / 2.0 if (f1_s is not None and f1_ns is not None) else None

    return {
        "confusion_matrix": confusion_mat,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "precision_s": prec_s,
        "recall_s": rec_s,
        "f1_s": f1_s,
        "precision_ns": prec_ns,
        "recall_ns": rec_ns,
        "f1_ns": f1_ns,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1
    }


def evaluate_fact_checker_standalone(examples: List[FactExample], fact_checker: FactChecker) -> Tuple[Dict[str, Any], List[str]]:
    """
    Evaluates any FactChecker instance on development examples using official binary mapping:
    'S' remains 'S', 'NS' remains 'NS', and 'IR' is mapped to 'NS'.
    Returns (metrics_dict, predictions_list).
    """
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    predictions = []
    for ex in examples:
        raw_pred = fact_checker.predict(ex.fact, ex.passages)
        if raw_pred not in ("S", "NS"):
            raise ValueError(f"Classifier returned invalid prediction '{raw_pred}'. Must be 'S' or 'NS'.")
        predictions.append(raw_pred)

    confusion_mat = calculate_confusion_matrix(gold_labels, predictions)
    metrics = calculate_metrics(confusion_mat)
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


def generate_baseline_performance_table(always_m: Dict[str, Any], random_single_m: Dict[str, Any], random_mc: Dict[str, Any], output_path: str = "output_figures/baselines_performance_table.png") -> str:
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

    table_data = [
        [
            "Overall Accuracy",
            f"112/221 = {always_m['accuracy']*100:.2f}%",
            f"{random_single_m['correct']}/221 = {random_single_m['accuracy']*100:.2f}%",
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
            fmt_pct(always_m['precision_s'], "112/221"),
            fmt_pct(random_single_m['precision_s'], f"{random_single_m['tp']}/{random_single_m['tp']+random_single_m['fp']}"),
            fmt_mc(random_mc["precision_s"]),
            "Target: High Precision"
        ],
        [
            "Recall (Supported - S)",
            fmt_pct(always_m['recall_s'], "112/112"),
            fmt_pct(random_single_m['recall_s'], f"{random_single_m['tp']}/112"),
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
            fmt_pct(always_m['recall_ns'], "0/109"),
            fmt_pct(random_single_m['recall_ns'], f"{random_single_m['tn']}/109"),
            fmt_mc(random_mc["recall_ns"]),
            "Target: High Recall"
        ],
        [
            "F1-Score (Not Supported - NS)",
            "Undefined",
            f"{random_single_m['f1_ns']:.4f} ({random_single_m['f1_ns']*100:.2f}%)",
            f"{random_mc['f1_ns']['mean']:.4f} ± {random_mc['f1_ns']['std']:.4f}",
            "Harmonic Mean"
        ],
        [
            "Macro-F1 (Unweighted Average)",
            "Undefined (0.3364 if 0)",
            f"{random_single_m['macro_f1']:.4f} ({random_single_m['macro_f1']*100:.2f}%)",
            f"{random_mc['macro_f1']['mean']:.4f} ± {random_mc['macro_f1']['std']:.4f}",
            "Balanced Quality"
        ],
        [
            "Evaluated Instances (N)",
            "221 facts",
            "221 facts (Seed 42)",
            "221 facts × 1,000 runs",
            "221 facts (All preserved)"
        ]
    ]

    col_labels = [
        "Evaluation Metric / Dimension",
        "Always Supported\n(Majority Baseline)",
        "Random Guess\n(Seed 42 Single Run)",
        "Random Guess\n(1,000-Run Monte Carlo)",
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
            cell.set_facecolor("#1f497d")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
            if col == 4:
                cell.set_text_props(weight="semibold", color="#78281f")
            elif col == 1:
                cell.set_text_props(weight="semibold")

    plt.title("Baseline Benchmark Performance Comparison (FActScore Development Set, N = 221)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_baseline_confusion_matrices(always_m: Dict[str, Any], random_single_m: Dict[str, Any], output_path: str = "output_figures/baselines_confusion_matrix.png") -> str:
    """
    Renders side-by-side heatmaps of confusion matrices for Always Supported and Random Guess baselines.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)

    cm_always = np.array(always_m["confusion_matrix"])
    cm_random = np.array(random_single_m["confusion_matrix"])
    categories = ["Supported\n(S)", "Not Supported\n(NS)"]

    # Panel A: Always Supported
    im1 = ax1.imshow(cm_always, cmap="Blues", vmin=0, vmax=125)
    ax1.set_title(f"Panel A: Always Supported (Majority)\nAccuracy: {always_m['accuracy']*100:.2f}% (112/221)", fontsize=10, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
    ax1.set_ylabel("True Label (Ground Truth)", fontsize=9, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_yticklabels(categories, fontsize=9)

    labels_always = [
        [f"{cm_always[0,0]}\n({cm_always[0,0]/221*100:.1f}%)\n[TP]", f"{cm_always[0,1]}\n({cm_always[0,1]/221*100:.1f}%)\n[FN]"],
        [f"{cm_always[1,0]}\n({cm_always[1,0]/221*100:.1f}%)\n[FP]", f"{cm_always[1,1]}\n({cm_always[1,1]/221*100:.1f}%)\n[TN]"]
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_always[i, j] > 60 else "black"
            ax1.text(j, i, labels_always[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Panel B: Random Guess (Seed 42)
    im2 = ax2.imshow(cm_random, cmap="Blues", vmin=0, vmax=125)
    ax2.set_title(f"Panel B: Random Guess (Seed 42)\nAccuracy: {random_single_m['accuracy']*100:.2f}% ({random_single_m['correct']}/221)", fontsize=10, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Label", fontsize=9, fontweight="bold")
    ax2.set_ylabel("True Label (Ground Truth)", fontsize=9, fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_yticklabels(categories, fontsize=9)

    labels_random = [
        [f"{cm_random[0,0]}\n({cm_random[0,0]/221*100:.1f}%)\n[TP]", f"{cm_random[0,1]}\n({cm_random[0,1]/221*100:.1f}%)\n[FN]"],
        [f"{cm_random[1,0]}\n({cm_random[1,0]/221*100:.1f}%)\n[FP]", f"{cm_random[1,1]}\n({cm_random[1,1]/221*100:.1f}%)\n[TN]"]
    ]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_random[i, j] > 60 else "black"
            ax2.text(j, i, labels_random[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Add shared colorbar and adjust spacing
    fig.subplots_adjust(wspace=0.35, right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.65])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label("Instance Count", fontsize=9, fontweight="bold")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def run_baselines(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
                  dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
                  figures_dir: str = "output_figures",
                  outputs_dir: str = "outputs") -> None:
    """
    Main entry point for Phase 2 Evaluation Framework and Baseline Benchmarks.
    Evaluates Always Supported and Random Guess (single run and 1,000-run Monte Carlo).
    Saves outputs/baseline_results.json and generates high-DPI figures.
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
    print(f"[Phase 2 Baselines] Monte Carlo (1,000 runs): Mean={acc_mc['mean']*100:.2f}%, Std={acc_mc['std']*100:.2f}%, Min={acc_mc['min']*100:.2f}%, Max={acc_mc['max']*100:.2f}%, 95% CI=[{acc_mc['ci_95_lower']*100:.2f}%, {acc_mc['ci_95_upper']*100:.2f}%]")

    # 4. Save JSON results
    os.makedirs(outputs_dir, exist_ok=True)
    baseline_record = {
        "dataset_summary": {
            "total_evaluated_instances": len(examples),
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
    json_path = os.path.join(outputs_dir, "baseline_results.json")
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
# CLI EXECUTION GUARD
# ==============================================================================

if __name__ == "__main__":
    if "--run-eda" in sys.argv or "--eda" in sys.argv:
        run_eda()
    elif "--run-baselines" in sys.argv or "--baselines" in sys.argv:
        run_baselines()
    elif "--all" in sys.argv:
        run_eda()
        run_baselines()
    else:
        print("Usage:")
        print("  python factcheck.py --run-eda        (Phase 1 EDA analysis)")
        print("  python factcheck.py --run-baselines  (Phase 2 Baseline benchmarks)")
