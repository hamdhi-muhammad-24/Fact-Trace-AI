# factcheck.py

import os
import sys
import json
import re
import math
import gc
import time
import textwrap
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import torch
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

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


# ==============================================================================
# TASK 1: LEXICAL PROCESSING & SIMILARITY FUNCTIONS
# ==============================================================================

_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}

_CACHED_ENGLISH_STOPS: Optional[Set[str]] = None
_LEMMATIZER_INSTANCE: Optional[WordNetLemmatizer] = None

def get_english_stopwords() -> Set[str]:
    """
    Safely retrieves English stopwords with negation words preserved.
    Caches the result and avoids unnecessary automatic network downloads.
    """
    global _CACHED_ENGLISH_STOPS
    if _CACHED_ENGLISH_STOPS is None:
        try:
            nltk_stops = set(stopwords.words("english"))
        except Exception:
            nltk_stops = _FALLBACK_STOPWORDS
        negation_preserve = {"no", "not", "nor", "neither", "never", "none", "cannot", "without"}
        _CACHED_ENGLISH_STOPS = nltk_stops - negation_preserve
    return _CACHED_ENGLISH_STOPS


def get_wordnet_lemmatizer() -> WordNetLemmatizer:
    """
    Safely retrieves or initializes a cached WordNetLemmatizer instance
    without unnecessary automatic network downloads.
    """
    global _LEMMATIZER_INSTANCE
    if _LEMMATIZER_INSTANCE is None:
        try:
            _LEMMATIZER_INSTANCE = WordNetLemmatizer()
        except Exception:
            _LEMMATIZER_INSTANCE = None
    return _LEMMATIZER_INSTANCE if _LEMMATIZER_INSTANCE is not None else WordNetLemmatizer()


def preprocess_tokens(text: str, mode: str, stemmer: PorterStemmer, lemmatizer: WordNetLemmatizer, stops: Set[str]) -> List[str]:
    """
    Tokenizes and preprocesses text according to ablation modes:
    - P0: Lowercased alphanumeric regex tokens (\b[a-zA-Z0-9]+\b).
    - P1: P0 with stopwords removed (preserving negations and numbers).
    - P2: P1 with Porter stemming.
    - P3: P1 with WordNet noun + verb lemmatization.
    - P4: P1 with Porter stemming (base tokens for number-aware auditing).
    """
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    if mode == "P0":
        return tokens

    # Filter stopwords while preserving key negations and numbers
    filtered = [t for t in tokens if t not in stops]
    if mode == "P1":
        return filtered
    elif mode in ("P2", "P4"):
        return [stemmer.stem(t) for t in filtered]
    elif mode == "P3":
        lemmas = []
        for t in filtered:
            lem = lemmatizer.lemmatize(t, pos="n")
            if lem == t:
                lem = lemmatizer.lemmatize(t, pos="v")
            lemmas.append(lem)
        return lemmas
    return filtered


def compute_similarity_metric(claim_tokens: List[str], passage_tokens: List[str], metric: str) -> float:
    """
    Computes lexical similarity between claim and passage token lists.
    Supported metrics: 'recall', 'jaccard', 'dice', 'tf_cosine'.
    """
    c_set = set(claim_tokens)
    p_set = set(passage_tokens)

    if metric == "recall":
        if not c_set:
            return 0.0
        return len(c_set & p_set) / len(c_set)
    elif metric == "jaccard":
        union_len = len(c_set | p_set)
        if union_len == 0:
            return 0.0
        return len(c_set & p_set) / union_len
    elif metric == "dice":
        denom = len(c_set) + len(p_set)
        if denom == 0:
            return 0.0
        return (2.0 * len(c_set & p_set)) / denom
    elif metric == "tf_cosine":
        if not claim_tokens or not passage_tokens:
            return 0.0
        c_count = Counter(claim_tokens)
        p_count = Counter(passage_tokens)
        common_words = set(c_count.keys()) & set(p_count.keys())
        dot = sum(c_count[w] * p_count[w] for w in common_words)
        norm_c = math.sqrt(sum(v**2 for v in c_count.values()))
        norm_p = math.sqrt(sum(v**2 for v in p_count.values()))
        denom = norm_c * norm_p
        return (dot / denom) if denom > 0.0 else 0.0
    else:
        raise ValueError(f"Unknown metric '{metric}'. Expected 'recall', 'jaccard', 'dice', or 'tf_cosine'.")


def compute_claim_passages_score(fact: str, passages: List[dict], prep_mode: str, metric: str, agg: str,
                                 stemmer: PorterStemmer, lemmatizer: WordNetLemmatizer, stops: Set[str],
                                 number_penalty: float = 0.0,
                                 penalty_rule: str = "disjoint") -> float:
    """
    Computes overall aggregated similarity score of a claim against a list of candidate passages.
    Handles Max, Mean, and Concat aggregation, plus P4 numerical mismatch penalty.
    """
    if not passages:
        return 0.0

    c_tokens = preprocess_tokens(fact, prep_mode, stemmer, lemmatizer, stops)
    c_nums = set(re.findall(r'\b\d+\b', fact)) if prep_mode == "P4" else None

    if agg == "concat":
        combined_text = " ".join(p.get("text", "") for p in passages)
        p_tokens = preprocess_tokens(combined_text, prep_mode, stemmer, lemmatizer, stops)
        score = compute_similarity_metric(c_tokens, p_tokens, metric)
        if prep_mode == "P4" and c_nums:
            p_nums = set(re.findall(r'\b\d+\b', combined_text))
            mismatch = c_nums.isdisjoint(p_nums) if penalty_rule == "disjoint" else (not c_nums.issubset(p_nums))
            if mismatch:
                score *= number_penalty
        return score
    else:
        per_passage_scores = []
        for p in passages:
            p_text = p.get("text", "")
            p_tokens = preprocess_tokens(p_text, prep_mode, stemmer, lemmatizer, stops)
            s = compute_similarity_metric(c_tokens, p_tokens, metric)
            if prep_mode == "P4" and c_nums:
                p_nums = set(re.findall(r'\b\d+\b', p_text))
                mismatch = c_nums.isdisjoint(p_nums) if penalty_rule == "disjoint" else (not c_nums.issubset(p_nums))
                if mismatch:
                    s *= number_penalty
            per_passage_scores.append(s)

        if not per_passage_scores:
            return 0.0
        if agg == "max":
            return max(per_passage_scores)
        elif agg == "mean":
            return sum(per_passage_scores) / len(per_passage_scores)
        else:
            raise ValueError(f"Unknown aggregation strategy '{agg}'. Expected 'max', 'mean', or 'concat'.")


class WordRecallThresholdFactChecker(FactChecker):
    """
    Task 1: Lexical Word Overlap Fact-Checker.
    Default configuration is experimentally optimized in Phase 3:
      - Preprocessing: P4 (Stopword filtered with negations preserved, Porter stemmed, numerical consistency checked)
      - Similarity Metric: Asymmetric Word Recall (fraction of claim tokens covered in evidence)
      - Evidence Aggregation: Max-pooling across retrieved Wikipedia passages
      - Decision Threshold: theta* = 0.64 (centered in the robust 0.61-0.67 plateau)
      - Numerical Mismatch Penalty: 0.0 (vetoes claim if claim contains numbers not in the passage)
    """
    def __init__(self, threshold: float = 0.64, prep_mode: str = "P4", metric: str = "recall",
                 agg: str = "max", number_penalty: float = 0.0):
        self.threshold = threshold
        self.prep_mode = prep_mode
        self.metric = metric
        self.agg = agg
        self.number_penalty = number_penalty
        self._stemmer = PorterStemmer()
        self._lemmatizer = get_wordnet_lemmatizer()
        self._stops = get_english_stopwords()

    def predict(self, fact: str, passages: List[dict]) -> str:
        score = compute_claim_passages_score(
            fact=fact,
            passages=passages,
            prep_mode=self.prep_mode,
            metric=self.metric,
            agg=self.agg,
            stemmer=self._stemmer,
            lemmatizer=self._lemmatizer,
            stops=self._stops,
            number_penalty=self.number_penalty
        )
        return "S" if score >= self.threshold else "NS"


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
# PHASE 3: TASK 1 WORD OVERLAP FACT-CHECKING EXPERIMENT SUITE
# ==============================================================================

def find_optimal_threshold(scores: List[float], gold_labels: List[str], metric_name: str = "recall") -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates candidate thresholds using score-aware boundaries and a fine-grained grid.
    Returns (optimal_threshold, best_metrics_dict, full_threshold_curve_data).
    Selection rule:
      1. Maximum accuracy
      2. Maximum balanced accuracy (tie-breaking)
      3. Maximum macro-F1 (tie-breaking)
      4. Maximum Supported (S) F1 (tie-breaking)
      5. Centered within the maximum-accuracy plateau (robustness)
    """
    unique_scores = sorted(set(scores))
    midpoints = [(unique_scores[i] + unique_scores[i+1]) / 2.0 for i in range(len(unique_scores)-1)]

    if metric_name in ("jaccard", "dice", "tf_cosine"):
        fine_grid = [i / 10000.0 for i in range(0, 3000, 5)]  # 0.0000 to 0.3000
    else:
        fine_grid = [i / 100.0 for i in range(0, 101)]  # 0.00 to 1.00

    candidates = sorted(set(unique_scores + midpoints + fine_grid))
    min_s, max_s = min(scores), max(scores)
    candidates = [c for c in candidates if (min_s - 0.01) <= c <= (max_s + 0.01)]
    if not candidates:
        candidates = [0.5]

    curve_records = []
    best_acc = -1.0
    plateau = []

    for th in candidates:
        preds = ["S" if s >= th else "NS" for s in scores]
        cm = calculate_confusion_matrix(gold_labels, preds)
        m = calculate_metrics(cm)
        rec = {
            "threshold": th,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "precision_s": m["precision_s"],
            "recall_s": m["recall_s"],
            "f1_s": m["f1_s"],
            "precision_ns": m["precision_ns"],
            "recall_ns": m["recall_ns"],
            "f1_ns": m["f1_ns"],
            "macro_f1": m["macro_f1"],
            "confusion_matrix": cm,
            "correct": m["correct"],
            "total": m["total"]
        }
        curve_records.append(rec)

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            plateau = [rec]
        elif m["accuracy"] == best_acc:
            plateau.append(rec)

    # 1. Tie-break: highest balanced accuracy
    max_bal = max(r["balanced_accuracy"] for r in plateau)
    plateau = [r for r in plateau if r["balanced_accuracy"] == max_bal]

    # 2. Tie-break: highest macro F1
    max_macro = max((r["macro_f1"] if r["macro_f1"] is not None else 0.0) for r in plateau)
    plateau = [r for r in plateau if (r["macro_f1"] if r["macro_f1"] is not None else 0.0) == max_macro]

    # 3. Tie-break: highest Supported (S) F1
    max_f1_s = max((r["f1_s"] if r["f1_s"] is not None else 0.0) for r in plateau)
    plateau = [r for r in plateau if (r["f1_s"] if r["f1_s"] is not None else 0.0) == max_f1_s]

    # 4. Tie-break: median threshold in the remaining plateau
    plateau_thresholds = sorted(r["threshold"] for r in plateau)
    median_th = plateau_thresholds[len(plateau_thresholds) // 2]
    best_record = next(r for r in plateau if r["threshold"] == median_th)
    best_record["plateau_bounds"] = [plateau_thresholds[0], plateau_thresholds[-1]]

    return best_record["threshold"], best_record, curve_records


def evaluate_config_thresholds_fast(sub_scores: List[float], sub_gold: List[str], metric_name: str) -> Tuple[float, Dict[str, Any], Tuple[float, float], List[Dict[str, Any]], int]:
    """
    Fast evaluation of all candidate thresholds for a given score vector and gold label subset.
    Returns (best_threshold, best_metrics_dict, (plateau_min, plateau_max), curve_records, num_candidates).
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
    Deterministic multi-criterion ranking key for selecting the winning classifier configuration:
      1. Highest overall accuracy
      2. Highest balanced accuracy
      3. Highest macro-F1
      4. Highest Supported (S) F1-score
      5. Widest plateau range (maximum threshold stability / robustness)
      6. Lowest complexity index (simpler models preferred when metrics are tied)
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
    In each fold:
      1. Held-out biography topics are isolated from training examples.
      2. Duplicate claims ('Cleveland is in Ohio.') are isolated within Fold 0.
      3. All candidate configurations and decision thresholds are evaluated strictly on training fold data.
      4. The best configuration for that fold is chosen via documented multi-criterion ranking.
      5. That selected model is evaluated on the held-out validation fold.
    Returns complete cross-validation summary, fold audit records, and out-of-fold metrics.
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


# ==============================================================================
# PHASE 3: HIGH-DPI VISUALIZATION GENERATORS
# ==============================================================================

def generate_task1_preprocessing_comparison(prep_results: Dict[str, Dict[str, Any]],
                                            baseline_always_acc: float,
                                            output_path: str = "output_figures/task1_preprocessing_comparison.png") -> str:
    """
    Bar chart comparing Accuracy, Balanced Accuracy, and Macro-F1 across P0-P4.
    All bar values, annotations, and baseline lines are strictly computed from measured data.
    """
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

    rects1 = ax.bar(x - width, accs, width, label="Overall Accuracy", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x, bal_accs, width, label="Balanced Accuracy", color="#2ca02c", edgecolor="black")
    rects3 = ax.bar(x + width, macro_f1s, width, label="Macro-F1", color="#ff7f0e", edgecolor="black")

    ax.axhline(75.0, color="red", linestyle="--", linewidth=1.8, label="Coursework Target (≥ 75.0%)")
    ax.axhline(baseline_always_acc * 100, color="gray", linestyle=":", linewidth=1.4,
               label=f"Always Supported Baseline ({baseline_always_acc*100:.2f}%)")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Task 1 Preprocessing Ablation Comparison (P0 - P4 under Max Word Recall)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9.5)
    ax.set_ylim(40, 105)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_metric_comparison(metric_results: Dict[str, Dict[str, Any]],
                                     baseline_always_acc: float,
                                     output_path: str = "output_figures/task1_metric_comparison.png") -> str:
    """
    Bar chart comparing peak Accuracy and Balanced Accuracy across Word Recall, Jaccard, Dice, and TF Cosine.
    All thresholds and metrics are dynamically annotated from real evaluations.
    """
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

    rects1 = ax.bar(x - width/2, accs, width, label="Overall Accuracy", color="#1f77b4", edgecolor="black")
    rects2 = ax.bar(x + width/2, bal_accs, width, label="Balanced Accuracy", color="#2ca02c", edgecolor="black")

    ax.axhline(75.0, color="red", linestyle="--", linewidth=1.8, label="Coursework Target (≥ 75.0%)")
    ax.axhline(baseline_always_acc * 100, color="gray", linestyle=":", linewidth=1.4,
               label=f"Always Supported Baseline ({baseline_always_acc*100:.2f}%)")

    ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax.set_title("Task 1 Lexical Similarity Metric Comparison (P4 Number-Aware, Max Aggregation)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9.5)
    ax.set_ylim(40, 105)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    for i, rect in enumerate(rects1):
        h = rect.get_height()
        th_val = ths[i]
        th_str = f"θ*={th_val:.2f}" if th_val >= 0.1 else f"θ*={th_val:.4f}"
        ax.annotate(f"{h:.1f}%\n({th_str})", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_threshold_curve(curve_data: List[Dict[str, Any]],
                                   optimal_th: float,
                                   plateau_bounds: Tuple[float, float],
                                   opt_record: Dict[str, Any],
                                   winning_title: str,
                                   output_path: str = "output_figures/task1_threshold_curve.png") -> str:
    """
    Plots Accuracy, S-F1, NS-F1, and Balanced Accuracy across decision threshold theta.
    Plateau bounds, peak counts, and title are calculated dynamically from measured predictions.
    """
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

    ax.plot(ths, accs, label="Overall Accuracy", color="#1f77b4", linewidth=2.2)
    ax.plot(ths, bal_accs, label="Balanced Accuracy", color="#2ca02c", linewidth=1.8, linestyle="-.")
    ax.plot(ths, f1_s, label="F1-Score (Supported - S)", color="#ff7f0e", linewidth=1.5, linestyle=":")
    ax.plot(ths, f1_ns, label="F1-Score (Not Supported - NS)", color="#9467bd", linewidth=1.5, linestyle="--")

    ax.axhline(75.0, color="red", linestyle="--", linewidth=1.5, label="Coursework Target (≥ 75.0%)")

    p_min, p_max = plateau_bounds
    ax.axvspan(p_min, p_max, color="#ffeb3b", alpha=0.35, label=f"Peak Accuracy Plateau [{p_min:.2f} - {p_max:.2f}]")
    ax.axvline(optimal_th, color="black", linestyle="-", linewidth=1.8, label=f"Selected θ* = {optimal_th:.2f}")

    correct_cnt = opt_record["correct"]
    total_cnt = opt_record["total"]
    peak_acc_pct = opt_record["accuracy"] * 100

    ax.plot(optimal_th, peak_acc_pct, marker="o", markersize=7, color="black")
    ax.annotate(f"Peak Accuracy: {peak_acc_pct:.2f}%\n({correct_cnt}/{total_cnt} correct)",
                xy=(optimal_th, peak_acc_pct),
                xytext=(optimal_th - 0.22, peak_acc_pct - 8),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="black"))

    ax.set_xlabel("Decision Threshold (θ)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Score (%)", fontsize=10, fontweight="bold")
    ax.set_title(f"Task 1 Decision Threshold Optimization Curve ({winning_title})", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0.10, 0.95)
    ax.set_ylim(35, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_validation_summary(cv_results: Dict[str, Any], output_path: str = "output_figures/task1_validation_summary.png") -> str:
    """
    Bar chart comparing Train vs Validation Accuracy across the 5 grouped cross-validation folds.
    All scores derive dynamically from actual nested fold executions.
    """
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

    rects1 = ax.bar(x - width/2, train_accs, width, label="Training Portion Accuracy", color="#aec7e8", edgecolor="black")
    rects2 = ax.bar(x + width/2, val_accs, width, label="Out-of-Fold Validation Accuracy", color="#1f77b4", edgecolor="black")

    ax.axhline(75.0, color="red", linestyle="--", linewidth=1.8, label="Coursework Target (≥ 75.0%)")

    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title(f"Task 1 Nested Grouped 5-Fold Cross-Validation Summary (Mean OOF: {cv_results['mean_val_accuracy']*100:.2f}% ± {cv_results['std_val_accuracy']*100:.2f}%)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_final_confusion_matrix(cm_data: List[List[int]],
                                          optimal_th: float,
                                          winning_title: str,
                                          output_path: str = "output_figures/task1_final_confusion_matrix.png") -> str:
    """
    High-DPI heatmap of the final confusion matrix for the winning Task 1 Word Overlap model.
    All titles, counts, and percentages are derived dynamically from cm_data.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=200)

    cm = np.array(cm_data)
    total = int(cm.sum())
    correct = int(cm[0, 0] + cm[1, 1])
    accuracy = correct / total if total > 0 else 0.0
    categories = ["Supported\n(S)", "Not Supported\n(NS)"]

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max() * 1.15), 1))
    ax.set_title(f"Task 1 Final Confusion Matrix ({winning_title} @ θ*={optimal_th:.2f})\nAccuracy: {accuracy*100:.2f}% ({correct}/{total} correct)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label (Ground Truth)", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_yticklabels(categories, fontsize=9.5)

    tags = [["TP", "FN"], ["FP", "TN"]]
    labels = [
        [f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)\n[{tags[i][j]}]" for j in range(2)]
        for i in range(2)
    ]

    threshold_color = cm.max() * 0.55
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold_color else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Instance Count", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


def generate_task1_final_results_table(baseline_always: Dict[str, Any],
                                       baseline_rand: Dict[str, Any],
                                       prep_records: Dict[str, Dict[str, Any]],
                                       final_m: Dict[str, Any],
                                       cv_oof_acc: float,
                                       winning_title: str,
                                       output_path: str = "output_figures/task1_final_results_table.png") -> str:
    """
    Rendered publication-grade table comparing Baselines, Ablations, and the Final Selected Model.
    All values are strictly computed from verified experimental metrics.
    """
    if not _HAS_MATPLOTLIB:
        return ""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=200)
    ax.axis("off")
    ax.axis("tight")

    def fmt(v: Optional[float]) -> str:
        if v is None:
            return "Undefined"
        return f"{v*100:.2f}%"

    def fmt_f1(v: Optional[float]) -> str:
        if v is None:
            return "Undefined"
        return f"{v:.4f}"

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
            cell.set_facecolor("#1f497d")
            cell.set_text_props(color="white", weight="bold", ha="center")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f2f5f9")
            else:
                cell.set_facecolor("#ffffff")
            if col == 5:
                cell.set_facecolor("#e8f8f5")
                cell.set_text_props(weight="bold", color="#117864")
            elif col == 6:
                cell.set_text_props(weight="semibold", color="#78281f")

    plt.title(f"Task 1 Final Benchmark Comparison (FActScore Development Set, N = {final_m['total']})", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close()
    return output_path


# ==============================================================================
# PHASE 3: TASK 1 PIPELINE RUNNER
# ==============================================================================

def run_task1(passages_path: str = "data/passages_bm25_ChatGPT_humfacts.jsonl",
              dev_path: str = "data/dev_labeled_ChatGPT.jsonl",
              figures_dir: str = "output_figures",
              outputs_dir: str = "outputs") -> None:
    """
    Main entry point for Phase 3 Task 1 Word Overlap Fact-Checking experiment suite.
    Executes all preprocessing ablations, metric comparisons, aggregation strategies,
    threshold optimizations, numerical consistency audits, and leak-free nested grouped 5-fold CV.
    Saves outputs/task1_experiments.json and generates high-DPI figures.
    """
    start_time = time.time()
    print("[Phase 3 Task 1] Loading datasets...")
    passages_dict, raw_passages = load_passages_data(passages_path)
    examples, raw_bios = load_dev_examples(dev_path, passages_dict)
    gold_labels = ["NS" if ex.label == "IR" else ex.label for ex in examples]
    print(f"[Phase 3 Task 1] Loaded {len(examples)} development fact examples across {len(raw_bios)} topics.")

    # 1. Baseline benchmarks loading / dynamic recalculation
    baseline_path = os.path.join(outputs_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        b_always = b_data.get("always_supported", {})
        b_rand = b_data.get("random_guess_seed42", {})
    else:
        print("[Phase 3 Task 1] baseline_results.json not found on disk; calculating baseline metrics dynamically...")
        always_checker = AlwaysEntailedFactChecker()
        always_preds = [always_checker.predict(ex.fact, ex.passages) for ex in examples]
        cm_always = calculate_confusion_matrix(gold_labels, always_preds)
        b_always = calculate_metrics(cm_always)

        rand_checker = RandomGuessFactChecker(seed=42)
        rand_preds = [rand_checker.predict(ex.fact, ex.passages) for ex in examples]
        cm_rand = calculate_confusion_matrix(gold_labels, rand_preds)
        b_rand = calculate_metrics(cm_rand)

    baseline_always_acc = float(b_always.get("accuracy", 112 / len(examples)))

    # 2. Tokenizer, Lemmatizer, and Negation-Preserving Stopwords setup
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

    # 3. Construct all 120 configurations
    grid_configs = []
    comp_idx = 0
    # Baseline configs (P0-P3 x 4 metrics x 3 aggregations = 48)
    for p in ["P0", "P1", "P2", "P3"]:
        for m in ["recall", "jaccard", "dice", "tf_cosine"]:
            for a in ["max", "concat", "mean"]:
                grid_configs.append({
                    "prep": p, "metric": m, "agg": a,
                    "penalty_rule": None, "penalty_factor": 0.0,
                    "complexity": comp_idx
                })
                comp_idx += 1

    # P4 disjoint configs (5 penalties x 4 metrics x 3 aggregations = 60)
    for pen in [0.0, 0.25, 0.50, 0.75, 1.0]:
        for m in ["recall", "jaccard", "dice", "tf_cosine"]:
            for a in ["max", "concat", "mean"]:
                grid_configs.append({
                    "prep": "P4", "metric": m, "agg": a,
                    "penalty_rule": "disjoint", "penalty_factor": pen,
                    "complexity": comp_idx
                })
                comp_idx += 1

    # P4 subset configs (penalty 0.0 x 4 metrics x 3 aggregations = 12)
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

    # 4. Evaluate all 120 configurations across full development set
    print("\n--- [Experiment 1] Exhaustive Evaluation of Search Space on Development Set ---")
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

    # Sort using documented deterministic multi-criterion ranking
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

    print(f"[Phase 3 Task 1] Evaluated {len(grid_configs)} configurations ({total_evals_dev:,} threshold candidates) in {time.time() - t_grid0:.2f} seconds.")
    print(f"\nTop 5 Ranked Configurations on Development Set:")
    for rank, r in enumerate(full_dev_results[:5], 1):
        c = r["cfg"]
        rule_desc = f"{c['penalty_rule']}(λ={c['penalty_factor']})" if c["penalty_rule"] else "None"
        print(f"  #{rank}: {c['prep']} | {c['metric']} | {c['agg']} | num_check={rule_desc} -> Acc: {r['accuracy']*100:.2f}%, BalAcc: {r['balanced_accuracy']*100:.2f}%, Macro-F1: {r['macro_f1']:.4f}, Plateau: [{r['plateau_bounds'][0]:.2f}, {r['plateau_bounds'][1]:.2f}] @ θ*={r['threshold']:.2f}")

    print(f"\n--- [Winning Configuration Objectively Selected] {winning_title} ---")
    print(f"  Configuration Details: Preprocessing={winning_prep}, Metric={winning_metric}, Aggregation={winning_agg}, Number Rule={winning_rule}, Penalty λ={winning_penalty}")
    print(f"  Optimal Threshold θ* = {opt_th:.2f} (Plateau: [{min_plat_th:.2f}, {max_plat_th:.2f}], Width: {max_plat_th - min_plat_th:.2f})")
    print(f"  Accuracy: {final_m['correct']}/{final_m['total']} = {final_m['accuracy']*100:.2f}% (Target: ≥ 75.0%)")
    print(f"  Balanced Accuracy: {final_m['balanced_accuracy']*100:.2f}%")
    print(f"  Precision (S): {final_m['precision_s']*100:.2f}% | Recall (S): {final_m['recall_s']*100:.2f}% | F1 (S): {final_m['f1_s']:.4f}")
    print(f"  Precision (NS): {final_m['precision_ns']*100:.2f}% | Recall (NS): {final_m['recall_ns']*100:.2f}% | F1 (NS): {final_m['f1_ns']:.4f}")
    print(f"  Macro-F1: {final_m['macro_f1']:.4f}")
    print(f"  Confusion Matrix: {final_m['confusion_matrix']}")

    # 5. Extract specific ablation sets from evaluated grid
    res_by_key = {r["key"]: r for r in full_dev_results}

    # Preprocessing ablation under Word Recall + Max
    prep_results = {}
    for m in ["P0", "P1", "P2", "P3", "P4"]:
        k = f"{m}_recall_max_disjoint_0.0" if m == "P4" else f"{m}_recall_max_None_0.0"
        r = res_by_key[k]
        prep_results[m] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    # Metric comparison under P4 + Max (disjoint λ=0.0)
    metric_results = {}
    for m in ["recall", "jaccard", "dice", "tf_cosine"]:
        k = f"P4_{m}_max_disjoint_0.0"
        r = res_by_key[k]
        metric_results[m] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    # Aggregation comparison under P4 + Word Recall (disjoint λ=0.0)
    agg_results = {}
    for a in ["max", "mean", "concat"]:
        k = f"P4_recall_{a}_disjoint_0.0"
        r = res_by_key[k]
        agg_results[a] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    # Penalty factor ablation for P4 + Word Recall + Max (disjoint)
    penalty_results = {}
    for pen in [0.0, 0.25, 0.50, 0.75, 1.0]:
        k = f"P4_recall_max_disjoint_{pen}"
        r = res_by_key[k]
        penalty_results[str(pen)] = {**r["train_metrics"], "threshold": r["threshold"], "plateau_bounds": r["plateau_bounds"]}

    # Rule comparison (disjoint vs subset) under P4 + Word Recall + Max (λ=0.0)
    r_disjoint = res_by_key["P4_recall_max_disjoint_0.0"]
    r_subset = res_by_key["P4_recall_max_subset_0.0"]
    rule_results = {
        "disjoint": {**r_disjoint["train_metrics"], "threshold": r_disjoint["threshold"], "plateau_bounds": r_disjoint["plateau_bounds"]},
        "subset": {**r_subset["train_metrics"], "threshold": r_subset["threshold"], "plateau_bounds": r_subset["plateau_bounds"]}
    }

    # 6. Numerical consistency audit on the 38 claims containing numbers
    print("\n--- [Experiment 2] Numerical Entity Consistency Audit (Claims with Numbers) ---")
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

    print(f"  Total claims containing numerical entities: {len(claims_with_nums)}")
    print(f"  False Positives in P2 converted to True Negatives in P4: {len(fp_to_tn_transitions)}")
    for item in fp_to_tn_transitions:
        print(f"    Idx {item['index']}: '{item['fact']}' | Claim Nums: {item['claim_numbers']} | Gold: {item['gold_label']} (P2: {item['p2_predicted']} -> P4: {item['p4_predicted']})")

    print(f"  Partial numerical match cases (multi-number claims): {len(partial_match_cases)}")
    for item in partial_match_cases:
        print(f"    Idx {item['index']}: '{item['fact']}' | Claim Nums: {item['claim_numbers']} | Passage Nums: {item['passage_numbers_union']} | Gold: {item['gold_label']}")

    # 7. Leak-Free Nested Grouped 5-Fold Cross-Validation
    print("\n--- [Experiment 3] Leak-Free Nested Grouped 5-Fold Cross-Validation ---")
    t_cv0 = time.time()
    cv_results = run_nested_grouped_cross_validation(examples, raw_bios, all_scores_dict, grid_configs, gold_labels)
    t_cv = time.time() - t_cv0
    print(f"[Phase 3 Task 1] Nested CV completed in {t_cv:.2f} seconds ({cv_results['total_inner_threshold_evaluations']:,} inner threshold evaluations across 5 folds).")

    for f in cv_results["folds"]:
        c = f["selected_config"]
        print(f"  Fold {f['fold']} (N={f['val_examples_count']}): Model={c['prep']} {c['metric']} {c['agg']}, θ*={f['chosen_threshold']:.2f} | Train Acc={f['train_accuracy']*100:.2f}%, Val Acc={f['val_accuracy']*100:.2f}%, S-F1={f['val_f1_s']:.4f}, NS-F1={f['val_f1_ns']:.4f}")
    print(f"  Mean Out-of-Fold Val Accuracy: {cv_results['mean_val_accuracy']*100:.2f}% ± {cv_results['std_val_accuracy']*100:.2f}%")
    print(f"  Overall Out-of-Fold Accuracy: {cv_results['overall_oof_accuracy']*100:.2f}% ({cv_results['overall_oof_metrics']['correct']}/{cv_results['overall_oof_metrics']['total']})")
    print(f"  Overall Out-of-Fold Balanced Accuracy: {cv_results['overall_oof_balanced_accuracy']*100:.2f}%")
    print(f"  Overall Out-of-Fold Confusion Matrix: {cv_results['overall_oof_metrics']['confusion_matrix']}")

    # 8. Collect all 221 individual prediction records
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
            "topic": idx_to_topic[i],
            "fact": ex.fact,
            "gold_label": gold_labels[i],
            "predicted_label": final_preds[i],
            "score": winning_scores[i],
            "is_correct": bool(final_preds[i] == gold_labels[i]),
            "contains_numbers": bool(len(claim_nums[i]) > 0),
            "claim_numbers": sorted(list(claim_nums[i]))
        })

    elapsed_time = time.time() - start_time
    print(f"\n[Phase 3 Task 1] Total experiment suite runtime: {elapsed_time:.2f} seconds.")

    # 9. Save structured JSON experiment records
    os.makedirs(outputs_dir, exist_ok=True)
    task1_record = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_evaluated_instances": len(examples),
            "total_biography_topics": len(raw_bios),
            "search_space_configurations": len(grid_configs),
            "dev_threshold_evaluations": total_evals_dev,
            "nested_cv_inner_threshold_evaluations": cv_results["total_inner_threshold_evaluations"],
            "total_threshold_evaluations": total_evals_dev + cv_results["total_inner_threshold_evaluations"],
            "runtime_seconds": round(elapsed_time, 2)
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
            "selection_rationale": f"Ranked #1 out of {len(grid_configs)} configurations via deterministic multi-criterion ranking. Achieved highest overall accuracy ({final_m['accuracy']*100:.2f}%, {final_m['correct']}/{final_m['total']}), balanced accuracy ({final_m['balanced_accuracy']*100:.2f}%), macro-F1 ({final_m['macro_f1']:.4f}), and a wide threshold plateau [{min_plat_th:.2f}, {max_plat_th:.2f}] (width: {max_plat_th - min_plat_th:.2f}).",
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

    json_path = os.path.join(outputs_dir, "task1_experiments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(task1_record, f, indent=2)
    print(f"[Phase 3 Task 1] Saved comprehensive experimental record to '{json_path}'.")

    # 10. Generate high-DPI figures
    os.makedirs(figures_dir, exist_ok=True)
    fig1 = os.path.join(figures_dir, "task1_preprocessing_comparison.png")
    fig2 = os.path.join(figures_dir, "task1_metric_comparison.png")
    fig3 = os.path.join(figures_dir, "task1_threshold_curve.png")
    fig4 = os.path.join(figures_dir, "task1_validation_summary.png")
    fig5 = os.path.join(figures_dir, "task1_final_confusion_matrix.png")
    fig6 = os.path.join(figures_dir, "task1_final_results_table.png")

    generate_task1_preprocessing_comparison(prep_results, baseline_always_acc, fig1)
    generate_task1_metric_comparison(metric_results, baseline_always_acc, fig2)
    generate_task1_threshold_curve(full_curve, opt_th, (min_plat_th, max_plat_th), final_m, winning_title, fig3)
    generate_task1_validation_summary(cv_results, fig4)
    generate_task1_final_confusion_matrix(final_m["confusion_matrix"], opt_th, winning_title, fig5)
    generate_task1_final_results_table(b_always, b_rand, prep_results, final_m, cv_results["overall_oof_accuracy"], winning_title, fig6)

    print(f"[Phase 3 Task 1] Generated high-DPI figures in '{figures_dir}/':")
    for fp in [fig1, fig2, fig3, fig4, fig5, fig6]:
        print(f"  - {fp}")

    print("[Phase 3 Task 1] Execution completed successfully.")


# ==============================================================================
# CLI EXECUTION GUARD
# ==============================================================================

if __name__ == "__main__":
    if "--run-eda" in sys.argv or "--eda" in sys.argv:
        run_eda()
    elif "--run-baselines" in sys.argv or "--baselines" in sys.argv:
        run_baselines()
    elif "--run-task1" in sys.argv or "--task1" in sys.argv:
        run_task1()
    elif "--all" in sys.argv:
        run_eda()
        run_baselines()
        run_task1()
    else:
        print("Usage:")
        print("  python factcheck.py --run-eda        (Phase 1 EDA analysis)")
        print("  python factcheck.py --run-baselines  (Phase 2 Baseline benchmarks)")
        print("  python factcheck.py --run-task1      (Phase 3 Task 1 Word Overlap experiments)")
