# factcheck.py

import os
import sys
import json
import re
import math
import gc
import time
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import torch
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer


#
# STARTER CLASSES & FRAMEWORK DATA STRUCTURES (READ-ONLY INTERFACES)
#

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
    """
    Wrapper for HuggingFace DeBERTa-v3 NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
    Provides memory-safe single and batched inference with explicit tensor lifecycle management.
    """
    def __init__(self, model, tokenizer, cuda: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda
        if self.model is not None:
            self.model.eval()
            id2label = getattr(self.model.config, 'id2label', {0: 'entailment', 1: 'neutral', 2: 'contradiction'})
            self.ent_idx = 0
            self.neu_idx = 1
            self.contra_idx = 2
            for idx, label in id2label.items():
                lbl = str(label).lower()
                if 'entail' in lbl:
                    self.ent_idx = int(idx)
                elif 'contra' in lbl:
                    self.contra_idx = int(idx)
                elif 'neut' in lbl:
                    self.neu_idx = int(idx)
        else:
            self.ent_idx = 0
            self.neu_idx = 1
            self.contra_idx = 2

    def check_entailment(self, premise: str, hypothesis: str, max_length: int = 256) -> Tuple[float, float, float]:
        """
        Computes entailment probabilities for a single (premise, hypothesis) pair.
        Returns (p_entailment, p_neutral, p_contradiction).
        """
        res = self.check_entailment_batch([premise], [hypothesis], batch_size=1, max_length=max_length)
        return res[0]

    def check_entailment_batch(self, premises: List[str], hypotheses: List[str],
                               batch_size: int = 8, max_length: int = 256) -> List[Tuple[float, float, float]]:
        """
        Memory-safe batched CPU inference with torch.no_grad() and explicit tensor lifecycle cleanup.
        Returns a list of (p_entailment, p_neutral, p_contradiction) tuples.
        """
        if not premises:
            return []
        results = []
        with torch.no_grad():
            for i in range(0, len(premises), batch_size):
                b_prem = premises[i:i+batch_size]
                b_hyp = hypotheses[i:i+batch_size]
                inputs = self.tokenizer(b_prem, b_hyp, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
                if self.cuda:
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()
                for p in probs:
                    results.append((float(p[self.ent_idx]), float(p[self.neu_idx]), float(p[self.contra_idx])))
                del inputs, outputs
                gc.collect()
        return results


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


#
# TASK 1: LEXICAL PROCESSING & SIMILARITY FUNCTIONS
#

_CANONICAL_ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't",
    "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re",
    "s", "same", "shan", "shan't", "she", "she'd", "she'll", "she's", "should", "should've", "shouldn", "shouldn't", "so",
    "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "ve", "very", "was", "wasn", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

_LEMMATIZER_INSTANCE: Optional[WordNetLemmatizer] = None

def get_english_stopwords() -> Set[str]:
    """
    Returns the deterministic canonical English stopword set with negation words preserved (195 words).
    Guarantees 100% reproducible preprocessing across all runtime environments
    without requiring automatic network downloads or environment-dependent fallbacks.
    """
    return _CANONICAL_ENGLISH_STOPWORDS


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
    - P0: Lowercased alphanumeric regex tokens (\\b[a-zA-Z0-9]+\\b).
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
    Default configuration is experimentally optimized:
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


#
# TASK 2: SENTENCE SPLITTING, EVIDENCE PRUNING & ENTAILMENT CLASSIFIER
#

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences deterministically without requiring external model downloads.
    Protects common abbreviations, honorifics, single initials, and decimal numbers.
    """
    if not text or not text.strip():
        return []
    cleaned = text.strip()
    protected = cleaned
    abbrs = [
        r'\bMr\.', r'\bMrs\.', r'\bMs\.', r'\bDr\.', r'\bProf\.', r'\bSr\.', r'\bJr\.',
        r'\bSt\.', r'\bAve\.', r'\bRd\.', r'\bGen\.', r'\bGov\.', r'\bCapt\.', r'\bCol\.',
        r'\bLt\.', r'\bRep\.', r'\bSen\.', r'\be\.g\.', r'\bi\.e\.', r'\bvs\.', r'\bU\.S\.',
        r'\bU\.K\.', r'\bJan\.', r'\bFeb\.', r'\bMar\.', r'\bApr\.', r'\bAug\.', r'\bSept\.',
        r'\bOct\.', r'\bNov\.', r'\bDec\.', r'\bno\.', r'\bvol\.', r'\bpp\.'
    ]
    for abbr in abbrs:
        protected = re.sub(abbr, lambda m: m.group(0).replace('.', '§§DOT§§'), protected, flags=re.IGNORECASE)
    protected = re.sub(r'\b([A-Z])\.', r'\1§§DOT§§', protected)
    protected = re.sub(r'(\d)\.(\d)', r'\1§§DOT§§\2', protected)
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])', protected)
    sents = []
    for s in raw_sents:
        restored = s.replace('§§DOT§§', '.').strip()
        if restored:
            sents.append(restored)
    return sents if sents else [cleaned]


def build_evidence_units(passages: List[dict], mode: str = "sentence") -> List[str]:
    """
    Constructs candidate evidence premise units from retrieved passages.
    Supported modes:
      - 'passage': unsegmented raw text of retrieved passages.
      - 'sentence': individual sentences segmented via rule-based splitter.
      - 'window2': sliding window of 2 consecutive sentences to maintain local discourse context.
      - 'title_sentence': sentence prepended with passage Wikipedia title ("Title: Sentence").
    """
    units = []
    for p in passages:
        title = p.get("title", "").strip()
        text = p.get("text", "").strip()
        if not text:
            continue
        if mode == "passage":
            units.append(text)
        elif mode == "sentence":
            for s in split_into_sentences(text):
                if s.strip():
                    units.append(s.strip())
        elif mode == "title_sentence":
            prefix = f"{title}: " if title else ""
            for s in split_into_sentences(text):
                if s.strip():
                    units.append(f"{prefix}{s.strip()}")
        elif mode == "window2":
            sents = split_into_sentences(text)
            if len(sents) <= 1:
                if sents:
                    units.append(sents[0])
            else:
                for i in range(len(sents) - 1):
                    units.append(f"{sents[i]} {sents[i+1]}")
        else:
            raise ValueError(f"Unknown formulation mode '{mode}'. Expected 'passage', 'sentence', 'window2', or 'title_sentence'.")
    return units


def get_tokens_for_pruning(text: str) -> List[str]:
    """
    Extracts lowercased alphanumeric tokens excluding canonical stopwords.
    """
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = clean.split()
    stops = get_english_stopwords()
    return [t for t in tokens if t not in stops]


def compute_overlap_recall(claim_tokens: List[str], unit_tokens: Set[str]) -> float:
    """
    Computes asymmetric token recall: fraction of claim content words appearing in evidence unit.
    """
    if not claim_tokens:
        return 0.0
    matches = sum(1 for t in claim_tokens if t in unit_tokens)
    return matches / len(claim_tokens)


def prune_candidate_units(fact: str, units: List[str], gamma: float = 0.05,
                          top_k: Optional[int] = None, fallback: str = "conservative") -> List[str]:
    """
    Prunes candidate evidence units based on token recall against the claim.
    - Filters units with recall < gamma.
    - Limits to top_k highest-scoring candidates (if specified).
    - Fallback:
        - 'conservative': if all units fall below gamma, retains the top-1 highest recall unit.
        - 'strict': if all units fall below gamma, returns empty list.
    """
    if not units:
        return []
    fact_tokens = get_tokens_for_pruning(fact) or fact.lower().split()
    scored = []
    for u in units:
        u_toks = set(get_tokens_for_pruning(u))
        rec = compute_overlap_recall(fact_tokens, u_toks)
        scored.append((u, rec))
    scored.sort(key=lambda x: x[1], reverse=True)

    passed = [u for u, score in scored if score >= gamma]
    if passed:
        if top_k is not None and top_k > 0:
            return passed[:top_k]
        return passed

    if fallback == "conservative":
        return [scored[0][0]] if scored else []
    else:
        return []


class EntailmentFactChecker(FactChecker):
    """
    Task 2: Textual Entailment Fact-Checker using MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.
    Default parameters reflect our verified Phase 5 winning All-Retained configuration:
      - Evidence Formulation: Sentence-level segmentation ('sentence')
      - Lexical Candidate Pruning: gamma = 0.05 (filters irrelevant sentences, 69.0% candidate reduction)
      - Candidate Selection: top_k = None (evaluates all candidate sentences passing gamma)
      - Pruning Fallback: 'conservative' (retains top-1 candidate if all fall below gamma)
      - Decision Strategy: 'threshold' (entailment probability thresholding)
      - Decision Threshold: tau* = 0.33 (calibrated for balanced accuracy and F1)
      - Batch Size: B = 8 (optimal CPU memory and throughput)
      - Max Sequence Length: 256 tokens (covers 100% of claims and candidate sentences without truncation)
    """
    def __init__(self, ent_model, formulation: str = "sentence", decision_rule: str = "threshold",
                 threshold: float = 0.33, prune_gamma: float = 0.05, top_k: Optional[int] = None,
                 fallback: str = "conservative", batch_size: int = 8, max_length: int = 256):
        self.ent_model = ent_model
        self.formulation = formulation
        self.decision_rule = decision_rule
        self.threshold = threshold
        self.prune_gamma = prune_gamma
        self.top_k = top_k
        self.fallback = fallback
        self.batch_size = batch_size
        self.max_length = max_length

    def predict(self, fact: str, passages: List[dict]) -> str:
        units = build_evidence_units(passages, mode=self.formulation)
        candidate_units = prune_candidate_units(
            fact, units, gamma=self.prune_gamma, top_k=self.top_k, fallback=self.fallback
        )
        if not candidate_units:
            return "NS"

        hypotheses = [fact] * len(candidate_units)
        scores = self.ent_model.check_entailment_batch(
            candidate_units, hypotheses, batch_size=self.batch_size, max_length=self.max_length
        )

        if self.decision_rule == "argmax":
            is_s = any(p_ent > p_neu and p_ent > p_contra for p_ent, p_neu, p_contra in scores)
            return "S" if is_s else "NS"
        elif self.decision_rule == "threshold":
            max_ent = max(p_ent for p_ent, p_neu, p_contra in scores)
            return "S" if max_ent >= self.threshold else "NS"
        elif self.decision_rule == "margin":
            max_margin = max(p_ent - p_contra for p_ent, p_neu, p_contra in scores)
            return "S" if max_margin >= self.threshold else "NS"
        else:
            raise ValueError(f"Unknown decision rule '{self.decision_rule}'. Must be 'argmax', 'threshold', or 'margin'.")


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


#
# DATASET LOADING HELPERS (OFFICIAL FORMAT INTERFACES)
#

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


#
# EVALUATION METRICS HELPERS
#

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


#
# MAIN ENTRY POINT
#

if __name__ == "__main__":
    print("=" * 70)
    print("CM3604 Deep Learning Coursework — Fact-Checking Submission Module")
    print("=" * 70)
    print("This file contains the production model implementations for submission.")
    print("\nTo evaluate models according to coursework specification:")
    print("  python factchecking_main.py --mode word_overlap")
    print("  python factchecking_main.py --mode entailment")
    print("  python factchecking_main.py --mode always_entail")
    print("  python factchecking_main.py --mode random")
    print("\nTo run experiments, ablations, cross-validation, and generate figures:")
    print("  python experiments.py --all")
    print("  python eda.py")
    print("  python error_analysis.py")
    print("  python model_comparison.py")
    print("=" * 70)
