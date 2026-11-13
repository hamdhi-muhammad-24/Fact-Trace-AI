# factcheck.py
'''
CM3604 Deep Learning Coursework — Modular Fact-Checking Library
Base interfaces, data structures, and evaluation metrics.
'''
import os, sys, json, re
from typing import List, Dict, Set, Tuple, Any, Optional

class FactExample(object):
    '''Represents a single fact to verify with associated retrieved Wikipedia passages.'''
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr('fact=' + repr(self.fact) + '; label=' + repr(self.label))

class FactChecker(object):
    '''Base fact checker class.'''
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception('Don\'t call me, call my subclasses')

def load_passages_data(path: str) -> Tuple[Dict[str, List[dict]], List[dict]]:
    '''Loads retrieved Wikipedia passages from JSONL file.'''
    fact_to_passage_dict = {}
    raw_records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            record = json.loads(line_str)
            raw_records.append(record)
            fact_to_passage_dict[record['sent']] = record.get('passages', [])
    return fact_to_passage_dict, raw_records

def load_dev_examples(labeled_facts_path: str, fact_to_passage_dict: Dict[str, List[dict]]) -> Tuple[List[FactExample], List[dict]]:
    '''Loads labeled development facts and maps associated passages.'''
    examples = []
    raw_bios = []
    with open(labeled_facts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            bio = json.loads(line_str)
            raw_bios.append(bio)
            if bio.get('annotations') is not None:
                for sent in bio['annotations']:
                    if sent.get('human-atomic-facts') is not None:
                        for fact in sent['human-atomic-facts']:
                            fact_text = fact.get('text', '')
                            if fact_text in fact_to_passage_dict:
                                examples.append(FactExample(fact_text, fact_to_passage_dict[fact_text], fact.get('label', '')))
    return examples, raw_bios

def calculate_confusion_matrix(gold_labels: List[str], pred_labels: List[str]) -> List[List[int]]:
    '''Computes 2x2 confusion matrix [[TN, FP], [FN, TP]] for S and NS labels.'''
    label_indexer = ['S', 'NS']
    confusion_mat = [[0, 0], [0, 0]]
    for g, p in zip(gold_labels, pred_labels):
        if g in label_indexer and p in label_indexer:
            confusion_mat[label_indexer.index(g)][label_indexer.index(p)] += 1
    return confusion_mat

def calculate_metrics(confusion_mat: List[List[int]]) -> Dict[str, Any]:
    '''Calculates accuracy, precision, recall, and macro F1 from confusion matrix.'''
    s_tp = confusion_mat[0][0]
    s_fn = confusion_mat[0][1]
    s_fp = confusion_mat[1][0]
    s_tn = confusion_mat[1][1]
    total = s_tp + s_fn + s_fp + s_tn
    accuracy = (s_tp + s_tn) / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat,
        'total_examples': total
    }

class RandomGuessFactChecker(FactChecker):
    """Baseline: Predicts S or NS uniformly at random."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def predict(self, fact: str, passages: List[dict]) -> str:
        if self.rng is not None:
            return str(self.rng.choice(["S", "NS"]))
        return str(np.random.choice(["S", "NS"]))

class AlwaysEntailedFactChecker(FactChecker):
    """Majority Baseline: Always predicts Supported (S)."""
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"

_CANONICAL_ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "each", "few", "for", "from", "further", "had",
    "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your"
}

def get_english_stopwords() -> Set[str]:
    return set(_CANONICAL_ENGLISH_STOPWORDS)

from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_tokens(text: str, mode: str = "stem",
                      stemmer: Optional[PorterStemmer] = None,
                      lemmatizer: Optional[WordNetLemmatizer] = None,
                      stops: Optional[Set[str]] = None) -> List[str]:
    """Tokenizes, lowercases, removes punctuation and stopwords, and applies stemming."""
    if not text: return []
    clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = clean.split()
    if stops:
        tokens = [t for t in tokens if t not in stops]
    if mode == "stem":
        st = stemmer or PorterStemmer()
        return [st.stem(t) for t in tokens]
    elif mode == "lemma":
        lm = lemmatizer or WordNetLemmatizer()
        return [lm.lemmatize(t) for t in tokens]
    return tokens
