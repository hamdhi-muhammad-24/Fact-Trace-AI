# FactTrace AI — Fact-Checking Coursework Implementation

**CM3604 Deep Learning Coursework Repository**

FactTrace AI implements automated factual claim verification over retrieved Wikipedia evidence passages using the FActScore ChatGPT benchmark dataset.

The project addresses all three required coursework components:
- **Part 1: Word Overlap Fact-Checking** — 40 marks (target: ≥ 75.0% accuracy; achieved: 85.07%)
- **Part 2: Textual Entailment Fact-Checking** — 40 marks (target: ≥ 83.0% accuracy; achieved: 84.16%)
- **Part 3: Manual & Empirical Error Analysis** — 20 marks (quantitative failure taxonomy & case studies)

---

## Repository Architecture

```text
Fact-Trace-AI/
├── factcheck.py             # Required submission module (self-contained, no matplotlib)
├── factchecking_main.py     # Official test driver (unchanged lecturer starter file)
├── requirements.txt         # Coursework dependencies specification (unchanged)
├── eda.py                   # Phase 1: Exploratory data analysis & data-quality audit
├── experiments.py           # Phases 2, 3, 5: Baselines, Task 1 grid, Task 2 pruning sweeps
├── error_analysis.py        # Task 3: Qualitative & quantitative failure taxonomy
├── model_comparison.py      # Phase 8: Cross-paradigm synthesis & evidence comparison
│
├── data/
│   ├── FACTSCORE-LICENSE.txt
│   ├── dev_labeled_ChatGPT.jsonl            # 221 labeled atomic claims across 31 biographies
│   └── passages_bm25_ChatGPT_humfacts.jsonl # BM25 retrieved Wikipedia passages corpus
│
├── outputs/
│   ├── eda/                 # summary.json & figures/ (8 charts)
│   ├── baselines/           # results.json & figures/ (2 charts)
│   ├── task1/               # experiments.json, results.json, comparisons.csv, predictions.csv, figures/ (7 charts)
│   ├── task2/               # experiments.json, results.json, comparisons.csv, predictions.csv, cache/, figures/ (7 charts)
│   ├── task3/               # error_records.json, error_review.csv, figures/ (3 charts)
│   └── comparison/          # overall_results.json, comparison.csv, figures/ (4 charts)
│
├── docs/
│   ├── CW.pdf               # Official coursework brief & Grade A assessment grid
│   └── IMPLEMENTATION_PLAN.md
│
├── README.md
└── .gitignore
```

---

## Required Environment

Use **Python 3.10**:

```bash
pip install -r requirements.txt
```

*(Matplotlib is only used by the modular experiment and visualization scripts, not by the submitted `factcheck.py`.)*

---

## Official Coursework Commands

The official evaluation driver executes without modification:

### Task 1: Word Overlap Fact-Checker
```bash
python factchecking_main.py --mode word_overlap
```
- **Configuration**: P4 Porter-Stemmed Number-Aware Word Recall (θ* = 0.64, Max aggregation, Disjoint number penalty λ = 0.0)
- **Coursework Target**: ≥ 75.0% accuracy
- **Result**: **188 / 221 = 85.07%** (Balanced Accuracy: 85.07%, Macro-F1: 0.8507)

### Task 2: DeBERTa-v3 Textual Entailment Fact-Checker
```bash
python factchecking_main.py --mode entailment
```
*(Optional GPU acceleration: `python factchecking_main.py --mode entailment --cuda`)*
- **Configuration**: Sentence-level evidence units, lexical pruning (γ = 0.05, top_k = all, conservative fallback), probability thresholding (τ* = 0.33)
- **Coursework Target**: ≥ 83.0% accuracy
- **Result**: **186 / 221 = 84.16%** (Balanced Accuracy: 84.28%, Macro-F1: 0.8408)

### Baseline Comparisons
```bash
python factchecking_main.py --mode always_entail
python factchecking_main.py --mode random
```

---

## Experimental Suite & Evaluation Methodology

The experimental suite maintains clear distinctions across evaluation modalities:
- **Development Set Evaluation**: Full evaluation over the labeled development set (221 facts) matching `factchecking_main.py`.
- **Grouped Cross-Validation**: Topic-grouped 5-fold CV to evaluate out-of-fold generalization across unseen biographical topics.
- **Cached NLI Experimental Evaluation**: Fast, reproducible ablation and configuration sweeps leveraging cryptographically validated inference caches.
- **Live DeBERTa Inference**: Direct neural forward-pass execution in `factchecking_main.py` using HuggingFace Transformers and PyTorch.
- **Recorded Historical CPU Benchmarks**: Documented wall-clock runtimes and throughput measured on standard CPU hardware.

To reproduce all exploratory analysis, experimental grids, cross-validation, and error analyses:

```bash
# 1. Exploratory Data Analysis & Quality Audit (generates outputs/eda/)
python eda.py

# 2. Baseline Benchmarks & Monte Carlo simulations (outputs/baselines/)
python experiments.py --baselines

# 3. Task 1 120-Configuration Grid Search & Nested Grouped 5-Fold CV (outputs/task1/)
python experiments.py --task1

# 4. Task 2 DeBERTa-v3 Formulation & Pruning Sweep (outputs/task2/)
python experiments.py --task2

# 5. Verify Cryptographic NLI Cache Integrity (15 negative corruption tests)
python experiments.py --test-cache

# 6. Run All Benchmarks at once
python experiments.py --all

# 7. Task 3 Failure Mode Taxonomy & Error Analysis (outputs/task3/)
python error_analysis.py

# 8. Overall Cross-Paradigm Comparison & Master Audit (outputs/comparison/)
python model_comparison.py
```

### Note on NLTK WordNet
The winning Task 1 P4 classifier in `factcheck.py` uses algorithmic `PorterStemmer` and requires no external downloads. If running the full Task 1 experimental suite with the P3 lemmatization ablation via `experiments.py --task1`, install the WordNet corpus once:
```bash
python -c "import nltk; nltk.download('wordnet')"
```

---

## Coursework Submission Details

As per coursework guidelines, submission consists of:
1. **`factcheck.py`**: The clean root implementation file implementing all required classes (`WordRecallThresholdFactChecker`, `EntailmentFactChecker`, etc.).
2. **PDF Written Report**: Coursework documentation including error analysis, ablation studies, and experimental findings.
