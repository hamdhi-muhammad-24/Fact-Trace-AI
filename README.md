# FactTrace AI

FactTrace AI verifies factual claims against retrieved Wikipedia passages using the FActScore ChatGPT benchmark dataset. Developed for CM3604 Deep Learning coursework, it includes a word overlap classifier, a textual entailment classifier, and supporting experiments and error analysis.

Predictions use two labels: `S` (supported) and `NS` (not supported).

## Project structure

```text
Fact-Trace-AI/
|-- factcheck.py           # Classifiers, evidence processing, and shared helpers
|-- factchecking_main.py   # Evaluation driver
|-- eda.py                 # Dataset exploration and data-quality checks
|-- experiments.py         # Baselines, configuration sweeps, and cross-validation
|-- error_analysis.py      # Error evidence, manual review, and analysis figures
|-- model_comparison.py    # Comparison tables and figures
|-- requirements.txt
|-- data/                  # Labeled claims, retrieved passages, and dataset license
|-- docs/CW.pdf            # Coursework brief
|-- outputs/
|   |-- eda/
|   |-- baselines/
|   |-- task1/
|   |-- task2/             # Includes NLI inference caches
|   |-- task3/
|   `-- comparison/
`-- README.md
```

Use the root `factcheck.py` for the current implementation.

## Setup

Use Python 3.10 and run commands from the repository root. Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To generate analysis figures, also install Matplotlib, which is not included in `requirements.txt`:

```powershell
python -m pip install matplotlib
```

The default word overlap classifier uses Porter stemming. The experimental lemmatization variant also requires the NLTK WordNet corpus:

```powershell
python -c "import nltk; nltk.download('wordnet')"
```

The entailment model is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Its first use requires internet access to download the model and tokenizer. CPU inference is supported; `--cuda` requires a compatible GPU and CUDA-enabled PyTorch installation.

## Evaluate the classifiers

```powershell
# Word overlap classifier
python factchecking_main.py --mode word_overlap

# Textual entailment classifier
python factchecking_main.py --mode entailment

# Optional GPU inference
python factchecking_main.py --mode entailment --cuda

# Baselines
python factchecking_main.py --mode always_entail
python factchecking_main.py --mode random
```

The default input files are:

- `data/dev_labeled_ChatGPT.jsonl`
- `data/passages_bm25_ChatGPT_humfacts.jsonl`

To evaluate other files, use the driver's `--labels_path` and `--passages_path` options.

## Recorded development results

The saved results cover 221 labeled claims. These are development-set measurements used during configuration selection; cross-validation results are recorded separately in the output files.

| Classifier | Correct | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Word overlap | 188 / 221 | 85.07% | 0.8507 |
| Textual entailment | 186 / 221 | 84.16% | 0.8408 |

The default word overlap configuration uses P4 preprocessing, Porter stemming, numerical consistency checks, maximum passage recall, and a threshold of `0.64`.

The default entailment configuration uses sentence-level evidence, lexical pruning at `0.05`, all retained candidates, a conservative fallback, and an entailment probability threshold of `0.33`.

See [Task 1 results](outputs/task1/results.json) and [Task 2 results](outputs/task2/results.json) for metrics, configurations, and cross-validation records. Runtime measurements depend on hardware and whether inference caches are reused.

## Run the analysis scripts

```powershell
# Dataset exploration
python eda.py

# Baseline experiments
python experiments.py --baselines

# Word overlap experiments and grouped cross-validation
python experiments.py --task1

# Entailment experiments; generate missing inference caches as needed
python experiments.py --task2

# Force fresh entailment inference
python experiments.py --task2 --rebuild-cache

# Run baselines, Task 1, and Task 2 together
python experiments.py --all

# Validate cache integrity with corruption tests
python experiments.py --test-cache

# Build error-review artifacts from Task 2 results
python error_analysis.py

# Compare saved system results
python model_comparison.py
```

Run the baseline and task experiments before the downstream analysis scripts when regenerating results. `--all` runs the three experiment groups; run EDA, error analysis, and model comparison separately. Commands write results and figures under `outputs/`.

## Error review status

The saved Task 2 predictions contain 35 errors: 8 false positives and 27 false negatives. The current Task 3 artifacts mark all 35 as pending manual review. Suggested error categories are provisional and should not be reported as confirmed human findings.

Use `outputs/task3/error_review.csv` to record manual decisions and `outputs/task3/error_evidence.csv` to inspect retrieved evidence, then rerun `python error_analysis.py` to refresh summaries. The script preserves existing manual reviews and checks their alignment with the predictions.

## Dataset license

See [data/FACTSCORE-LICENSE.txt](data/FACTSCORE-LICENSE.txt) for the supplied dataset license.
