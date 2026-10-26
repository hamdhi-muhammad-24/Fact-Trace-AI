# FactTrace AI

**CM3604 Deep Learning Group Coursework Development Repository**

FactTrace AI is the internal project name for our coursework implementation.

The coursework checks whether factual claims are supported by retrieved Wikipedia evidence using:

- **Part 1:** Word Overlap
- **Part 2:** Textual Entailment using DeBERTa-v3
- **Part 3:** Error Analysis

---

## Read First

All team members should read:

- [`docs/00_COMMON_PROJECT_GUIDE.md`](docs/00_COMMON_PROJECT_GUIDE.md)

Then read the guide for your task:

- **Member 1:** [`docs/01_TASK1_WORD_OVERLAP.md`](docs/01_TASK1_WORD_OVERLAP.md)
- **Member 2:** [`docs/02_TASK2_ENTAILMENT.md`](docs/02_TASK2_ENTAILMENT.md)
- **Member 3:** [`docs/03_TASK3_ERROR_ANALYSIS.md`](docs/03_TASK3_ERROR_ANALYSIS.md)

---

## Important Repository Rule

Keep the lecturer's starter project structure clean.

Do **not** create unnecessary Python files, folders, services, models, utilities, pipelines, or other runtime structure.

The final code submission is only:

```text
factcheck.py
```

Therefore, all custom code required for **Part 1 and Part 2** must work from `factcheck.py` with the lecturer's unchanged framework.

Do not make `factcheck.py` depend on custom Python files that will not be submitted.

Documentation files under `docs/` are for the team only and are not required at runtime.

---

## Actual Starter Framework

```text
Fact-Trace-AI/
│
├── factchecking_main.py
├── factcheck.py
├── requirements.txt
│
├── data/
│   ├── FACTSCORE-LICENSE.txt
│   ├── dev_labeled_ChatGPT.jsonl
│   └── passages_bm25_ChatGPT_humfacts.jsonl
│
├── docs/
│   ├── 00_COMMON_PROJECT_GUIDE.md
│   ├── 01_TASK1_WORD_OVERLAP.md
│   ├── 02_TASK2_ENTAILMENT.md
│   └── 03_TASK3_ERROR_ANALYSIS.md
│
├── README.md
└── .gitignore
```

`.venv/` may exist locally, but it must be ignored by Git.

---

## Required Environment

Use:

```text
Python 3.10
```

Install exactly from the lecturer's framework `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Required Commands

Before submission, both commands must run without errors:

```bash
python factchecking_main.py --mode word_overlap
```

```bash
python factchecking_main.py --mode entailment
```

Optional GPU form:

```bash
python factchecking_main.py --mode entailment --cuda
```

Use `--cuda` only when PyTorch CUDA support and a CUDA-capable GPU are correctly available.

---

## Final Coursework Submission

The coursework asks for two separate files:

1. **PDF written submission** for Part 3 Error Analysis
2. **`factcheck.py`** for Parts 1 and 2

The whole Git repository is not the required submission.

---

## Academic / GenAI Rule

The coursework brief marks GenAI use as **Category 3 – Editing**.

These Markdown files are only internal planning and understanding guides.

The team must make its own assessed decisions and produce its own experimental evidence for:

- preprocessing
- tokenization
- stemming
- stop-word handling
- similarity method
- thresholds
- entailment decision strategy
- pruning threshold
- error categories
- final analysis and conclusions
