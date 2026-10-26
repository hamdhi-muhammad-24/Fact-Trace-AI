# 00 – Common Project Guide

## 1. Purpose

This guide is for all three members.

Read this before writing code.

It explains:

- what the coursework actually requires
- what the provided framework already does
- which files we are allowed to change
- the exact project structure
- how Parts 1, 2 and 3 connect
- how three members can work in parallel
- how to avoid Git conflicts
- what must be submitted
- what must run before submission

---

# 2. Main Coursework Goal

The coursework is a fact-verification task.

The data contains factual claims extracted from ChatGPT-generated biographies.

Each claim has retrieved Wikipedia passages.

The goal is to decide whether the fact is:

```text
S  = Supported
NS = Not Supported
```

The original dataset may also contain:

```text
IR = Irrelevant
```

The actual provided `factchecking_main.py` converts `IR` to `NS` during two-class evaluation.

Therefore, our models should predict only:

```text
S
NS
```

Do not create an `IR` prediction class.

---

# 3. Coursework Parts

```text
Part 1 – Word Overlap          40 marks
Part 2 – Textual Entailment    40 marks
Part 3 – Error Analysis        20 marks
```

---

# 4. Final Submission

The coursework asks us to submit these files separately:

```text
1. PDF written submission
2. factcheck.py
```

This has a very important design consequence:

> All custom code needed for Parts 1 and 2 must work from `factcheck.py` inside the lecturer's framework.

Do not make the final solution depend on extra custom Python modules.

---

# 5. Actual Lecturer Framework

The uploaded framework ZIP contains exactly:

```text
Fact-Trace-AI/
│
├── factchecking_main.py
├── factcheck.py
├── requirements.txt
│
└── data/
    ├── FACTSCORE-LICENSE.txt
    ├── dev_labeled_ChatGPT.jsonl
    └── passages_bm25_ChatGPT_humfacts.jsonl
```

For teamwork, we may add documentation only:

```text
docs/
README.md
.gitignore
```

These documentation files must not be required for runtime.

---

# 6. Final Recommended Development Structure

Use:

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

Local only:

```text
.venv/
```

Do not push `.venv/`.

---

# 7. Do Not Add Unnecessary Project Structure

Do not create unnecessary runtime folders such as:

```text
src/
utils/
helpers/
models/
services/
pipeline/
experiments/
backend/
frontend/
api/
controllers/
```

Do not create custom runtime modules such as:

```text
word_overlap.py
preprocessing.py
similarity.py
entailment.py
deberta_utils.py
pruning.py
evaluation.py
```

if `factcheck.py` would need to import them.

Why?

Because only `factcheck.py` is submitted for Parts 1 and 2.

The lecturer must be able to place our submitted `factcheck.py` into the original framework and run the required commands.

Helper functions are allowed, but keep required custom helper functions inside `factcheck.py`.

---

# 8. `factchecking_main.py`

The actual starter file:

- loads passages
- loads labelled facts
- creates `FactExample` objects
- selects the requested mode
- loads DeBERTa for entailment mode
- runs predictions
- prints evaluation statistics

## Important Rule

Do not modify:

```text
factchecking_main.py
```

The coursework explicitly says not to modify it.

Our submitted `factcheck.py` must work with this unchanged file.

---

# 9. Actual Data Paths

The provided `factchecking_main.py` uses:

```text
data/dev_labeled_ChatGPT.jsonl
```

and:

```text
data/passages_bm25_ChatGPT_humfacts.jsonl
```

Do not move these files.

Do not rename them.

Do not modify the dataset.

---

# 10. `FactExample`

The provided `FactExample` stores:

```text
fact
passages
label
```

Each passage is a dictionary with:

```text
title
text
```

The comment in the starter code says:

- `title` is normally not needed
- `text` is a chunk of text
- the text may not align with clean sentence/paragraph boundaries

That point matters for Part 2 sentence/evidence handling.

---

# 11. Actual `factcheck.py` Structure

The provided file contains:

```text
FactExample
EntailmentModel
FactChecker
RandomGuessFactChecker
AlwaysEntailedFactChecker
WordRecallThresholdFactChecker
EntailmentFactChecker
DependencyRecallThresholdFactChecker
```

The final class is marked:

```text
# OPTIONAL
```

Therefore:

```text
DependencyRecallThresholdFactChecker
```

is not one of the required Part 1/Part 2 implementations unless the teaching team gives another instruction.

Do not spend required project time on optional work before Parts 1–3 are complete.

---

# 12. Actual Unfinished Required Areas

## Part 1

The starter code contains:

```text
WordRecallThresholdFactChecker.predict(...)
```

with:

```text
raise Exception("Implement me")
```

This is the main required Part 1 implementation area.

---

## Part 2 – Entailment Model Output

The starter code contains:

```text
EntailmentModel.check_entailment(...)
```

The model inference boilerplate is already present.

After obtaining logits, it currently contains:

```text
raise Exception("Not implemented")
```

Member 2 must complete how the model output is interpreted.

The starter comment states the label order is:

```text
["entailment", "neutral", "contradiction"]
```

Still, Member 2 should understand the code and output carefully before implementing the final decision logic.

---

## Part 2 – Fact-Level Prediction

The starter code also contains:

```text
EntailmentFactChecker.predict(...)
```

with:

```text
raise Exception("Implement me")
```

This is where fact-level evidence checking and final S/NS prediction must be implemented.

---

# 13. How the Two Required Commands Work

## Command 1

```bash
python factchecking_main.py --mode word_overlap
```

Execution flow:

```text
factchecking_main.py
        ↓
loads data
        ↓
creates FactExample objects
        ↓
WordRecallThresholdFactChecker()
        ↓
factcheck.py
        ↓
predict(fact, passages)
        ↓
"S" or "NS"
        ↓
evaluation statistics
```

So Member 1 must make `WordRecallThresholdFactChecker.predict()` work.

---

## Command 2

```bash
python factchecking_main.py --mode entailment
```

Execution flow:

```text
factchecking_main.py
        ↓
loads DeBERTa tokenizer/model
        ↓
creates EntailmentModel
        ↓
creates EntailmentFactChecker
        ↓
factcheck.py
        ↓
checks evidence
        ↓
"S" or "NS"
        ↓
evaluation statistics
```

So Member 2 must make:

```text
EntailmentModel.check_entailment(...)
```

and:

```text
EntailmentFactChecker.predict(...)
```

work correctly.

---

# 14. DeBERTa Model

The actual framework loads:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

inside `factchecking_main.py`.

Do not manually download a model into the repository.

Do not change the model unless the coursework specifically allows and your team intentionally experiments with an allowed alternative.

The required final command must work with the provided framework.

---

# 15. CUDA

The starter code supports:

```bash
python factchecking_main.py --mode entailment --cuda
```

This moves the model and model inputs to CUDA.

But the required final command is also:

```bash
python factchecking_main.py --mode entailment
```

without CUDA.

Therefore, do not make the final solution require a GPU.

---

# 16. Memory / OOM Rule

The starter `EntailmentModel` already contains cleanup logic using:

```text
del ...
gc.collect()
```

The comment explains this is for preventing out-of-memory issues during autograding.

Do not remove memory cleanup without a very good reason.

Part 2 must run without causing OOM problems.

---

# 17. Exact Framework Requirements

Use the provided framework `requirements.txt`.

It contains:

```text
numpy==2.0.2
torch==2.5.0
torchvision==0.20.0
tqdm==4.66.5
spacy==3.8.2
nltk==3.9.1
transformers==4.45.2
sentencepiece==0.2.0
protobuf==5.28.3
rouge-score==0.1.2
```

Do not use the separate FActScore repository's `requirements.txt`.

The coursework says to use Python 3.10 and the specified package versions.

---

# 18. Team Division

## Member 1

Owns:

```text
Part 1 – Word Overlap
```

Primary code area:

```text
WordRecallThresholdFactChecker
```

Member 1 may add required Word Overlap helper functions inside `factcheck.py`.

---

## Member 2

Owns:

```text
Part 2 – Textual Entailment
```

Primary code areas:

```text
EntailmentModel
EntailmentFactChecker
```

Member 2 may add required entailment/pruning helper functions inside `factcheck.py`.

---

## Member 3

Owns:

```text
Part 3 – Error Analysis
```

Member 3 should normally avoid changing core Part 1/Part 2 implementation.

Member 3 can:

- track experiments
- test integrated builds
- collect FP/FN examples
- coordinate final error analysis
- coordinate the PDF

---

# 19. Can All Three Work in Parallel?

Yes.

Use this model:

```text
                    START
                      │
              Common setup
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
   Member 1        Member 2       Member 3
   Word Overlap    Entailment     Evaluation Prep
       │              │              │
       ▼              ▼              ▼
   Tune Part 1     Build Part 2   Track/Test
       │              │              │
       └──────┐       │       ┌──────┘
              ▼       ▼       ▼
                 Integration
                      │
              Optional pruning
                      │
              Final predictions
                      │
                Error Analysis
```

Member 2 does not need to wait for Member 1.

Member 2 can first build entailment without pruning.

Later, Word Overlap can be used for pruning if useful.

Member 3 can start evaluation preparation before final predictions exist.

---

# 20. Git Branch Structure

Recommended internal team workflow:

```text
main
│
└── develop
    │
    ├── word-overlap
    ├── entailment
    └── error-analysis
```

This Git structure is our team recommendation, not a coursework requirement.

---

# 21. Code Ownership to Avoid Conflicts

The physical file is shared:

```text
factcheck.py
```

Therefore, branch separation alone is not enough.

Use code ownership.

## Member 1 owns

```text
WordRecallThresholdFactChecker
Word Overlap helper functions
Word Overlap threshold/classification logic
```

## Member 2 owns

```text
EntailmentModel
EntailmentFactChecker
Entailment helper functions
Pruning integration
```

## Shared areas

If both members need to modify:

```text
imports
shared helper area
shared base code
```

one person should temporarily own that edit.

Never independently rewrite the same shared block.

---

# 22. Import Rule

Because `factcheck.py` is the submitted code file:

Before adding any new package import, check:

1. Is it already in Python standard library?
2. Is it already listed in the lecturer's `requirements.txt`?
3. Is the import really necessary?

Do not casually add dependencies that are not in the provided environment.

---

# 23. Daily Git Workflow

Before work:

```bash
git checkout develop
git pull origin develop
```

Then Member 1:

```bash
git checkout word-overlap
git merge develop
```

Member 2:

```bash
git checkout entailment
git merge develop
```

Member 3:

```bash
git checkout error-analysis
git merge develop
```

Work on a small subtask.

Commit clearly.

Push.

Do not wait until the entire coursework is finished before integrating.

---

# 24. Merge Rules

Before merging to `develop`:

- pull latest `develop`
- merge `develop` into your feature branch
- resolve conflicts with the relevant code owner
- run the relevant command
- make sure `factchecking_main.py` is unchanged
- make sure data files are unchanged
- request another member to review
- then merge

---

# 25. Part 1 Requirement Summary

Member 1 must:

- decide preprocessing
- consider tokenization
- consider stemming
- consider stop words
- produce a real-valued overlap score
- choose/test a suitable lexical metric
- tune classification threshold
- evaluate accuracy
- aim for at least 75% for the stated full-performance target
- make the overlap logic reusable if pruning is used later

Detailed guide:

```text
docs/01_TASK1_WORD_OVERLAP.md
```

---

# 26. Part 2 Requirement Summary

Member 2 must:

- understand premise/hypothesis
- complete model output handling
- convert entailment/neutral/contradiction into S/NS
- handle multiple evidence chunks/sentences
- use max entailment score if using score-based classification
- compare/tune decision strategy
- aim for at least 83% for the stated full-performance target
- optionally use Word Overlap pruning
- prevent OOM problems

Detailed guide:

```text
docs/02_TASK2_ENTAILMENT.md
```

---

# 27. Part 3 Requirement Summary

Member 3 must coordinate:

- 10 false positives, or all if fewer
- 10 false negatives, or all if fewer
- manual examination
- 2–4 meaningful error categories
- category definitions
- aggregate statistics
- three detailed examples
- final PDF

Detailed guide:

```text
docs/03_TASK3_ERROR_ANALYSIS.md
```

---

# 28. Required Final Commands

Before submission:

```bash
python factchecking_main.py --mode word_overlap
```

must run without errors.

Then:

```bash
python factchecking_main.py --mode entailment
```

must run without errors.

Do not treat only one successful command as enough.

---

# 29. Final Repository Cleanliness Checklist

Before finalizing the repository:

- [ ] `factchecking_main.py` unchanged
- [ ] `factcheck.py` contains all required custom runtime logic
- [ ] `requirements.txt` is the lecturer version
- [ ] both JSONL files remain inside `data/`
- [ ] `FACTSCORE-LICENSE.txt` remains in `data/`
- [ ] no unnecessary custom Python modules
- [ ] no unnecessary project folders
- [ ] no downloaded model folder committed
- [ ] no `.venv/` committed
- [ ] no cache files committed
- [ ] no dataset modifications
- [ ] both required commands run successfully

---

# 30. Final Submission Checklist

Submit separately:

```text
1. Error Analysis PDF
2. factcheck.py
```

Do not submit the whole Git repository unless the lecturer gives an additional instruction.

---

# 31. GenAI / Academic Integrity Rule

The coursework marks GenAI use as Category 3 – Editing.

These guides are planning and understanding documents only.

The group must independently make and justify assessed decisions such as:

- preprocessing
- similarity metric
- threshold values
- entailment strategy
- pruning threshold
- error categories
- interpretation of results

Do not invent results.

Do not copy generated conclusions as if they came from experiments.
