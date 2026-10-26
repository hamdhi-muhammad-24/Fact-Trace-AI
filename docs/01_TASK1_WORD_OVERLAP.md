# 01 – Member 1 Guide: Part 1 – Word Overlap

## Owner

**Member 1**

## Weight

```text
40 marks
```

## Primary Code Ownership

Member 1 primarily owns:

```text
WordRecallThresholdFactChecker
```

inside:

```text
factcheck.py
```

Member 1 may add required Word Overlap helper functions inside `factcheck.py`.

Do not create required runtime files such as:

```text
word_overlap.py
preprocessing.py
similarity.py
```

The final code submission is only `factcheck.py`.

---

# 1. Main Goal

Build a Word Overlap fact checker that receives:

```text
fact
+
passages
```

and returns:

```text
"S"
or
"NS"
```

The method must produce a real-valued lexical overlap/similarity score and use it to make the classification decision.

The coursework asks you to consider:

- tokenization
- stemming
- stop words
- overlap/similarity method
- classification threshold

The stated full-performance target is:

```text
at least 75% accuracy
```

---

# 2. Do Not Modify

Do not modify:

```text
factchecking_main.py
```

Do not modify:

```text
data/dev_labeled_ChatGPT.jsonl
data/passages_bm25_ChatGPT_humfacts.jsonl
```

Do not rewrite Member 2's entailment code.

---

# 3. Branch

Use:

```text
word-overlap
```

Before work:

```bash
git checkout develop
git pull origin develop
git checkout word-overlap
git merge develop
```

---

# 4. T1.1 – Understand the Actual Starter Code

Open `factcheck.py`.

Find:

```text
class WordRecallThresholdFactChecker(FactChecker):
```

Its `predict(...)` currently contains:

```text
raise Exception("Implement me")
```

This is your main Part 1 implementation point.

Understand the method signature:

```text
predict(fact, passages)
```

You must eventually return:

```text
"S"
or
"NS"
```

---

# 5. T1.2 – Understand `FactExample`

The actual starter code says:

```text
fact
```

is a string.

```text
passages
```

is a list of dictionaries.

Each passage has:

```text
"title"
"text"
```

The title is normally not needed.

The text is the retrieved evidence.

The passage text may not align with perfect sentence or paragraph boundaries.

For Part 1, first focus on lexical comparison between fact and passage text.

---

# 6. T1.3 – Understand the Evaluation

The actual `factchecking_main.py` converts:

```text
IR → NS
```

during evaluation.

Your Word Overlap model should only return:

```text
S
NS
```

Do not return `IR`.

---

# 7. T1.4 – Run the Initial Command

Before implementing:

```bash
python factchecking_main.py --mode word_overlap
```

It will fail because `WordRecallThresholdFactChecker.predict()` is unfinished.

This is expected.

Use this first run to confirm:

- environment is working
- data files are found
- program reaches your class
- no separate setup problem exists

---

# 8. T1.5 – Inspect Real Examples

Read several examples from the data through the program or carefully inspect the JSONL format.

Look at:

- short facts
- long facts
- names
- dates
- numbers
- punctuation
- repeated words
- common stop words
- long passage text
- multiple passages

Do not edit the dataset.

The purpose is to understand what your lexical method must handle.

---

# 9. T1.6 – Plan Preprocessing Experiments

The coursework explicitly asks you to make preprocessing decisions.

Plan a small number of meaningful variants.

Possible areas:

## Lowercasing

Question:

Should:

```text
London
london
```

match?

## Tokenization

Decide how to convert text into tokens/words.

## Punctuation

Decide how punctuation affects matching.

## Stop Words

Consider whether removing common words helps.

Examples:

```text
the
a
an
is
was
in
of
to
```

## Stemming

Consider whether stemming improves matching.

Do not assume one choice is automatically correct.

Your final choice must come from experiments.

---

# 10. T1.7 – Check Available Packages

Use only:

- Python standard library
- packages already in the lecturer's `requirements.txt`

Do not introduce a new package casually.

The actual provided requirements include, among others:

```text
numpy
torch
spacy
nltk
transformers
rouge-score
```

If you use NLTK/spaCy functionality that requires extra downloaded resources, ensure the final lecturer environment can run it.

Do not create avoidable environment dependencies.

---

# 11. T1.8 – Build a Preprocessing Helper

If you need helper functions, define them inside:

```text
factcheck.py
```

Keep them near the Word Overlap implementation or in a clearly shared helper area agreed by the team.

A preprocessing helper should:

- accept a string
- handle empty/short text safely
- produce consistent output
- avoid modifying global data
- be deterministic

Do not over-engineer it.

---

# 12. T1.9 – Implement One Baseline Similarity Method

The coursework allows lexical methods such as:

- TF-IDF cosine similarity
- Jaccard similarity
- precision/recall overlap
- ROUGE
- BLEU
- similar overlap scores

Start with one valid baseline.

Do not implement every metric at once.

Your baseline should produce:

```text
real-valued score
```

for fact vs evidence.

---

# 13. T1.10 – Decide Fact-Level Passage Scoring

A fact can have multiple passages.

You must determine how to obtain one fact-level decision from those passages.

Possible design questions include:

- score each passage separately?
- use the strongest supporting passage?
- compare against combined passage text?

The coursework does not prescribe one exact Word Overlap aggregation rule.

Therefore:

- choose a sensible approach
- test it
- record the result
- justify the final choice from evidence

Do not invent a "correct" rule without testing.

---

# 14. T1.11 – Add Threshold Classification

Convert the real-valued score into:

```text
S
NS
```

Conceptually:

```text
score
  ↓
threshold
  ↓
S / NS
```

Do not treat the first threshold as final.

---

# 15. T1.12 – Run First Baseline

Run:

```bash
python factchecking_main.py --mode word_overlap
```

Record:

```text
Experiment ID
Preprocessing
Similarity method
Passage aggregation
Threshold
Accuracy
Notes
```

This becomes the baseline.

---

# 16. T1.13 – Tune Threshold

The coursework explicitly says the threshold can be tuned on the development set.

Test a sensible range.

For each threshold record:

```text
threshold
accuracy
```

Do not invent values.

Do not only keep the best number; keep enough results to understand the trend.

---

# 17. T1.14 – Compare Preprocessing Variants

Compare meaningful alternatives such as:

```text
with stop-word removal
vs
without stop-word removal
```

and, if relevant:

```text
with stemming
vs
without stemming
```

and tokenization choices.

Use the same evaluation command.

Keep experiments controlled where possible.

---

# 18. T1.15 – Compare Suitable Similarity Methods

If time allows, compare another reasonable lexical method.

The coursework marking grid rewards justified comparison and tuning.

Do not add complexity only to make the project look larger.

A small number of well-run experiments is better than many unexplained methods.

---

# 19. T1.16 – Work Toward 75%

The stated target is:

```text
>= 75% accuracy
```

If below target, debug systematically:

1. preprocessing
2. score calculation
3. passage aggregation
4. threshold
5. return labels
6. possible implementation errors

Do not fake or manually force results.

---

# 20. T1.17 – Prepare Overlap for Part 2 Pruning

The coursework says Part 2 may use Word Overlap to prune very low-overlap evidence.

Therefore, Member 1 should leave a reusable overlap scoring function inside `factcheck.py`.

Member 2 should be able to call it.

Important:

The pruning threshold is not automatically the same as the Part 1 classification threshold.

Pruning must be tested separately by Member 2.

---

# 21. T1.18 – Avoid Merge Conflict with Member 2

Member 1 owns Word Overlap code.

Member 2 owns Entailment code.

If you need to change:

```text
imports
shared helper section
base classes
```

tell Member 2 first.

One person makes the shared edit.

Merge it.

Then the other member updates their branch.

---

# 22. T1.19 – Clean Final Part 1 Code

Before PR:

- remove debug prints
- remove dead experimental code
- keep only required final logic
- keep helper names understandable
- do not modify `factchecking_main.py`
- do not add runtime files
- do not add new dependency without team agreement

---

# 23. T1.20 – Final Part 1 Test

Run:

```bash
python factchecking_main.py --mode word_overlap
```

It must finish without errors.

Record the final real accuracy.

Do not claim the coursework target as your achieved accuracy unless it is actually achieved.

---

# 24. T1.21 – Merge Workflow

Before PR:

```bash
git checkout develop
git pull origin develop
git checkout word-overlap
git merge develop
```

Resolve only your own conflicts.

If conflict touches Member 2-owned entailment code, ask Member 2.

Then run the Part 1 command again.

Open PR to:

```text
develop
```

---

# 25. Member 1 Definition of Done

- [ ] `WordRecallThresholdFactChecker.predict()` implemented
- [ ] preprocessing implemented
- [ ] real-valued score implemented
- [ ] multiple passages handled
- [ ] S/NS classification implemented
- [ ] threshold experiments completed
- [ ] preprocessing experiments completed
- [ ] similarity choice tested/justified
- [ ] final configuration based on real results
- [ ] overlap logic reusable for pruning if needed
- [ ] no custom runtime Python file required
- [ ] no change to `factchecking_main.py`
- [ ] data unchanged
- [ ] merged into `develop`
- [ ] required command runs without errors

Required command:

```bash
python factchecking_main.py --mode word_overlap
```
