# 02 – Member 2 Guide: Part 2 – Textual Entailment

## Owner

**Member 2**

## Weight

```text
40 marks
```

## Primary Code Ownership

Member 2 primarily owns:

```text
EntailmentModel
EntailmentFactChecker
```

inside:

```text
factcheck.py
```

Member 2 may add required entailment/pruning helper functions inside `factcheck.py`.

Do not create required runtime files such as:

```text
entailment.py
deberta_utils.py
pruning.py
```

The final code submission is only `factcheck.py`.

---

# 1. Main Goal

Use the provided pretrained DeBERTa-v3 model to determine whether the fact is:

```text
S
or
NS
```

The actual framework loads:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

inside `factchecking_main.py`.

Do not manually add a model folder.

---

# 2. Actual Required Areas

The starter code already contains inference boilerplate in:

```text
EntailmentModel.check_entailment(...)
```

but currently stops at:

```text
raise Exception("Not implemented")
```

The starter code also contains:

```text
EntailmentFactChecker.predict(...)
```

which currently stops at:

```text
raise Exception("Implement me")
```

These are the main Part 2 areas.

---

# 3. Do Not Modify

Do not modify:

```text
factchecking_main.py
```

Do not modify the data files.

Do not rewrite Member 1's Word Overlap code without agreement.

---

# 4. Branch

Use:

```text
feature/entailment
```

Before work:

```bash
git checkout develop
git pull origin develop
git checkout feature/entailment
git merge develop
```

---

# 5. T2.1 – Run Initial Entailment Command

Run CPU first:

```bash
python factchecking_main.py --mode entailment
```

It will fail because Part 2 is unfinished.

Use this run to confirm:

- data loads
- Hugging Face model loading starts
- execution reaches your incomplete code
- environment is otherwise correct

Optional GPU:

```bash
python factchecking_main.py --mode entailment --cuda
```

Use only if CUDA is correctly available.

Final solution must not require CUDA.

---

# 6. T2.2 – Understand Premise and Hypothesis

For this task:

```text
Premise    = evidence/source text
Hypothesis = fact
```

Do not swap them accidentally.

The existing call:

```text
self.tokenizer(premise, hypothesis, ...)
```

shows the intended order.

---

# 7. T2.3 – Understand Existing Model Inference

The starter code already:

1. tokenizes premise and hypothesis
2. uses truncation and padding
3. moves tensors to CUDA if requested
4. performs inference under `torch.no_grad()`
5. obtains:

```text
outputs.logits
```

Do not unnecessarily rewrite working boilerplate.

Your task starts from interpreting the output and returning useful results.

---

# 8. T2.4 – Understand Class Order

The starter code comment says:

```text
["entailment", "neutral", "contradiction"]
```

Your implementation must map logits/probabilities carefully.

Do not assume a different class order.

Understand what each value means before writing final logic.

---

# 9. T2.5 – Decide What `check_entailment()` Returns

You need a return format that is useful to `EntailmentFactChecker`.

Possible useful information may include:

- discrete predicted class
- entailment probability
- neutral probability
- contradiction probability

The coursework allows either:

- discrete entailment classification
- score/threshold-based classification

Design the return value so it supports the experiments you actually intend to run.

Keep it inside `factcheck.py`.

---

# 10. T2.6 – Preserve Memory Cleanup

The starter code includes cleanup intended to reduce OOM risk.

It explicitly deletes:

```text
inputs
outputs
logits
```

and calls:

```text
gc.collect()
```

Do not place your return before cleanup in a way that makes cleanup unreachable.

Do not remove cleanup casually.

If you create additional large tensors/results, release them when no longer needed.

---

# 11. T2.7 – Implement One Pair Inference

First make:

```text
one premise
+
one hypothesis
```

produce a valid entailment result.

Do not immediately build the entire multi-passage system.

Verify:

- no crash
- output probabilities/classes look sensible
- returned values are usable

---

# 12. T2.8 – Inspect Real Passage Text

The starter code says passage `text` may not align with clean sentence or paragraph boundaries.

Inspect several real passages.

Decide how you will provide appropriate text units to DeBERTa.

Possible considerations:

- whole passage
- sentence-level pieces
- simple cleaning
- empty text
- long text/truncation

The marking grid values suitable sentence splitting and cleaning.

Your team must decide the exact method experimentally.

---

# 13. T2.9 – Implement Evidence Processing

Keep evidence processing:

- simple
- reproducible
- compatible with available packages
- inside `factcheck.py`

Do not create a separate custom processing module.

If using an NLP tokenizer/sentence splitter that requires external downloaded resources, confirm it will work in the lecturer environment.

Avoid unnecessary runtime dependencies.

---

# 14. T2.10 – Implement Discrete Baseline

The coursework allows a discrete strategy.

General coursework rule:

If using the discrete entailment classifier, a fact may be treated as entailed if any checked part of any passage is entailed.

Implement a correct baseline based on the actual model class outputs.

Return:

```text
"S"
or
"NS"
```

from:

```text
EntailmentFactChecker.predict(...)
```

---

# 15. T2.11 – Evaluate Discrete Baseline

Run:

```bash
python factchecking_main.py --mode entailment
```

Record:

```text
Experiment ID
Evidence processing
Decision strategy
Accuracy
Runtime notes
Memory notes
```

This is your baseline.

---

# 16. T2.12 – Implement Score-Based Strategy

The coursework also allows probability/threshold decisions.

If using scores:

1. obtain entailment score for each checked evidence unit
2. use the maximum entailment score for the fact
3. compare it with a threshold
4. return S/NS

The coursework explicitly says the fact score should be the maximum over entailment scores when this strategy is used.

---

# 17. T2.13 – Tune Entailment Threshold

Test a sensible range.

Record:

```text
threshold
accuracy
```

Do not hard-code an untested value.

The final threshold must come from real experiments.

---

# 18. T2.14 – Compare Discrete vs Score-Based

Compare:

```text
discrete strategy
vs
score/threshold strategy
```

Use real measurements.

Consider:

- accuracy
- runtime
- stability
- failure patterns

Choose the final method based on evidence.

---

# 19. T2.15 – Handle Multiple Passages

A fact may have several passages.

Ensure all relevant evidence is handled according to your chosen strategy.

For score-based logic:

```text
fact score = maximum entailment score
```

over the checked evidence units, as required by the brief.

Avoid accidental early stopping that ignores stronger evidence later, unless your selected strategy logically allows it.

---

# 20. T2.16 – Work Toward 83%

The stated full-performance target is:

```text
>= 83% accuracy
```

If below target, inspect:

1. class mapping
2. premise/hypothesis order
3. evidence splitting
4. cleaning
5. max-score logic
6. threshold
7. implementation bugs

Do not invent performance.

---

# 21. T2.17 – Integrate Member 1 Word Overlap Pruning

Do this after Member 1 provides a stable reusable overlap score.

Concept:

```text
fact + evidence
      ↓
overlap score
      ↓
very low overlap?
  ├── yes → skip DeBERTa
  └── no  → run DeBERTa
```

The coursework says this can reduce runtime because very low-overlap evidence is unlikely to entail the fact.

Important:

The pruning threshold is a separate purpose from Part 1's S/NS classification threshold.

Do not automatically reuse the same value.

Test it.

---

# 22. T2.18 – Compare With and Without Pruning

Measure:

```text
without pruning
vs
with pruning
```

Record:

- accuracy
- runtime
- memory behavior
- amount of evidence skipped if practical

Keep pruning only if it is justified.

---

# 23. T2.19 – OOM / Reliability Test

The coursework says code must run without out-of-memory problems.

Test full entailment mode.

Check:

- CPU mode
- optional GPU mode if available
- memory cleanup
- large passage handling
- repeated calls
- no retained unnecessary tensors

Do not make GPU mandatory.

---

# 24. T2.20 – Avoid Merge Conflict with Member 1

Member 2 owns Entailment sections.

Member 1 owns Word Overlap sections.

If adding an import or shared helper:

1. tell Member 1
2. decide temporary owner
3. make one shared edit
4. merge it
5. update branches

Do not independently rewrite Member 1's functions.

---

# 25. T2.21 – Clean Final Part 2 Code

Before PR:

- remove debug prints
- remove temporary experiments
- keep final code readable
- keep memory cleanup
- keep all required code in `factcheck.py`
- do not add runtime Python modules
- do not change `factchecking_main.py`
- do not change data

---

# 26. T2.22 – Final Part 2 Test

Run:

```bash
python factchecking_main.py --mode entailment
```

This required command must run without errors.

Also ensure Part 1 still runs after integration:

```bash
python factchecking_main.py --mode word_overlap
```

---

# 27. T2.23 – Merge Workflow

Before PR:

```bash
git checkout develop
git pull origin develop
git checkout feature/entailment
git merge develop
```

If conflict is inside Word Overlap code, ask Member 1.

Run both required commands after conflict resolution.

Open PR to:

```text
develop
```

---

# 28. Member 2 Handoff to Member 3

When final entailment configuration is fixed, Member 3 needs:

- final predictions
- gold labels
- fact text
- relevant evidence
- score if available
- final configuration description

All error categories must come from actual final errors.

---

# 29. Member 2 Definition of Done

- [ ] `EntailmentModel.check_entailment()` completed
- [ ] `EntailmentFactChecker.predict()` completed
- [ ] class mapping understood
- [ ] premise/hypothesis correct
- [ ] evidence processing implemented
- [ ] multiple passages handled
- [ ] discrete baseline evaluated
- [ ] score-based strategy evaluated if used
- [ ] threshold tuned if used
- [ ] final strategy based on real results
- [ ] pruning evaluated if used
- [ ] OOM risk handled
- [ ] CPU command works
- [ ] no custom runtime Python file required
- [ ] `factchecking_main.py` unchanged
- [ ] data unchanged
- [ ] merged into `develop`
- [ ] both required commands still work

Required command:

```bash
python factchecking_main.py --mode entailment
```
