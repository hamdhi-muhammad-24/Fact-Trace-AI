# 03 – Member 3 Guide: Part 3 – Error Analysis

## Owner

**Member 3**

## Weight

```text
20 marks
```

## Main Output

Part 3 produces the written error-analysis content for the final PDF.

Member 3 should normally not add runtime Python files.

The final required code file remains:

```text
factcheck.py
```

---

# 1. Main Goal

Analyse errors from the final Part 2 entailment system.

Two broad error types:

## False Positive

```text
Predicted: S
Gold:      NS
```

## False Negative

```text
Predicted: NS
Gold:      S
```

Remember:

The provided evaluator converts:

```text
IR → NS
```

for two-class evaluation.

---

# 2. Required Error Sample

The coursework requires manual examination of:

```text
10 false positives
10 false negatives
```

If fewer than 10 exist for a type:

```text
analyse all available errors of that type
```

---

# 3. Required Categories

Create around:

```text
2–4 fine-grained error categories
```

The categories must describe meaningful recurring patterns.

Do not use trivial categories like:

```text
False Positive
False Negative
Wrong Prediction
```

The categories must come from the actual final model errors.

---

# 4. Required PDF Content

The written submission must include:

## Category Definitions

```text
1–2 sentences per category
```

## Aggregate Statistics

How many examined errors belong to each category.

## Three Detailed Examples

For each example:

```text
(a) example text
(b) ground-truth label
(c) predicted label
(d) fine-grained category
(e) 1–3 sentence explanation
```

---

# 5. Branch

Use:

```text
feature/error-analysis
```

Member 3 should avoid editing Part 1/Part 2 core code unless specifically agreed.

---

# 6. T3.1 – Understand the Actual Evaluation

Read:

```text
factchecking_main.py
```

Understand:

```text
converted_label = "NS" if example.label == "IR" else example.label
```

This means the final evaluation classes are:

```text
S
NS
```

Make sure your FP/FN logic uses the same two-class interpretation.

---

# 7. T3.2 – Understand Prediction Output

Ask Members 1 and 2 to explain:

```text
predict(...)
```

output.

Both models return:

```text
"S"
or
"NS"
```

Member 3 must be able to compare:

```text
gold label
vs
prediction
```

correctly.

---

# 8. T3.3 – Start Experiment Tracking Early

Do not wait for final Part 2.

Maintain one simple shared experiment record.

Suggested fields:

```text
Experiment ID
Part
Configuration
Threshold
Accuracy
Runtime notes
Memory notes
Comments
```

Only enter real values.

Do not invent results.

---

# 9. T3.4 – Prepare Error Record Format

Prepare a consistent record format:

```text
Error ID
Fact
Gold label
Predicted label
FP/FN
Evidence
Entailment score if available
Fine-grained category
Notes
```

Do not create final categories yet.

---

# 10. T3.5 – Test the Collection Process on Early Models

When Member 2 has an early working model:

- run it
- identify a few wrong predictions
- verify FP/FN logic
- verify evidence can be inspected
- verify records are understandable

Do not use these early errors as final analysis unless the team later confirms the model version is final.

---

# 11. T3.6 – Integration Testing Role

Member 3 helps keep `develop` healthy.

After major merges, test:

```bash
python factchecking_main.py --mode word_overlap
```

and:

```bash
python factchecking_main.py --mode entailment
```

Report:

- crashes
- broken imports
- wrong file paths
- unexpected outputs
- memory errors

Do not silently change another member's core code.

---

# 12. T3.7 – Confirm Final Entailment Version

Before collecting final errors, all three members must agree:

```text
This is the final Part 2 configuration.
```

Record:

- evidence processing strategy
- binary decision strategy
- threshold if used
- pruning setting if used
- final accuracy

Use errors only from this final version for the written analysis.

---

# 13. T3.8 – Collect False Positives

Collect:

```text
10 FP
```

or all available if fewer.

For each FP keep:

- fact
- relevant evidence
- gold label
- prediction
- useful score if available

---

# 14. T3.9 – Collect False Negatives

Collect:

```text
10 FN
```

or all available if fewer.

Use the same format.

---

# 15. T3.10 – Manual Review by All Three Members

All members should read the selected errors.

For each error ask:

- what does the claim say?
- what does the evidence actually say?
- why might the model have failed?
- is the same type of failure repeated?
- is it lexical, semantic, structural, numerical, entity-related, or another real pattern?

These are prompts for thinking.

Do not force categories before examining examples.

---

# 16. T3.11 – Create 2–4 Categories

After review, create around:

```text
2–4 categories
```

Every category should represent a repeated pattern found in the actual examples.

A category should help answer:

```text
Why did the model make this kind of mistake?
```

Do not create categories only to satisfy the number requirement.

---

# 17. T3.12 – Assign Every Examined Error

Assign each selected FP/FN to a fine-grained category.

Check:

- categories are consistent
- definitions do not overlap too much
- category assignment makes sense
- counts will be reproducible

---

# 18. T3.13 – Write Category Definitions

For each category:

```text
1–2 sentences
```

Explain:

- what pattern it represents
- why that pattern is difficult for the model

Use your own analysis.

---

# 19. T3.14 – Calculate Aggregate Statistics

Count:

```text
number of errors in each category
```

Check:

```text
sum of category counts
=
number of examined errors
```

Do not invent numbers.

---

# 20. T3.15 – Choose Three Detailed Examples

Choose three useful examples.

Each must include:

```text
example text
ground-truth label
predicted label
category
1–3 sentence explanation
```

Use examples that clearly demonstrate important patterns.

---

# 21. T3.16 – Explain Examples Properly

Do not write only:

```text
The model predicted incorrectly.
```

Explain:

- what information the claim requires
- what the evidence provides
- where the mismatch/difficulty appears
- why the selected category applies

Keep explanations concise and evidence-based.

---

# 22. T3.17 – Prepare PDF Structure

Keep the PDF focused on Part 3.

Suggested simple structure:

```text
Title

1. Error Analysis Method
2. Fine-Grained Error Categories
3. Aggregate Error Statistics
4. Three Detailed Error Examples
5. Short Discussion of Main Model Limitations
```

Do not turn the PDF into a large unrelated software-project report.

---

# 23. T3.18 – Team Review

## Member 1 checks

- lexical-overlap related explanations
- evidence wording observations

## Member 2 checks

- entailment/model explanations
- predicted labels/scores

## Member 3 checks

- category consistency
- counts
- required example fields
- PDF completeness

All three approve the final content.

---

# 24. T3.19 – Repository Cleanliness Check

Member 3 can help with the final repository check.

Confirm:

- no unnecessary custom Python files
- no unnecessary runtime folders
- no `.venv/`
- no model cache
- no modified dataset
- no modified `factchecking_main.py`
- `factcheck.py` contains required custom runtime logic

---

# 25. T3.20 – Final Required Command Check

Run from clean project root:

```bash
python factchecking_main.py --mode word_overlap
```

Then:

```bash
python factchecking_main.py --mode entailment
```

Both must complete without errors.

---

# 26. T3.21 – Final Submission Check

Required separate files:

```text
1. PDF written submission
2. factcheck.py
```

Do not submit:

```text
whole Git repo
docs folder
README
.venv
FActScore-main repository
```

unless the lecturer separately asks for them.

---

# 27. Member 3 Definition of Done

- [ ] final Part 2 version confirmed
- [ ] FP sample collected
- [ ] FN sample collected
- [ ] up to 10 of each manually examined
- [ ] all three members reviewed errors
- [ ] 2–4 meaningful categories created
- [ ] category definitions written
- [ ] aggregate counts correct
- [ ] three detailed examples complete
- [ ] PDF complete
- [ ] both required commands pass
- [ ] repository clean
- [ ] final `factcheck.py` ready
- [ ] final PDF ready
