# 02 – Task 2 Guide: Textual Entailment with DeBERTa-v3

## 1. Overview & Weight

```text
Task:   Part 2 – Textual Entailment (NLI)
Weight: 40 Marks
Target: >= 83% accuracy (Goal: Maximum performance, aiming for 88-92%+)
File:   factcheck.py (The only submitted code file)
```

In this task, we move beyond surface-level word overlap to semantic reasoning. We use a deep transformer Natural Language Inference (NLI) model—**DeBERTa-v3-base** fine-tuned on MNLI, FEVER, and ANLI—to determine whether Wikipedia evidence logically entails the claim.

---

## 2. NLI Formulation & Architecture

We formulate fact verification as premise-hypothesis Natural Language Inference:
- **Premise ($P$)**: Candidate Wikipedia evidence sentence.
- **Hypothesis ($H$)**: Atomic factual claim.

The model maps the input pair $(P, H)$ to logits over 3 classes:
```text
Class 0: entailment
Class 1: neutral
Class 2: contradiction
```

The framework loads the pre-trained checkpoint inside `factchecking_main.py`:
```python
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
ent_tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta_ent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
Our job in `factcheck.py` is to implement:
1. `EntailmentModel.check_entailment(premise, hypothesis)`
2. `EntailmentFactChecker.predict(fact, passages)`

---

## 3. Step-by-Step Implementation Workflow

```text
Step 1: Passage Preprocessing & Sentence Segmentation
        Split 140-word Wikipedia passages into clean individual sentences using spaCy / regex
        ↓
Step 2: Lexical Evidence Pruning
        Compute word overlap between claim and each sentence; discard sentences below threshold gamma
        ↓
Step 3: Neural NLI Inference with DeBERTa-v3
        Run DeBERTa on remaining candidate sentences with torch.no_grad() and memory cleanup
        ↓
Step 4: Decision Strategy Comparison
        Evaluate Discrete Argmax vs Continuous Probability vs Margin Scoring
        ↓
Step 5: Max-Evidence Aggregation
        Take the maximum entailment score across all candidate sentences
        ↓
Step 6: Systematic Threshold Tuning & 5-Fold Cross-Validation
        Tune pruning threshold gamma and entailment threshold tau with cross-validation
        ↓
Step 7: Final Model Freeze & Evaluation
        Evaluate Confusion Matrix, Precision, Recall, F1 on S and NS
        ↓
Step 8: Verification Command
        Verify with: python factchecking_main.py --mode entailment
```

---

## 4. Passage Cleaning & Sentence Splitting

As discovered in our EDA, retrieved passages average **141 words** and contain multiple disparate propositions. Feeding an entire 140-word passage as the premise dilutes transformer attention, risks truncation, and introduces noise from unrelated sentences.

### Solution:
1. Strip Wikipedia metadata, wiki formatting, citation tags, and irregular whitespace.
2. Segment each passage into individual sentences using sentence boundary detection (`spacy` `en_core_web_sm` or robust regex).
3. Evaluate each sentence as an atomic premise candidate.

---

## 5. Word-Overlap Evidence Pruning (Speed & Precision)

Running a heavy 86-million parameter DeBERTa model across every sentence in 3 to 5 passages per fact is computationally expensive and slow on CPU. Furthermore, completely unrelated sentences can occasionally trigger false-positive neutral/entailment logits.

### Pruning Method:
Before passing a sentence to DeBERTa, compute its word overlap with the claim:
$$\text{overlap} = \text{compute\_word\_overlap\_score}(F, \text{sentence})$$

If $\text{overlap} < \gamma$ (pruning threshold, e.g. $0.15$–$0.25$), discard the sentence immediately:
- **Efficiency**: Discards 60%–75% of irrelevant sentences, reducing inference latency by over 3×.
- **Predictive Benefit**: Eliminates distractor sentences, reducing False Positives.

---

## 6. Binary Decision Strategies

We evaluate three distinct strategies for converting 3-class NLI outputs into binary `S` vs `NS`:

### Strategy A: Discrete Argmax
```python
pred_class = argmax([logits_entail, logits_neutral, logits_contradict])
is_supported = (pred_class == 0) # Entailment -> S, Neutral/Contradiction -> NS
```
- *Limitation*: Treats marginal 51% entailment the same as 99% confident entailment; lacks threshold tuning.

### Strategy B: Continuous Calibrated Entailment Probability (Recommended)
Compute probabilities via softmax:
$$P(\text{entailment}) = \frac{e^{z_{\text{entail}}}}{\sum_{j} e^{z_j}}$$
Decision:
$$\text{Score} = \max_{s \in \text{Sentences}} P(\text{entailment} \mid s, F)$$
$$\text{Predict } S \iff \text{Score} \ge \tau$$

### Strategy C: Margin Scoring (Entailment vs Contradiction)
$$\text{Score}(s) = P(\text{entailment}) - P(\text{contradiction})$$
Penalizes sentences that express contradictory evidence.

---

## 7. Systematic Threshold Tuning & 5-Fold Cross-Validation

We jointly optimize:
1. **Pruning threshold ($\gamma$)**: Range $[0.10, 0.35]$.
2. **Entailment threshold ($\tau$)**: Range $[0.50, 0.95]$.

We use 5-fold cross-validation on the 221 development facts to determine the optimal $(\gamma, \tau)$ without data leakage. The target is to exceed the 83% pass threshold and push overall accuracy to the highest possible performance (~88%–92%+).

---

## 8. Memory Management & OOM Prevention

To prevent Out-Of-Memory (OOM) errors during automated grading on CPU/GPU:
1. Always wrap inference in `torch.no_grad()`:
   ```python
   with torch.no_grad():
       outputs = self.model(**inputs)
   ```
2. Explicitly delete tensors and call garbage collection:
   ```python
   del inputs, outputs, logits
   if self.cuda:
       torch.cuda.empty_cache()
   gc.collect()
   ```
3. Use sensible batching when querying multiple candidate sentences.

---

## 9. Code Implementation Standards

- All code lives in `factcheck.py` inside:
  - `EntailmentModel.check_entailment(premise, hypothesis)`
  - `EntailmentFactChecker.predict(fact, passages)`
  - Helper functions for sentence splitting and pruning.
- Write short, clear, human-style comments:
  - `# Task 2: Sentence segmentation`
  - `# Task 2: Prune low-overlap sentences`
  - `# Task 2: DeBERTa forward pass`
- Verify with both CPU and optional GPU execution:
  ```bash
  python factchecking_main.py --mode entailment
  python factchecking_main.py --mode entailment --cuda
  ```
