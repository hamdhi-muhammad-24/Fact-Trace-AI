# 01 – Task 1 Guide: Word Overlap Fact-Checking

## 1. Overview & Weight

```text
Task:   Part 1 – Word Overlap Fact-Checking
Weight: 40 Marks
Target: >= 75% accuracy (Goal: Maximum empirical accuracy)
File:   factcheck.py (The only submitted code file)
```

In this task, we build a bag-of-words lexical classifier that predicts whether a factual claim is Supported (`S`) or Not Supported (`NS`) based on word overlap with retrieved Wikipedia passages.

---

## 2. Core Methodological Concept

Supported claims generally share essential vocabulary with supporting Wikipedia evidence, while unsupported claims lack grounding or contradict key entities.

However, fact-checking exhibits an extreme **Length Asymmetry**:
- Average atomic claim length: **~6.76 words**
- Average retrieved passage length: **~141.13 words**

Because passages are 20× longer than claims, symmetric set metrics (like Jaccard or Dice) divide by the massive passage length in their denominator, unfairly penalizing true claims. Asymmetric **Word Recall** measures claim grounding directly:
$$\text{Recall}(F, P) = \frac{|Tokens(F) \cap Tokens(P)|}{|Tokens(F)|}$$

---

## 3. Step-by-Step Implementation Workflow

```text
Step 1: Baseline Establishment
        Run Random Guess (~50.0%) and Always Supported (50.68%)
        ↓
Step 2: Preprocessing Ablation
        Investigate Tokenization, Lowercasing, Stopword Removal, Stemming, and Number-Awareness
        ↓
Step 3: Similarity Metric Comparison
        Evaluate Word Recall, Jaccard, Dice, and TF-IDF Cosine
        ↓
Step 4: Evidence Aggregation
        Apply max-pooling across all retrieved passages for a claim
        ↓
Step 5: Threshold Optimization & 5-Fold Cross-Validation
        Sweep threshold theta from 0.40 to 0.90; validate across 5 folds
        ↓
Step 6: Final Model Freeze & Evaluation
        Evaluate Confusion Matrix, Precision, Recall, F1 on S and NS (85.07% Accuracy)
        ↓
Step 7: Verification Command
        Verify with: python factchecking_main.py --mode word_overlap
```

---

## 4. Preprocessing Pipeline & Ablation Study

Text preprocessing choices significantly influence overlap accuracy. We conduct a systematic ablation across 5 configurations:

1. **Raw Tokens (P0)**: Lowercase regex alphanumeric tokens without stopword removal or stemming.
2. **Stopwords Removed (P1)**: Filter out high-frequency grammatical words (e.g. *the, is, in, at*) using NLTK English stopwords so they do not artificially inflate overlap scores.
3. **Porter Stemming (P2)**: Reduce inflected forms to root stems (e.g. *nominated, nominating, nominates* $\rightarrow$ *nomin*).
4. **Number & Date Verification (P3 - Final Model)**: Biographical claims frequently hinge on birth years, dates, and numbers. If numerical tokens in the claim are missing from the evidence, a penalty multiplier (0.5×) is applied.

### Experimental Ablation Results on Dev Set:
| Configuration | Best Threshold | Dev Accuracy | S F1 | NS F1 |
| :--- | :---: | :---: | :---: | :---: |
| Raw Tokens (No stopwords, no stemming) | 0.720 | 76.92% | 0.7733 | 0.7647 |
| Stopwords Removed | 0.610 | 79.19% | 0.7909 | 0.7928 |
| Stopwords Removed + Porter Stemming | 0.667 | 83.26% | 0.8370 | 0.8279 |
| **Stopwords + Stemming + Number Verification (Final)** | **0.660** | **85.07%** | **0.8520** | **0.8493** |

---

## 5. Metric Comparison

We evaluate four distinct lexical similarity metrics on the 221 development facts:

1. **Unigram Recall**: Fraction of claim words found in the passage:
   $$\text{Score} = \frac{|F \cap P|}{|F|}$$
2. **Jaccard Similarity**: Intersection divided by union:
   $$\text{Score} = \frac{|F \cap P|}{|F \cup P|}$$
3. **Dice Coefficient**: Harmonic overlap normalized by total tokens:
   $$\text{Score} = \frac{2 \times |F \cap P|}{|F| + |P|}$$
4. **TF-IDF Cosine Similarity**: Cosine angle between term frequency vectors:
   $$\text{Score} = \frac{\vec{v}_F \cdot \vec{v}_P}{\|\vec{v}_F\| \|\vec{v}_P\|}$$

### Experimental Metric Comparison Results:
| Metric | Best Threshold | Dev Accuracy | S F1 | NS F1 | Why it performs this way |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF Cosine** | 0.3397 | 66.97% | 0.5629 | 0.7345 | Passage vector magnitude dominates denominator. |
| **Dice Coefficient** | 0.0822 | 70.59% | 0.6857 | 0.7236 | Penalized by long passage length $|P|$. |
| **Jaccard Similarity** | 0.0429 | 71.04% | 0.6860 | 0.7308 | Union $|F \cup P|$ is dominated by passage vocabulary. |
| **Unigram Recall (Base)** | 0.6667 | 83.26% | 0.8370 | 0.8279 | Asymmetric grounding: measures claim coverage directly. |
| **Unigram Recall + Number Check** | **0.6600** | **85.07%** | **0.8520** | **0.8493** | Direct claim grounding with date/entity integrity. |

---

## 6. Multi-Passage Evidence Aggregation

A single claim is paired with 2 to 5 retrieved Wikipedia passages ($P_1, P_2, \dots, P_k$).

We evaluate passage aggregation using **max-pooling**:
$$\text{ClaimScore} = \max_{i \in \{1, \dots, k\}} \text{compute\_word\_overlap\_score}(F, P_i)$$

If any single retrieved passage provides sufficient grounding for the fact ($\text{ClaimScore} \ge \theta$), the claim is classified as Supported (`S`), otherwise Not Supported (`NS`). Max-pooling correctly mirrors human fact-checking: a single verified source passage is sufficient to confirm a claim.

---

## 7. Systematic Threshold Optimization & Cross-Validation

We evaluate classification thresholds from $\theta = 0.40$ to $0.90$:

| Threshold ($\theta$) | Accuracy | Correct | False Positives | False Negatives | S F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.40 | 73.76% | 163/221 | 53 | 5 | 0.7868 |
| 0.50 | 76.47% | 169/221 | 43 | 9 | 0.7984 |
| 0.60 | 81.90% | 181/221 | 25 | 15 | 0.8291 |
| 0.65 | 85.07% | 188/221 | 16 | 17 | 0.8520 |
| **0.66** | **85.07%** | **188/221** | **16** | **17** | **0.8520 (Optimal)** |
| 0.70 | 79.64% | 176/221 | 9 | 36 | 0.7716 |
| 0.80 | 75.11% | 166/221 | 7 | 48 | 0.6995 |

### 5-Fold Cross-Validation:
To confirm that threshold $\theta = 0.66$ does not overfit to the dev set, we conduct 5-fold cross-validation:
- Across all 5 validation folds, the optimal threshold consistently falls within $[0.64, 0.67]$.
- Mean out-of-fold accuracy: **84.62% ($\pm 1.2\%$)**, proving excellent generalization and stability.

---

## 8. Final Task 1 Model Performance (Freeze)

```text
Overall Accuracy : 188 / 221 = 85.07% (Exceeds the 75% target by +10.07%)

Confusion Matrix:
                 Predicted S    Predicted NS
Actual S  (112):      95             17        (Recall: 84.82%)
Actual NS (109):      16             93        (Recall: 85.32%)

Precision        :    85.59%         84.55%
F1-Score         :    0.8520         0.8493
```

---

## 9. Error Analysis & Lexical Limitations

Why does word overlap plateau at 85.07%?
The remaining 33 errors (14.93%) reveal fundamental limitations of bag-of-words matching:
1. **Paraphrase Mismatch (False Negatives)**: Words like *invented* vs *patented*, or *served as a pastor* vs *was an ordained minister*. Lexical matching misses synonyms.
2. **Syntactic & Entity Traps (False Positives)**: A person's name matching an incidental title (e.g. *Love, American Style* matching a marriage claim).
3. **Negation & Contradiction**: Identical vocabulary discussing a topic, but inverted by a single negation token (*never*, *not*).

These linguistic limits motivate **Task 2 (Textual Entailment with DeBERTa-v3)**.

---

## 10. Code Implementation Standards

- All logic is implemented in `factcheck.py` inside:
  - `tokenize_and_preprocess(text, remove_stopwords, stem)`
  - `compute_word_overlap_score(fact, passage, number_penalty)`
  - `compute_fact_passages_overlap(fact, passages, number_penalty)`
  - `WordRecallThresholdFactChecker(threshold=0.66).predict(fact, passages)`
- Use short, simple, human comments (e.g. `# Preprocessing: remove stopwords and stem`).
- Verify execution:
  ```bash
  python factchecking_main.py --mode word_overlap
  ```
