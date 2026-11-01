# 03 – Task 3 Guide: Qualitative Error Analysis

## 1. Overview & Weight

```text
Task:   Part 3 – Error Analysis
Weight: 20 Marks
Input:  Predictions from the final Part 2 Textual Entailment model
Deliverable: Qualitative Error Analysis section in the final PDF report
```

In this task, we perform a deep qualitative and quantitative investigation into why the final DeBERTa entailment model makes mistakes. We identify recurring linguistic, semantic, and structural failure modes, define fine-grained categories, compute aggregate statistics, and analyze three detailed case studies.

---

## 2. Core Concepts: Error Definitions

Remember that the evaluator converts `IR` (Irrelevant) to `NS` (Not Supported) during binary evaluation.

### Two Broad Error Types:
1. **False Positive (FP)**:
   - Ground Truth: `NS` (or `IR`)
   - Model Prediction: `S`
   - *Meaning*: The model claimed the fact is supported when Wikipedia evidence does not actually support it.
2. **False Negative (FN)**:
   - Ground Truth: `S`
   - Model Prediction: `NS`
   - *Meaning*: The fact is actually supported by the source, but the model failed to detect entailment.

---

## 3. Required Error Sample Size

Per coursework specifications:
- Manually examine **at least 10 False Positives** and **at least 10 False Negatives** (or all available errors if fewer exist).
- For each error, record:
  - Atomic fact text
  - Gold human label
  - Model prediction & confidence score
  - Retrieved passage text & candidate sentences
  - Linguistic/semantic cause of failure

---

## 4. Error Taxonomy (3–4 Fine-Grained Categories)

Do not use trivial labels like "wrong prediction" or "false positive". The categories must reflect genuine linguistic and semantic phenomena observed in the data:

### 1. Entity & Context Misalignment (False Positives)
- **Definition**: The evidence discusses a namesake, an artistic work (e.g. a book or movie title), or an unrelated person sharing the same surname/first name, leading the model to falsely infer that the attribute belongs to the biographical subject.
- **Example Pattern**: A claim about a marriage or occupation matching words in a television episode title (*"Love, American Style"*).

### 2. Lexical & World Knowledge Paraphrase Gap (False Negatives)
- **Definition**: The source text uses implicit phrasing, professional terminology, or synonyms that the NLI model fails to map to the claim proposition without external background knowledge.
- **Example Pattern**: *"served as a pastor"* vs Wikipedia text stating *"was ordained as a minister in the Presbyterian Church"*.

### 3. Numerical & Temporal Granularity Mismatch (False Positives / Negatives)
- **Definition**: Discrepancies involving specific calendar dates, birth years, durations, or quantities. The model either overlooks subtle digit mismatches (e.g. 1974 vs 1976) or penalizes approximate temporal phrasing (e.g. *"in the late 1970s"* vs *"in 1978"*).

### 4. Negation & Scope Inversion (False Positives)
- **Definition**: High semantic similarity between the premise and hypothesis where the premise contains a subtle negation operator, restrictive adverb (*never*, *rarely*, *failed to*), or conditional clause that flips the factual truth value.

---

## 5. Aggregate Statistics & Distribution

In the final report, summarize the distribution across examined errors:

| Error Category | False Positives (FP) | False Negatives (FN) | Total Errors | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| Entity & Context Misalignment | Actual | Actual | Actual | % |
| Paraphrase & World Knowledge Gap | Actual | Actual | Actual | % |
| Numerical & Temporal Mismatch | Actual | Actual | Actual | % |
| Negation & Scope Inversion | Actual | Actual | Actual | % |
| **Total Examined** | **>= 10** | **>= 10** | **>= 20** | **100%** |

Include a visual bar chart illustrating this distribution in the report.

---

## 6. Three Detailed Case Studies (Rubric Requirement)

For the three selected case studies, provide the following structured breakdown:

### Case Study 1: [Category Name]
- **Fact Text**: *"..."*
- **Ground Truth**: `S` or `NS`
- **Model Prediction**: `S` or `NS` (Confidence score: `0.xx`)
- **Wikipedia Evidence**: *"..."*
- **Fine-Grained Error Category**: `[Category Name]`
- **Root-Cause Analysis (1–3 sentences)**: Explain precisely why the language, syntax, or reasoning failed.

### Case Study 2: [Category Name]
- *(Same structured format)*

### Case Study 3: [Category Name]
- *(Same structured format)*

---

## 7. Critical Reflection & Future Improvements

For full Grade A credit, connect observed errors to fundamental model limitations and propose technical remedies:

1. **Cross-Sentence Coreference Resolution**:
   - *Limitation*: Passages often refer to the subject with pronouns (*"He"*, *"She"*, *"The actor"*) in subsequent sentences. Isolated sentence splitting can lose the subject referent.
   - *Improvement*: Apply neural coreference resolution (e.g. fastcoref) before sentence segmentation.
2. **Multi-Hop Evidence Synthesis**:
   - *Limitation*: Some facts require combining two separate sentences (e.g., Sentence 1 states birth in Munich; Sentence 2 states Munich is in Germany).
   - *Improvement*: Implement graph-based multi-hop retrieval or concatenating top-2 adjacent sentences.
3. **Structured Entity & Date Verification**:
   - *Limitation*: NLI models can treat numbers as generic tokens without strict numerical equality checks.
   - *Improvement*: Hybrid NLI with dedicated Named Entity Recognition (NER) and temporal constraint filters.

---

## 8. Code Implementation in `factcheck.py`

To ensure complete reproducibility with zero extra files, an automated error extraction function is embedded in `factcheck.py`:
- It compares the predictions of `EntailmentFactChecker` against gold labels.
- It extracts all FP and FN cases along with claim text, passage text, and prediction confidence.
- It prints formatted markdown tables ready to inspect and document in the report.
