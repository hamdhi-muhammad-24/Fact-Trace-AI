# error_analysis.py
'''CM3604 Deep Learning Coursework — Task 3 Error Analysis Workflow'''
import os, sys, json, csv

if __name__ == '__main__':
    print('Error analysis module initialized.')

CATEGORY_DEFINITIONS: List[Dict[str, str]] = [
    {
        "category_id": "PRAGMATIC_GAP",
        "title": "Meaning and Negation Errors",
        "short_label": "Meaning and\nNegation Errors",
        "definition": "The model may misinterpret word meanings, descriptive or evaluative expressions, sentence relationships, qualifications, or negation clauses."
    },
    {
        "category_id": "RETRIEVAL_MISSING",
        "title": "Missing or Conflicting Evidence",
        "short_label": "Missing or Conflicting\nEvidence",
        "definition": "Retrieved passages lack required information to establish the claim, or contain different or conflicting details across retrieved biographical text."
    },
    {
        "category_id": "STRUCTURAL_FORMAT",
        "title": "Structured Information Errors",
        "short_label": "Structured Information\nErrors",
        "definition": "Relevant information appears in bibliographies, lists, tables, or section headings that may be difficult for sentence-level textual entailment."
    },
    {
        "category_id": "ANNOTATION_NOISE",
        "title": "Annotation and Evidence Conflicts",
        "short_label": "Annotation and\nEvidence Conflicts",
        "definition": "Available evidence appears to support an interpretation differing from the original human annotation (e.g., IR converted to NS) or context-dependent labeling."
    }
]

CATEGORY_ID_TO_TITLE = {c["category_id"]: c["title"] for c in CATEGORY_DEFINITIONS}
CATEGORY_TITLE_TO_ID = {c["title"]: c["category_id"] for c in CATEGORY_DEFINITIONS}

def clean_evidence_text(text: str) -> str:
    '''Removes sentence markers <s> and </s> and normalizes whitespace.'''
    if not text: return ""
    cleaned = re.sub(r"\s*</?s>\s*", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()

def map_selected_premise_to_passages(selected_premise: str, passages: list) -> dict:
    '''Matches selected premise sentence back to exact retrieved passage title.'''
    target = (selected_premise or "").strip()
    if not passages or not target:
        return {"selected_evidence_title": "Not reliably identified", "passage_matches": ["Uncertain"] * len(passages)}
    matched_indices = []
    for p_idx, p in enumerate(passages):
        raw_text = p.get("text", "")
        if target in raw_text:
            matched_indices.append(p_idx)
    if len(matched_indices) == 1:
        hit = matched_indices[0]
        title = str(passages[hit].get("title", "")).strip() or "Not reliably identified"
        return {"selected_evidence_title": title, "passage_matches": ["Yes" if i == hit else "No" for i in range(len(passages))]}
    return {"selected_evidence_title": "Not reliably identified", "passage_matches": ["Uncertain"] * len(passages)}
