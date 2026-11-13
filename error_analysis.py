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
