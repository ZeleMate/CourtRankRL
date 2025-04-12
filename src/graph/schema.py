"""
Defines the schema (node labels, relationship types, properties) for the legal graph database.
"""

# Node Labels
NODE_JUDGMENT = "Judgment"
NODE_STATUTE = "Statute"
NODE_REGULATION = "Regulation"
NODE_JUDGE = "Judge"
NODE_COURT = "Court"
NODE_KEYWORD = "Keyword" # Extracted keywords/topics
NODE_LEGAL_PRINCIPLE = "LegalPrinciple" # Abstract legal concepts

# Relationship Types
REL_CITES = "CITES" # Judgment cites Judgment/Statute/Regulation
REL_APPLIES = "APPLIES" # Judgment applies Statute/Regulation
REL_INTERPRETS = "INTERPRETS" # Judgment interprets Statute/Regulation
REL_PRESIDES_OVER = "PRESIDES_OVER" # Judge presides over Judgment
REL_HEARD_AT = "HEARD_AT" # Judgment heard at Court
REL_HAS_KEYWORD = "HAS_KEYWORD" # Judgment/Statute has Keyword
REL_RELATED_TO_PRINCIPLE = "RELATED_TO_PRINCIPLE" # Judgment/Statute related to LegalPrinciple
REL_AMENDS = "AMENDS" # Statute amends Statute

# Common Properties
PROP_ID = "id" # Unique identifier
PROP_TEXT_SNIPPET = "text_snippet" # Short relevant text
PROP_FULL_TEXT_REF = "full_text_ref" # Reference to full document
PROP_DATE = "date"
PROP_NAME = "name" # e.g., Judge name, Court name, Statute title

# Example Schema Structure (can be loaded from JSON/YAML if preferred)
SCHEMA = {
    "nodes": [
        {"label": NODE_JUDGMENT, "properties": [PROP_ID, PROP_DATE, PROP_FULL_TEXT_REF]},
        {"label": NODE_STATUTE, "properties": [PROP_ID, PROP_NAME, PROP_DATE]},
        {"label": NODE_JUDGE, "properties": [PROP_ID, PROP_NAME]},
        # ... other nodes
    ],
    "relationships": [
        {"type": REL_CITES, "start_node": NODE_JUDGMENT, "end_node": [NODE_JUDGMENT, NODE_STATUTE]},
        {"type": REL_APPLIES, "start_node": NODE_JUDGMENT, "end_node": NODE_STATUTE},
        # ... other relationships
    ]
}

def get_schema():
    """Returns the defined graph schema."""
    return SCHEMA

