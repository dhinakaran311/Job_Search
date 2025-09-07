# backend/services/skill_extractor.py
"""
Skill extraction service:
- Attempts LLM-based extraction via extract_skills_llm from llm.py
- Falls back to deterministic lexicon-based matching when LLM unavailable/fails.
"""

from __future__ import annotations
import os
import re
from typing import List, Set
from dotenv import load_dotenv

load_dotenv()

USE_LLM_BY_DEFAULT = os.getenv("USE_LLM_FOR_SKILLS", "1") not in ("0", "false", "False")

# Local import; allow module to import even if llm has no dependencies at dev time.
try:
    from .llm import extract_skills_llm
except Exception:
    def extract_skills_llm(_: str) -> List[str]:
        return []

# Lexicon: multi-word first (greedy)
LEXICON_MULTI = [
    "machine learning", "deep learning", "data structures", "object oriented programming",
    "natural language processing", "computer vision", "time series",
    "cloud computing", "distributed systems", "test driven development",
    "continuous integration", "continuous delivery", "prompt engineering",
    "microservices", "rest api", "data engineering"
]

LEXICON_SINGLE = [
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "sql", "nosql",
    "docker", "kubernetes", "git", "linux", "bash", "aws", "azure", "gcp",
    "pytorch", "tensorflow", "numpy", "pandas", "scikit-learn", "react", "nodejs"
]

WORD_RE = re.compile(r"\b[a-z0-9#+\-]+\b", re.I)

def _lexicon_match(text: str) -> List[str]:
    txt = (text or "").lower()
    found = []
    used_spans = []

    # Multi-word greedy
    for phrase in LEXICON_MULTI:
        if phrase in txt:
            found.append(phrase)
            txt = txt.replace(phrase, " ")  # remove occurrences to avoid duplicates

    # Single tokens
    tokens = set(w.group(0).lower() for w in WORD_RE.finditer(txt))
    for tok in LEXICON_SINGLE:
        if tok.lower() in tokens and tok not in found:
            found.append(tok)

    # Deduplicate preserving order
    seen = set()
    out = []
    for s in found:
        norm = s.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out

def extract_skills(text: str, use_llm: bool = None) -> List[str]:
    """
    Try LLM-based extraction if allowed; otherwise use lexicon fallback.
    Returns normalized lower-case skills list.
    """
    allow_llm = USE_LLM_BY_DEFAULT if use_llm is None else bool(use_llm)
    if allow_llm:
        try:
            skills = extract_skills_llm(text)
            if skills:
                return skills
        except Exception:
            pass
    # Fallback
    return _lexicon_match(text or "")
