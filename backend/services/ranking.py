"""
Ranking helpers: fused score and bucket assignment.
"""

from typing import Literal

def fused_score(similarity: float, overlap: float, w_sim: float = 0.7, w_overlap: float = 0.3) -> float:
    """
    Compute fused score from similarity and skill overlap.
    similarity, overlap expected in [0,1]. Result clipped to [0,1].
    """
    if similarity is None:
        similarity = 0.0
    if overlap is None:
        overlap = 0.0
    score = w_sim * similarity + w_overlap * overlap
    # clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score

def bucket_for_score(score: float) -> Literal["Safe", "Stretch", "Fallback"]:
    """Return bucket name for a fused score."""
    if score >= 0.80:
        return "Safe"
    if score >= 0.50:
        return "Stretch"
    return "Fallback"
