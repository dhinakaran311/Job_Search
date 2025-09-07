"""
Retrieval service: embed resume text, query Chroma, compute similarity/overlap,
and return ranked results with buckets.

Provides:
- recommend_jobs(resume_text, top_k=10, collection=None, collection_name=None, lexicon=None)
- small CLI for quick manual tests when run as module.
"""

from typing import List, Dict, Any, Optional, Set
import re
import os

from backend.services import embedding as emb_service
from backend.db import chroma as chroma_helper
from backend.services.ranking import fused_score, bucket_for_score

# small fallback lexicon (keeps consistent with previous modules)
DEFAULT_LEXICON = [
    "python","sql","streamlit","pytorch","tensorflow","keras","scikit-learn",
    "pandas","numpy","docker","kubernetes","aws","azure","gcp","react","node",
    "java","javascript","c++","git","linux","spark","hadoop","nlp","cv","flask",
    "fastapi","rest","graphql"
]


def _safe_unwrap(resp: Any, key: str) -> List:
    """
    Extract key results from Chroma query response.
    Handles both new dict (values are lists-of-lists) and older list-of-dict formats.
    """
    if isinstance(resp, dict) and key in resp:
        val = resp[key]
        # typical format: val = [[...]] for single query -> return inner list
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            return val[0]
        return val
    if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], dict):
        return resp[0].get(key, [])
    return []


def _parse_skills_from_metadata(metadata: Dict[str, Any], description: str = "", lexicon: Optional[List[str]] = None) -> Set[str]:
    """Return a set of normalized skills from metadata['skills'] or by scanning description."""
    lex = set([s.lower() for s in (lexicon or DEFAULT_LEXICON)])
    skills_raw = metadata.get("skills") if metadata else None

    skills_set = set()

    # Accept lists or comma-separated strings
    if isinstance(skills_raw, list):
        for s in skills_raw:
            if not s:
                continue
            skills_set.add(str(s).strip().lower())
    elif isinstance(skills_raw, str) and skills_raw.strip():
        # split on comma or semicolon
        parts = re.split(r"[;,]\s*|\s+\|\s+|\s+/\s+", skills_raw)
        for p in parts:
            p_s = p.strip().lower()
            if p_s:
                skills_set.add(p_s)

    # Fallback: pattern-based detection from description if no skills found
    if not skills_set and description:
        txt = description.lower()
        for term in lex:
            if re.search(r"\b" + re.escape(term) + r"\b", txt):
                skills_set.add(term)

    return skills_set


def _distance_to_similarity(distance: Optional[float]) -> Optional[float]:
    """
    Convert a chroma distance to similarity in [0,1].
    We use similarity = 1 - distance and clamp.
    If distance is None or not numeric, return None.
    """
    if distance is None:
        return None
    try:
        sim = 1.0 - float(distance)
    except Exception:
        return None
    # clamp
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim


def recommend_jobs(
    resume_text: str,
    top_k: int = 10,
    collection=None,
    collection_name: Optional[str] = None,
    lexicon: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Main function that takes resume text and returns top_k ranked job matches.

    Returns a list of dicts with fields:
    { id, title, company, url, similarity, skill_overlap, score, bucket, missing_skills, metadata, document }
    """

    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text must be a non-empty string")

    # Ensure we have a collection
    if collection is None:
        client = chroma_helper.init_chroma_client()
        collection = chroma_helper.get_or_create_collection(collection_name=collection_name, client=client)

    # 1) embed resume
    q_vec = emb_service.embed_text(resume_text)

    # 2) query chroma
    try:
        resp = collection.query(query_embeddings=[q_vec], n_results=top_k, include=["metadatas", "documents", "distances"])
    except TypeError:
        # older versions may not support include list exactly; try without include
        resp = collection.query(query_embeddings=[q_vec], n_results=top_k)

    # 3) unwrap response
    ids = _safe_unwrap(resp, "ids")
    metadatas = _safe_unwrap(resp, "metadatas")
    docs = _safe_unwrap(resp, "documents")
    distances = _safe_unwrap(resp, "distances")

    # 4) build user skills via lexicon extraction (simple)
    resume_txt = resume_text.lower()
    lex = set([s.lower() for s in (lexicon or DEFAULT_LEXICON)])
    user_skills = set([term for term in lex if re.search(r"\b" + re.escape(term) + r"\b", resume_txt)])

    results = []
    n = max(len(ids), len(metadatas), len(docs), len(distances))
    for i in range(n):
        job_id = ids[i] if i < len(ids) else None
        metadata = metadatas[i] if i < len(metadatas) else {}
        document = docs[i] if i < len(docs) else ""
        distance = distances[i] if i < len(distances) else None

        # Parse job skills
        job_skills = _parse_skills_from_metadata(metadata or {}, document or "", lexicon=lexicon)
        # Compute overlap
        overlap = 0.0
        if job_skills:
            overlap = len(job_skills.intersection(user_skills)) / max(1, len(job_skills))
        else:
            overlap = 0.0

        similarity = _distance_to_similarity(distance)
        if similarity is None:
            # As a fallback, set similarity to 0
            similarity = 0.0

        score = fused_score(similarity, overlap)
        bucket = bucket_for_score(score)
        missing_skills = sorted(list(job_skills.difference(user_skills)))

        result = {
            "id": job_id,
            "title": (metadata or {}).get("title") if metadata else None,
            "company": (metadata or {}).get("company") if metadata else None,
            "url": (metadata or {}).get("url") if metadata else None,
            "similarity": similarity,
            "skill_overlap": overlap,
            "score": score,
            "bucket": bucket,
            "missing_skills": missing_skills,
            "metadata": metadata,
            "document": document
        }
        results.append(result)

    # sort by score desc
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


# Simple CLI for manual testing
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Recommend jobs from resume text using Chroma.")
    parser.add_argument("--resume_text", type=str, default=None, help="Resume text (pass directly)")
    parser.add_argument("--resume_file", type=str, default=None, help="Path to resume text file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name (optional)")
    args = parser.parse_args()

    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print("Resume file not found:", args.resume_file)
            raise SystemExit(1)

        # If PDF or DOCX, use resume_parser
        ext = os.path.splitext(args.resume_file)[1].lower()
        if ext in [".pdf", ".docx", ".doc"]:
            from backend.ingestion import resume_parser
            resume = resume_parser.parse_resume_file(args.resume_file)
        else:
            # plain text
            resume = open(args.resume_file, "r", encoding="utf-8").read()



    client = chroma_helper.init_chroma_client()
    collection = chroma_helper.get_or_create_collection(collection_name=args.collection, client=client)
    recs = recommend_jobs(resume, top_k=args.top_k, collection=collection)
    print(json.dumps(recs[:args.top_k], indent=2, ensure_ascii=False))
