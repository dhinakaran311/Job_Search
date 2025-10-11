"""
job_loader.py

Load jobs from a local JSON file (data/sample_jobs.json) and ingest into Chroma.

Usage (from project root):
    python backend/ingestion/job_loader.py data/sample_jobs.json
or (default path):
    python backend/ingestion/job_loader.py
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional

from backend.db import chroma as chroma_helper
from backend.services import embedding as emb_service

# Small skills lexicon for fallback skill extraction
SKILLS_LEXICON = [
    "python","sql","streamlit","pytorch","tensorflow","keras","scikit-learn",
    "pandas","numpy","docker","kubernetes","aws","azure","gcp","react","node",
    "java","javascript","c++","git","linux","spark","hadoop","nlp","cv","flask",
    "fastapi","rest","graphql"
]

def _make_id(raw: Dict[str, Any]) -> str:
    """Deterministic id: prefer 'id' or url, else sha1(title+company)."""
    if "id" in raw and raw["id"]:
        return str(raw["id"])
    if "job_id" in raw and raw["job_id"]:
        return str(raw["job_id"])
    if raw.get("url"):
        return hashlib.sha1(raw.get("url", "").encode("utf-8")).hexdigest()
    
    # Handle company field which might be a dictionary
    company = raw.get("company", "")
    if isinstance(company, dict):
        company = company.get("display_name") or company.get("name") or ""
    
    # fallback to title+company
    key = f"{raw.get('title', '')}|{company}".strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _extract_skills_from_text(text: str) -> List[str]:
    text_lower = (text or "").lower()
    found = set()
    for skill in SKILLS_LEXICON:
        if skill in text_lower:
            found.add(skill)
    return sorted(found)

def normalize_job(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw JSON job object to canonical schema."""
    # Handle case where company might be nested object
    company = ""
    if isinstance(raw.get("company"), dict):
        company = raw["company"].get("name") or raw["company"].get("company_name") or ""
    else:
        company = raw.get("company") or raw.get("company_name") or ""

    location = ""
    if isinstance(raw.get("location"), dict):
        # common Adzuna style
        location = raw["location"].get("display_name") or raw["location"].get("city") or ""
    else:
        location = raw.get("location") or raw.get("area") or raw.get("city") or ""

    url = raw.get("redirect_url") or raw.get("url") or raw.get("link") or raw.get("jobUrl") or ""

    description = raw.get("description") or raw.get("job_description") or raw.get("details") or raw.get("summary") or ""

    skills = raw.get("skills")
    if not skills or not isinstance(skills, list):
        skills = _extract_skills_from_text(description)

    normalized = {
        "id": _make_id(raw),
        "title": raw.get("title") or raw.get("position") or raw.get("jobTitle") or "",
        "company": company,
        "location": location,
        "url": url,
        "posted_at": raw.get("created") or raw.get("posted_at") or raw.get("date") or raw.get("posted") or None,
        "description": description,
        "skills": skills
    }
    return normalized

def ingest_jobs(collection, jobs: List[Dict[str, Any]], delete_existing: bool = True, batch_size: int = 64):
    """
    Ingest normalized jobs into a Chroma collection.
    - collection: chroma collection object
    - jobs: list of raw job dicts from JSON
    - delete_existing: if True, delete any existing docs with same ids before adding
    """
    # Normalize jobs
    normalized_jobs = [normalize_job(j) for j in jobs]

    ids = [j["id"] for j in normalized_jobs]
    documents = [j["description"] or "" for j in normalized_jobs]
    metadatas = [
    {
        "title": j["title"],
        "company": j["company"],
        "location": j["location"],
        "url": j["url"],
        "posted_at": j["posted_at"],
        # Chroma only allows primitive values, so join list into a string
        "skills": ", ".join(j["skills"]) if isinstance(j["skills"], list) else str(j["skills"] or "")
    }
    for j in normalized_jobs
]


    # Optional: delete existing items with the same ids for idempotency
    if delete_existing:
        try:
            # Some chroma versions support collection.delete(ids=[...])
            collection.delete(ids=ids)
        except Exception:
            # If delete fails (old/new APIs), ignore and continue (add may update or raise)
            pass

    # Compute embeddings in batches to avoid memory issues
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_emb = emb_service.embed_texts(batch)
        all_embeddings.extend(batch_emb)

    # Add to collection
    # Use try/except to support slight API differences across chroma versions
    try:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=all_embeddings)
    except TypeError:
        # some versions expect different arg order/keywords
        collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=all_embeddings)
    except Exception as e:
        # raise or print helpful error
        raise RuntimeError(f"Failed to add documents to Chroma: {e}")

    print(f"Ingested {len(ids)} jobs into collection '{getattr(collection, 'name', 'unknown')}'")

def ingest_jobs_from_file(collection, json_path: str):
    """Load JSON file and ingest."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # support top-level dict with 'jobs' list or a plain list
    if isinstance(data, dict) and ("jobs" in data and isinstance(data["jobs"], list)):
        jobs = data["jobs"]
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError("Unsupported JSON structure. Expected list or {'jobs':[...]}")

    ingest_jobs(collection, jobs)

# CLI
if __name__ == "__main__":
    import sys

    # Default JSON path
    default_path = os.path.join("data", "sample_jobs.json")
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        sys.exit(1)

    # init chroma client and collection
    client = chroma_helper.init_chroma_client()
    collection = chroma_helper.get_or_create_collection(client=client)

    print(f"Loading jobs from: {json_path}")
    ingest_jobs_from_file(collection, json_path)

    # Quick check: print collection count if available
    try:
        cnt = collection.count()
        print("Collection count:", cnt)
    except Exception:
        # Some versions may not have count()
        print("Ingestion complete (collection.count() not available in this chroma version).")
