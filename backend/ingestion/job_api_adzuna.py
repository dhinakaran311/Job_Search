"""
Adzuna adapter + CLI.

Provides:
- fetch_jobs_adzuna(what, where, page=1, results_per_page=20, country='in')
- fetch_jobs_for_queries(queries, country='in', pages_per_query=1)
- CLI entrypoint to fetch and optionally ingest into Chroma using backend.ingestion.job_loader.ingest_jobs

Notes:
- Reads ADZUNA_APP_ID and ADZUNA_APP_KEY from env (or .env via python-dotenv).
- Implements simple throttling and exponential backoff for 429/5xx.
- Deduplicates jobs by adzuna 'id' or URL.
"""
import os
import time
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import urlencode

# optional .env loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# local helpers
from backend.db import chroma as chroma_helper
from backend.ingestion import job_loader

LOG = logging.getLogger("adzuna")
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.StreamHandler())

# Rate limit throttle: max_requests_per_minute (default conservative)
DEFAULT_MAX_REQ_PER_MIN = 20
_MIN_INTERVAL = 60.0 / DEFAULT_MAX_REQ_PER_MIN  # seconds between requests

# Module-level last request time for simple throttling
_last_request_time = 0.0

def _throttle():
    """Ensure we respect min interval between requests."""
    global _last_request_time
    now = time.perf_counter()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        to_sleep = _MIN_INTERVAL - elapsed
        LOG.debug(f"Throttling: sleeping {to_sleep:.2f}s to respect rate limits")
        time.sleep(to_sleep)
    _last_request_time = time.perf_counter()


def fetch_jobs_adzuna(
    what: str,
    where: str,
    page: int = 1,
    results_per_page: int = 20,
    country: str = "in",
    max_retries: int = 5,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetch jobs from Adzuna.

    Returns list of raw job dicts (Adzuna's 'results' elements).
    """
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        raise EnvironmentError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in environment or .env")

    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    # Build params WITHOUT logging secrets
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": what,
        "where": where,
        "results_per_page": results_per_page,
        "content-type": "application/json"
    }

    # For logging, show a sanitized URL (no keys)
    log_params = {k: v for k, v in params.items() if k not in ("app_id", "app_key")}
    LOG.info(f"Fetching Adzuna: {base_url}?{urlencode(log_params)}")

    attempt = 0
    while attempt < max_retries:
        _throttle()
        try:
            resp = requests.get(base_url, params=params, timeout=timeout)
        except requests.RequestException as e:
            LOG.warning(f"Network error fetching Adzuna (attempt {attempt+1}): {e}")
            backoff = 2 ** attempt
            time.sleep(backoff)
            attempt += 1
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                LOG.error("Failed to parse Adzuna JSON response")
                return []
            results = data.get("results", [])
            LOG.info(f"Adzuna returned {len(results)} results")
            return results

        if resp.status_code == 429:
            # Too many requests -> exponential backoff
            backoff = 2 ** attempt
            LOG.warning(f"Adzuna 429 rate limit. Backing off {backoff}s (attempt {attempt+1})")
            time.sleep(backoff)
            attempt += 1
            continue

        if 500 <= resp.status_code < 600:
            backoff = 2 ** attempt
            LOG.warning(f"Adzuna server error {resp.status_code}. Backing off {backoff}s (attempt {attempt+1})")
            time.sleep(backoff)
            attempt += 1
            continue

        # other client errors
        LOG.error(f"Adzuna returned {resp.status_code}: {resp.text[:200]}")
        return []

    LOG.error("Exceeded max retries fetching Adzuna")
    return []


def fetch_jobs_for_queries(
    queries: List[Dict[str, str]],
    country: str = "in",
    pages_per_query: int = 1,
    results_per_page: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run a list of queries (each {'what':..., 'where':...}) and return a deduplicated list of jobs.
    """
    seen = set()
    out = []

    for q in queries:
        what = q.get("what", "")
        where = q.get("where", "")
        for page in range(1, pages_per_query + 1):
            results = fetch_jobs_adzuna(what=what, where=where, page=page, results_per_page=results_per_page, country=country)
            for r in results:
                # Adzuna's id or redirect_url used for de-duplication
                rid = r.get("id") or r.get("redirect_url") or r.get("url")
                if not rid:
                    # fallback to hashing title+company+location
                    rid = f"{r.get('title')}_{r.get('company')}_{r.get('location')}"
                if rid in seen:
                    continue
                seen.add(rid)
                out.append(r)
    LOG.info(f"Total unique Adzuna jobs fetched: {len(out)}")
    return out


def normalize_adzuna_job(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map Adzuna raw job -> canonical job schema used by our ingestion:
    { id, title, company, location, url, posted_at, description, skills: [] }
    """
    # Adzuna naming: title, company, location, redirect_url, created, description
    comp = raw.get("company")
    company_name = ""
    if isinstance(comp, dict):
        company_name = comp.get("display_name") or comp.get("name") or ""
    else:
        company_name = comp or ""

    loc = raw.get("location")
    location_name = ""
    if isinstance(loc, dict):
        location_name = loc.get("display_name") or loc.get("area") or ""
    else:
        location_name = loc or ""

    job = {
        "id": raw.get("id") or raw.get("redirect_url") or raw.get("url") or "",
        "title": raw.get("title") or "",
        "company": company_name,
        "location": location_name,
        "url": raw.get("redirect_url") or raw.get("url") or "",
        "posted_at": raw.get("created") or raw.get("created_date") or None,
        "description": raw.get("description") or raw.get("description_snippet") or "",
        # skills left empty for lightweight post-processing (job_loader will try to extract)
        "skills": []
    }
    return job


# Optional: file-based cache for fetched Adzuna IDs to avoid re-fetching across runs
_CACHE_PATH = os.path.join("data", "adzuna_cache.json")


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]):
    os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def fetch_and_ingest(
    queries: List[Dict[str, str]],
    country: str = "in",
    pages_per_query: int = 1,
    results_per_page: int = 20,
    ingest: bool = True,
    chroma_collection_name: Optional[str] = None,
):
    """
    High-level helper: fetch jobs for queries, normalize them and optionally ingest to Chroma.
    """
    raw_jobs = fetch_jobs_for_queries(queries, country=country, pages_per_query=pages_per_query, results_per_page=results_per_page)

    # Load cache and skip previously ingested IDs if present
    cache = _load_cache()
    seen_ids = set(cache.get("seen_ids", []))

    normalized = []
    for r in raw_jobs:
        nid = r.get("id") or r.get("redirect_url") or r.get("url") or ""
        if nid in seen_ids:
            continue
        normalized.append(normalize_adzuna_job(r))
        if nid:
            seen_ids.add(nid)

    # Save updated cache
    cache["seen_ids"] = list(seen_ids)
    _save_cache(cache)

    LOG.info(f"Normalized {len(normalized)} new jobs (after cache dedup).")

    if ingest and normalized:
        client = chroma_helper.init_chroma_client()
        collection = chroma_helper.get_or_create_collection(
        collection_name=chroma_collection_name or os.environ.get("COLLECTION_NAME", "jobs"),
        client=client
    )

        # Reuse job_loader's ingest_jobs which will normalize metadata and compute embeddings
        job_loader.ingest_jobs(collection, normalized, delete_existing=False)
        LOG.info("Ingested fetched Adzuna jobs into Chroma.")

    return normalized


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch jobs from Adzuna and optionally ingest into Chroma.")
    parser.add_argument("--what", type=str, default="python", help="Search term (what)")
    parser.add_argument("--where", type=str, default="india", help="Location (where)")
    parser.add_argument("--country", type=str, default="in", help="Adzuna country code (e.g., in, gb, us)")
    parser.add_argument("--pages", type=int, default=1, help="Pages per query")
    parser.add_argument("--results", type=int, default=20, help="Results per page")
    parser.add_argument("--ingest", action="store_true", help="Ingest fetched jobs into Chroma")
    parser.add_argument("--queries_file", type=str, default=None, help="Optional JSON file with list of {what,where} queries")
    args = parser.parse_args()

    queries = [{"what": args.what, "where": args.where}]
    if args.queries_file:
        if os.path.exists(args.queries_file):
            with open(args.queries_file, "r", encoding="utf-8") as f:
                qlist = json.load(f)
                if isinstance(qlist, list):
                    queries = qlist

    LOG.info(f"Running Adzuna fetch for queries: {queries}")
    new_jobs = fetch_and_ingest(queries, country=args.country, pages_per_query=args.pages, results_per_page=args.results, ingest=args.ingest)
    LOG.info(f"Done. Fetched {len(new_jobs)} normalized jobs.")
    if new_jobs:
        print(json.dumps(new_jobs, indent=2))
