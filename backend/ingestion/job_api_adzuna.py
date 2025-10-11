"""
Optimized Adzuna API client with caching and request batching.

Features:
- Request caching with configurable TTL
- Parallel request execution
- Rate limiting with exponential backoff
- Request batching for multiple queries
- Automatic retry on failures
"""
import os
import time
import json
import logging
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from functools import lru_cache

import requests
from urllib.parse import urlencode

# Local imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.db import chroma as chroma_helper
from backend.ingestion import job_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("adzuna")
LOG.setLevel(logging.INFO)

# Constants
DEFAULT_MAX_REQ_PER_MIN = 30  # Increased default rate limit
CACHE_DIR = Path("data/cache")
CACHE_TTL = 3600 * 24  # 24 hours cache TTL

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class RateLimiter:
    """Thread-safe rate limiter with exponential backoff."""
    
    def __init__(self, max_requests: int = 30, per_seconds: float = 60.0):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = []
        self.lock = threading.Lock()
    
    def wait(self):
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [t for t in self.requests if now - t < self.per_seconds]
            
            if len(self.requests) >= self.max_requests:
                # Calculate sleep time
                oldest = self.requests[0]
                sleep_time = max(0, (oldest + self.per_seconds) - now)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(time.time())

# Global rate limiter
RATE_LIMITER = RateLimiter(max_requests=DEFAULT_MAX_REQ_PER_MIN)

def _throttle():
    """Ensure we respect min interval between requests."""
    RATE_LIMITER.wait()

def _get_cache_key(what: str, where: str, page: int, results_per_page: int) -> str:
    """Generate a unique cache key for a query."""
    key_str = f"{what}:{where}:{page}:{results_per_page}"
    return hashlib.md5(key_str.encode()).hexdigest()

def _get_cache_file_path(cache_key: str) -> Path:
    """Get the full path to a cache file."""
    return CACHE_DIR / f"{cache_key}.pkl"

def _load_from_cache(cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """Load data from cache if it exists and is fresh."""
    cache_file = _get_cache_file_path(cache_key)
    if not cache_file.exists():
        return None
        
    try:
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime > CACHE_TTL:
            return None
            
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        LOG.warning(f"Cache load failed: {e}")
        return None

def _save_to_cache(cache_key: str, data: List[Dict[str, Any]]) -> None:
    """Save data to cache."""
    try:
        cache_file = _get_cache_file_path(cache_key)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        LOG.warning(f"Cache save failed: {e}")

def fetch_jobs_adzuna(
    what: str,
    where: str,
    page: int = 1,
    results_per_page: int = 20,
    country: str = "in",
    max_retries: int = 3,
    timeout: int = 10,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch jobs from Adzuna API with caching, retries, and rate limiting.
    """
    # Check cache first
    cache_key = _get_cache_key(what, where, page, results_per_page)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            LOG.debug(f"Cache hit for {what} in {where} (page {page})")
            return cached
    
    # Apply rate limiting
    RATE_LIMITER.wait()
    
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    
    if not app_id or not app_key:
        LOG.error("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in environment")
        return []
    
    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page,
        "what": what,
        "where": where,
        "content-type": "application/json",
    }
    
    for attempt in range(max_retries):
        try:
            LOG.debug(f"Fetching Adzuna: {what} in {where} (page {page})")
            
            # Make the request
            resp = requests.get(base_url, params=params, timeout=timeout)
            
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("results", [])
                
                # Cache the results
                if jobs and use_cache:
                    _save_to_cache(cache_key, jobs)
                    
                return jobs
                
            elif resp.status_code == 429:  # Rate limited
                retry_after = int(resp.headers.get('Retry-After', min(30, 5 * (attempt + 1))))
                LOG.warning(f"Rate limited, retrying after {retry_after}s...")
                time.sleep(retry_after)
                continue
                
            elif 500 <= resp.status_code < 600:  # Server error
                backoff = min(30, (2 ** attempt))  # Cap backoff at 30s
                LOG.warning(f"Server error {resp.status_code}, retrying in {backoff}s...")
                time.sleep(backoff)
                continue
                
            else:
                LOG.error(f"Error {resp.status_code}: {resp.text[:200]}")
                break
                
        except requests.exceptions.RequestException as e:
            LOG.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                LOG.error(f"Max retries reached for {what} in {where}")
                raise
                
            backoff = min(30, (2 ** attempt))  # Cap backoff at 30s
            time.sleep(backoff)
    
    return []

def make_dedup_key(job: Dict[str, Any]) -> str:
    """
    Create a unique key for deduplication based on job title, company, and description.
    
    Args:
        job: Job dictionary containing title, company, and description
        
    Returns:
        SHA1 hash string that uniquely identifies similar job postings
    """
    import hashlib
    
    title = (job.get("title") or "").strip().lower()

    company_obj = job.get("company")
    if isinstance(company_obj, dict):
        company = (company_obj.get("display_name") or company_obj.get("name") or "").strip().lower()
    else:
        company = (str(company_obj) if company_obj else "").strip().lower()

    # Use first 100 chars of description for deduplication
    desc = (job.get("description") or job.get("description_snippet") or "")[:100].strip().lower()
    
    # Create a consistent string representation and hash it
    base = f"{title}|{company}|{desc}"
    return hashlib.sha1(base.encode()).hexdigest()


def fetch_jobs_for_queries(
    queries: List[Dict[str, str]],
    country: str = "gb",
    pages_per_query: int = 1,
    results_per_page: int = 20,
    max_workers: int = 4,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Run multiple queries in parallel and return deduplicated results.
    
    Deduplication is done based on a hash of (title + company + first 100 chars of description)
    to prevent duplicate job postings even if they have different IDs/URLs.
    """
    seen_hashes = set()
    all_jobs = []
    
    def process_query(query: Dict[str, str], page: int) -> List[Dict[str, Any]]:
        """Process a single query page."""
        try:
            return fetch_jobs_adzuna(
                what=query.get("what", ""),
                where=query.get("where", ""),
                page=page,
                results_per_page=results_per_page,
                country=country,
                use_cache=use_cache
            )
        except Exception as e:
            LOG.error(f"Error processing query {query} (page {page}): {e}")
            return []
    
    # Prepare all tasks
    tasks = []
    for query in queries:
        for page in range(1, pages_per_query + 1):
            tasks.append((query, page))
    
    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_query, query, page)
            for query, page in tasks
        ]
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                jobs = future.result()
                for job in jobs:
                    # Create deduplication key
                    dedup_key = make_dedup_key(job)
                    
                    # Only add if we haven't seen this job before
                    if dedup_key not in seen_hashes:
                        seen_hashes.add(dedup_key)
                        all_jobs.append(job)
                        
            except Exception as e:
                LOG.error(f"Error processing job result: {e}")
    
    LOG.info(f"Fetched {len(all_jobs)} unique jobs from {len(queries)} queries")
    return all_jobs


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
    country: str = "gb",  # Changed from "in" to "gb" for United Kingdom
    pages_per_query: int = 1,
    results_per_page: int = 20,
    ingest: bool = True,
    chroma_collection_name: Optional[str] = None,
):
    """
    High-level helper: fetch jobs for queries, normalize them and optionally ingest to Chroma.
    Jobs that have been seen before (based on a cache of IDs) are still processed and returned.
    """
    raw_jobs = fetch_jobs_for_queries(queries, country=country, pages_per_query=pages_per_query, results_per_page=results_per_page)

    # Load cache to update it with new IDs, but we won't filter based on it.
    cache = _load_cache()
    seen_ids = set(cache.get("seen_ids", []))

    normalized = []
    for r in raw_jobs:
        # We still normalize all jobs returned from the query.
        normalized.append(normalize_adzuna_job(r))
        
        # And we update the cache of seen IDs for future reference (if needed).
        nid = r.get("id") or r.get("redirect_url") or r.get("url") or ""
        if nid:
            seen_ids.add(nid)

    # Save updated cache
    cache["seen_ids"] = list(seen_ids)
    _save_cache(cache)

    LOG.info(f"Normalized {len(normalized)} jobs from query.")

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
