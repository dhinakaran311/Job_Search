import os
import logging
from pprint import pprint
from dotenv import load_dotenv
from backend.ingestion.job_api_adzuna import fetch_jobs_for_queries, fetch_jobs_adzuna

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_updated")

# Load environment variables
load_dotenv()

def test_updated_functions():
    # Test 1: Test fetch_jobs_adzuna directly
    print("\n=== Testing fetch_jobs_adzuna ===")
    jobs = fetch_jobs_adzuna(
        what="python",
        where="london",
        country="gb",
        results_per_page=2,
        use_cache=False
    )
    print(f"Fetched {len(jobs)} jobs")
    if jobs:
        print("First job:")
        pprint(jobs[0])
    
    # Test 2: Test fetch_jobs_for_queries with default country (should be 'gb' now)
    print("\n=== Testing fetch_jobs_for_queries ===")
    queries = [
        {"what": "python", "where": "london"},
        {"what": "data scientist", "where": "uk"}
    ]
    
    all_jobs = fetch_jobs_for_queries(
        queries=queries,
        pages_per_query=1,
        results_per_page=2,
        use_cache=False
    )
    
    print(f"\nFetched {len(all_jobs)} total jobs from {len(queries)} queries")
    if all_jobs:
        print("\nSample job from batch:")
        pprint(all_jobs[0])

if __name__ == "__main__":
    test_updated_functions()
