import os
import sys
import logging
import requests
import json
from pprint import pprint
from dotenv import load_dotenv
from backend.ingestion.job_api_adzuna import fetch_jobs_adzuna, fetch_jobs_for_queries

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Check if API keys are set
ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID')
ADZUNA_APP_KEY = os.getenv('ADZUNA_APP_KEY')

# Enable requests logging for debugging
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1

# Configure requests logging
requests_log = logging.getLogger("urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

print("Environment Variables:")
print(f"ADZUNA_APP_ID: {'Set' if ADZUNA_APP_ID else 'Not Set'}")
print(f"ADZUNA_APP_KEY: {'Set' if ADZUNA_APP_KEY else 'Not Set'}")

def test_direct_api_call():
    """Make a direct API call to verify credentials and get sample data."""
    print("\nTesting direct API call...")
    test_url = f"https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {
        'app_id': ADZUNA_APP_ID,
        'app_key': ADZUNA_APP_KEY,
        'what': 'python',
        'results_per_page': 5
    }
    
    try:
        print(f"Making request to: {test_url}")
        print(f"With params: {params}")
        response = requests.get(test_url, params=params, timeout=10)
        
        print(f"\nResponse Status: {response.status_code}")
        print("Response Headers:")
        pprint(dict(response.headers))
        
        if response.status_code == 200:
            data = response.json()
            print("\nAPI Response Sample:")
            if 'results' in data and len(data['results']) > 0:
                print(f"Found {len(data['results'])} jobs")
                print("\nSample Job:")
                pprint(data['results'][0])
            else:
                print("No results found in response")
                print("Full response:")
                pprint(data)
            return True
        else:
            print(f"\nAPI Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\nError during API call: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Run the test
print("Environment Variables:")
print(f"ADZUNA_APP_ID: {'Set' if ADZUNA_APP_ID else 'Not Set'}")
print(f"ADZUNA_APP_KEY: {'Set' if ADZUNA_APP_KEY else 'Not Set'}")

if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
    print("\nError: Missing Adzuna API credentials. Please check your .env file.")
    sys.exit(1)

if not test_direct_api_call():
    print("\nAPI test failed. Please check your credentials and try again.")
    sys.exit(1)

print("\nProceeding with job search using library...\n")

def test_library_call():
    """Test the fetch_jobs_for_queries function."""
    print("\nTesting fetch_jobs_for_queries...")
    
    # Test with a single, simple query first
    test_queries = [{"what": "python", "where": "london"}]
    
    try:
        print(f"\nCalling fetch_jobs_for_queries with: {test_queries}")
        jobs = fetch_jobs_for_queries(
            test_queries,
            pages_per_query=1,
            use_cache=False,
            max_workers=1
        )
        
        print(f"\nFetched {len(jobs)} jobs")
        if jobs:
            print("\nFirst job details:")
            pprint(jobs[0])
        
        # Now test with direct fetch_jobs_adzuna call
        print("\nTesting direct fetch_jobs_adzuna call...")
        direct_jobs = fetch_jobs_adzuna(
            what="python",
            where="london",
            page=1,
            results_per_page=5,
            country="gb"
        )
        
        print(f"\nDirect fetch_jobs_adzuna returned {len(direct_jobs)} jobs")
        if direct_jobs:
            print("\nFirst job from direct call:")
            pprint(direct_jobs[0])
        
        return jobs
        
    except Exception as e:
        print(f"\nError in library call: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Run the tests
if __name__ == "__main__":
    try:
        test_library_call()
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()