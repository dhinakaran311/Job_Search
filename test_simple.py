import os
from pprint import pprint
from dotenv import load_dotenv
from backend.ingestion.job_api_adzuna import fetch_jobs_adzuna

# Load environment variables
load_dotenv()

# Test with direct API call
print("Testing direct API call...")
test_url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
params = {
    'app_id': os.getenv('ADZUNA_APP_ID'),
    'app_key': os.getenv('ADZUNA_APP_KEY'),
    'what': 'python',
    'results_per_page': 5
}

print(f"Making request to: {test_url}")
print(f"With params: {params}")

import requests
response = requests.get(test_url, params=params, timeout=10)
print(f"Status code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"Found {len(data.get('results', []))} jobs")
    if data.get('results'):
        print("\nSample job:")
        pprint(data['results'][0])

# Now test the fetch_jobs_adzuna function
print("\nTesting fetch_jobs_adzuna function...")
try:
    jobs = fetch_jobs_adzuna(
        what="python",
        where="london",
        page=1,
        results_per_page=5,
        country="gb"
    )
    print(f"\nFetched {len(jobs)} jobs")
    if jobs:
        print("\nFirst job:")
        pprint(jobs[0])
except Exception as e:
    print(f"Error in fetch_jobs_adzuna: {str(e)}")
    import traceback
    traceback.print_exc()
