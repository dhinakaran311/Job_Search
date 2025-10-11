import os
import sys
import json
import logging
import requests
from pprint import pprint
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("adzuna_test")

# Enable requests logging
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1
requests_log = logging.getLogger("urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

# Load environment variables
load_dotenv()

# Get API credentials
ADZUNA_APP_ID = os.getenv('ADZUNA_APP_ID')
ADZUNA_APP_KEY = os.getenv('ADZUNA_APP_KEY')

if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
    logger.error("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in .env file")
    sys.exit(1)

def test_adzuna_api():
    """Test Adzuna API directly with various endpoints and parameters."""
    base_url = "https://api.adzuna.com/v1/api"
    
    # Test 1: Basic job search
    test_endpoint = f"{base_url}/jobs/gb/search/1"
    params = {
        'app_id': ADZUNA_APP_ID,
        'app_key': ADZUNA_APP_KEY,
        'what': 'python',
        'results_per_page': 2,
        'content-type': 'application/json'
    }
    
    logger.info("\n=== Testing Basic Job Search ===")
    response = make_api_call(test_endpoint, params)
    
    if response and 'results' in response:
        logger.info(f"Found {len(response['results'])} jobs")
        if response['results']:
            logger.info("\nSample job:")
            pprint(response['results'][0])
    
    # Test 2: Get job by ID (using an ID from the first test if available)
    if response and 'results' in response and response['results']:
        job_id = response['results'][0]['id']
        test_endpoint = f"{base_url}/jobs/gb/{job_id}"
        logger.info(f"\n=== Testing Job Details for ID: {job_id} ===")
        job_response = make_api_call(test_endpoint, {
            'app_id': ADZUNA_APP_ID,
            'app_key': ADZUNA_APP_KEY
        })
        
        if job_response:
            logger.info("Job details:")
            pprint(job_response)

def make_api_call(url: str, params: dict):
    """Make an API call with error handling and logging."""
    try:
        logger.debug(f"Making request to: {url}")
        logger.debug(f"Params: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        logger.debug(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API Error ({response.status_code}): {response.text}")
            return None
            
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.debug(f"Response content: {response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

if __name__ == "__main__":
    test_adzuna_api()
