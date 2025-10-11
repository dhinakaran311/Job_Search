# app_streamlit/app.py
"""
Phase 7 — Job Recommender Streamlit App
- Offline ingestion from sample_jobs.json
- Optional Adzuna ingestion
- Resume upload/paste
- Skill extraction (LLM if key set, else lexicon)
- Ranking (TF-IDF + overlap)
- Grouped results (Safe / Stretch / Fallback)
- LLM job summary + upskilling plan (if key set)
- CSV export
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dotenv import load_dotenv
from scipy.sparse import issparse

# Optional parsers
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# Import backend services
try:
    from backend.services.skill_extractor import extract_skills as extract_skills_service
except Exception:
    def extract_skills_service(text: str, use_llm: Optional[bool] = None) -> List[str]:
        return [w for w in text.lower().split() if w in ["python", "docker", "aws", "sql", "pytorch"]]

try:
    from backend.services.llm import job_summary_1line, upskilling_plan_bulleted
except Exception:
    def job_summary_1line(text: str) -> str:
        return text[:100]
    def upskilling_plan_bulleted(role: str, skills: List[str], context: Optional[str] = None) -> str:
        return "- Week 1: Learn basics\n- Week 2: Build project"

from backend.ingestion.job_api_adzuna import fetch_and_ingest
from backend.db.chroma import get_or_create_collection

# Env keys
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")

# Session state
if "jobs" not in st.session_state:
    st.session_state["jobs"] = []
if "vectorizer" not in st.session_state:
    st.session_state["vectorizer"] = None
if "job_vectors" not in st.session_state:
    st.session_state["job_vectors"] = None
if "job_skills" not in st.session_state:
    st.session_state["job_skills"] = {}
if "log" not in st.session_state:
    st.session_state["log"] = []

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state["log"].append(f"[{ts}] {msg}")

# ---------------- Resume Parsing ----------------
def parse_resume(uploaded) -> str:
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf") and pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join([p.extract_text() or "" for p in pdf.pages])
        except:
            return ""
    if name.endswith(".docx") and docx:
        try:
            docf = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in docf.paragraphs])
        except:
            return ""
    try:
        return data.decode("utf-8")
    except:
        return ""

# ---------------- Job Ingestion ----------------
def load_offline_jobs(path="data/sample_jobs.json") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        log("sample_jobs.json not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    for i, j in enumerate(jobs):
        j.setdefault("id", f"job-{i}")
        j.setdefault("description", "")
    log(f"Loaded {len(jobs)} offline jobs")
    return jobs

def ingest_jobs(jobs: List[Dict[str, Any]]):
    if not jobs:
        log("No jobs to ingest")
        return
    log(f"Ingesting {len(jobs)} jobs...")
    
    # Initialize recommender if not already done
    if "recommender" not in st.session_state:
        from backend.services.retrieval import TFIDFRecommender
        st.session_state["recommender"] = TFIDFRecommender()
    
    # Fit the recommender with the new jobs
    st.session_state["recommender"].fit(jobs)
    st.session_state["jobs"] = jobs
    log(f"Ingested {len(jobs)} jobs")

# ---------------- Recommendation ----------------
# Initialize TF-IDF recommender in session state
if "recommender" not in st.session_state:
    from backend.services.retrieval import TFIDFRecommender, bucketize_recommendations
    st.session_state["recommender"] = TFIDFRecommender()

def recommend(resume_text: str, role: str, top_k: int = 10, use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Get job recommendations using the TF-IDF recommender.
    
    Args:
        resume_text: The resume text to match against jobs
        role: Target role (not currently used)
        top_k: Number of results to return
        use_llm: Whether to use LLM for skill extraction
        
    Returns:
        List of job recommendations with scores and metadata
    """
    if not hasattr(st.session_state, "recommender") or not st.session_state.recommender.jobs:
        log("No jobs ingested - please load jobs first")
        return []
        
    return st.session_state.recommender.recommend(
        resume_text,
        top_k=top_k,
        use_llm=use_llm and bool(OPENAI_KEY)
    )

def bucketize(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize recommendations into buckets based on score thresholds.
    
    Args:
        results: List of recommendation results with 'score' field
        
    Returns:
        Dictionary with keys 'Safe', 'Stretch', 'Fallback' containing matching results
    """
    from backend.services.retrieval import bucketize_recommendations
    return bucketize_recommendations(results)

# Country code mapping
COUNTRY_CODES = {
    "United Kingdom": "gb",
    "United States": "us",
    "Canada": "ca",
    "Australia": "au",
    "India": "in",
    "Germany": "de",
    "France": "fr",
    "Brazil": "br",
    "Japan": "jp",
    "Singapore": "sg"
}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("Job Search & Recommendation System")

with st.sidebar:
    st.header("Job Search")
    search_query = st.text_input("Job title, keywords, or company")
    location = st.text_input("Location")
    
    # Add country selection
    selected_country = st.selectbox(
        "Select Country",
        options=list(COUNTRY_CODES.keys()),
        index=0,  # Default to first country (UK)
        help="Select the country for job search"
    )
    
    search_button = st.button("Search Jobs")
    
    st.markdown("---")
    st.header("Controls & Ingest")
    st.write("Ollama LLM: enabled (local)")
    st.button("Ingest offline jobs", on_click=lambda: ingest_jobs(load_offline_jobs()))
    st.markdown("---")
    st.subheader("Adzuna Real-Time Jobs")
    adz_what = st.text_input("Search term (what)", "python")
    adz_where = st.text_input("Location (where)", "india")
    adz_country = st.text_input("Country code", "in")
    adz_pages = st.number_input("Pages per query", 1, 5, 1)
    adz_results = st.number_input("Results per page", 1, 50, 20)
    if st.button("Fetch Adzuna Jobs"):
        queries = [{"what": adz_what, "where": adz_where}]
        jobs = fetch_and_ingest(queries, country=adz_country, pages_per_query=adz_pages, results_per_page=adz_results, ingest=False)
        ingest_jobs(jobs)
        log(f"Fetched and ingested {len(jobs)} jobs from Adzuna")
    st.markdown("---")
    top_k = st.number_input("top-K", 1, 50, 10)
    use_llm = st.checkbox("Use LLM for skills/summaries (Ollama)", value=True)
    st.text_area("Log", "\n".join(st.session_state["log"][-20:]), height=200)

# Handle job search
if search_button and (search_query or location):
    with st.spinner("Searching for jobs..."):
        try:
            # Get the country code from the selected country name
            country_code = COUNTRY_CODES.get(selected_country, "gb")  # Default to GB if not found
            
            # Convert the search to queries format expected by fetch_jobs_for_queries
            queries = [{"what": search_query, "where": location}]
            jobs = fetch_and_ingest(
                queries, 
                country=country_code,  # Pass the selected country code
                pages_per_query=1
            )
            
            if jobs:
                st.success(f"Found {len(jobs)} jobs in {selected_country}")
                # Display jobs in a nice format
                for i, job in enumerate(jobs[:10]):  # Show first 10 jobs
                    with st.expander(f"{job.get('title', 'No title')} at {job.get('company', 'Unknown company')}"):
                        st.write(f"**Location:** {job.get('location', 'Not specified')}")
                        st.write(f"**Posted:** {job.get('created', 'Date not available')}")
                        st.write(f"**Description:** {job.get('description', 'No description available')[:300]}...")
                        if job.get('redirect_url'):
                            st.markdown(f"[View Job]({job['redirect_url']})")
            else:
                st.warning(f"No jobs found in {selected_country}. Try different search terms or location.")
                
        except Exception as e:
            st.error(f"Error searching for jobs in {selected_country}: {str(e)}")
            st.exception(e)

# Main layout
resume_file = st.file_uploader("Upload resume (pdf/docx/txt/md)")
resume_text = st.text_area("Or paste resume text")
if resume_file and not resume_text:
    resume_text = parse_resume(resume_file)

role = st.text_input("Target role", "Software Developer")
if st.button("Recommend Jobs"):
    # Extract skills from resume using Ollama
    skills = extract_skills_service(resume_text, use_llm=True)
    log(f"Extracted skills from resume: {skills}")
    # Limit to top 5 skills
    top_skills = skills[:5]
    log(f"Using top skills for Adzuna queries: {top_skills}")
    # Use skills as search terms for Adzuna
    queries = [{"what": skill, "where": adz_where} for skill in top_skills]
    # Always use 10 results per skill for optimization
    with st.spinner("Fetching jobs from Adzuna and computing recommendations..."):
        jobs = fetch_and_ingest(queries, country=adz_country, pages_per_query=adz_pages, results_per_page=10, ingest=False)
        ingest_jobs(jobs)
        log(f"Fetched and ingested {len(jobs)} jobs from Adzuna using resume skills")
        recs = recommend(resume_text, role, top_k=top_k, use_llm=use_llm)
        st.session_state["recs"] = recs
        log(f"Computed {len(recs)} recs")

recs = st.session_state.get("recs", [])
if recs:
    buckets = bucketize(recs)
    tabs = st.tabs(["Safe", "Stretch", "Fallback"])
    for name, tab in zip(["Safe", "Stretch", "Fallback"], tabs):
        with tab:
            for r in buckets[name]:
                j = r["job"]
                with st.expander(f"{j['title']} @ {j['company']} (score {r['score']:.2f})"):
                    st.write(f"**Location:** {j.get('location','')}")
                    st.write(f"**Posted:** {j.get('posted_at','')}")
                    st.write(f"**URL:** {j.get('url','')}")
                    st.write(f"**Similarity:** {r['similarity']:.3f}")
                    st.write(f"**Job skills:** {', '.join(r['job_skills'])}")
                    st.write(f"**Resume skills:** {', '.join(r['resume_skills'])}")
                    st.write(f"**Missing skills:** {', '.join(r['missing_skills']) or 'None'}")
                    if use_llm:
                        st.write("**LLM summary (Ollama):**", job_summary_1line(j["description"]))
                        st.write("**Upskilling plan (Ollama):**")
                        st.write(upskilling_plan_bulleted(role, r["missing_skills"], context="Candidate resume"))
                    else:
                        st.write("LLM disabled")

    # Export CSV
    rows = []
    for r in recs:
        j = r["job"]
        rows.append({
            "id": j["id"], "title": j["title"], "company": j["company"],
            "location": j.get("location"), "url": j.get("url"),
            "score": r["score"], "job_skills": ";".join(r["job_skills"]),
            "resume_skills": ";".join(r["resume_skills"]),
            "missing": ";".join(r["missing_skills"])
        })
    df = pd.DataFrame(rows)
    st.download_button("Export CSV", df.to_csv(index=False), "recs.csv", "text/csv")

# Adzuna ingestion example (uncomment to use)
# queries = [{"what": "python", "where": "india"}]  # or get from user input
# jobs = fetch_and_ingest(queries, ingest=False)
# st.write(jobs)

st.caption("First run? Ingest jobs on the sidebar, then upload/paste resume and click 'Recommend Jobs'.")
