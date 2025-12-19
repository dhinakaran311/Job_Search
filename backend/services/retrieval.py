"""
Retrieval service: TF-IDF + Skill Match + Experience Match + (optional Chroma)
"""

from typing import List, Dict, Any, Optional, Set
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Chroma DB support (existing functionality)
from backend.services import embedding as emb_service
from backend.db import chroma as chroma_helper

from backend.services.ranking import fused_score, bucket_for_score
from backend.services.skill_extractor import extract_skills as extract_skills_service


# ─────────────────────────────────────────────────────────────────────────────
#                          COMMON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LEXICON = [
    "python", "sql", "streamlit", "pytorch", "tensorflow", "keras", "scikit-learn",
    "pandas", "numpy", "docker", "kubernetes", "aws", "azure", "gcp", "react", "node",
    "java", "javascript", "c++", "git", "linux", "spark", "hadoop", "nlp", "cv",
    "flask", "fastapi", "rest", "graphql"
]

def _safe_unwrap(resp: Any, key: str) -> List:
    """
    Extract values safely from response returned by Chroma DB.
    Handles list/dict formats.
    """
    if isinstance(resp, dict) and key in resp:
        val = resp[key]
        if isinstance(val, list) and isinstance(val[0], list):
            return val[0]
        return val

    if isinstance(resp, list) and isinstance(resp[0], dict):
        return resp[0].get(key, [])

    return []


def _parse_skills_from_metadata(metadata: Dict[str, Any], description: str = "",
                                lexicon: Optional[List[str]] = None) -> Set[str]:
    """Extract skills from metadata or job description."""
    lex = set([s.lower() for s in (lexicon or DEFAULT_LEXICON)])
    skills_set = set()

    raw = metadata.get("skills") if metadata else None

    if isinstance(raw, list):
        skills_set.update([str(x).lower().strip() for x in raw])

    elif isinstance(raw, str):
        for p in re.split(r"[;,/|]", raw):
            p = p.strip().lower()
            if p:
                skills_set.add(p)

    # fallback: detect skills from description text
    if not skills_set and description:
        txt = description.lower()
        for term in lex:
            if re.search(r"\b" + re.escape(term) + r"\b", txt):
                skills_set.add(term)

    return skills_set


def _distance_to_similarity(distance: Optional[float]) -> float:
    """Convert Chroma DB distance into similarity [0-1]"""
    if distance is None:
        return 0.0

    try:
        sim = 1 - float(distance)
        return max(0.0, min(sim, 1.0))
    except:
        return 0.0



# ─────────────────────────────────────────────────────────────────────────────
#                          CHROMA BASED RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def recommend(
    self,
    resume_text: str,
    candidate_exp: int = 0,
    top_k: int = 10,
    role_filter: Optional[str] = None,  # ✅ New parameter
    use_llm: bool = False
) -> List[Dict[str, Any]]:
    """
    Recommend jobs based on TF-IDF + skill match + experience + role filtering
    """
    if not self.jobs or self.job_vectors is None:
        return []

    # Convert resume to TF-IDF vector
    query_vector = self.vectorizer.transform([resume_text])

    # Calculate similarity scores
    similarity_scores = linear_kernel(query_vector, self.job_vectors).flatten()

    # Extract skills from resume
    resume_skills = extract_skills_service(resume_text, use_llm=use_llm)

    # ✅ ROLE FILTER STEP
    filtered_jobs = []
    for idx, job in enumerate(self.jobs):

        if role_filter:
            title = job.get("title", "").lower()
            role_filter_lower = role_filter.lower()

            if role_filter_lower not in title:
                continue   # skip jobs that don’t match the role

        filtered_jobs.append((idx, job))

    # If no jobs matched filtering, fallback to ALL jobs
    if not filtered_jobs:
        filtered_jobs = list(enumerate(self.jobs))

    results = []
    for idx, job in filtered_jobs:
        job_skills = self.job_skills.get(job["id"], [])

        matched = set(job_skills) & set(resume_skills)
        missing = set(job_skills) - set(resume_skills)

        skill_score = len(matched) / max(len(job_skills), 1)

        job_exp = int(job.get("experience", 0))
        exp_score = min(candidate_exp / job_exp, 1) if job_exp > 0 else 1

        # Final score
        final_score = (similarity_scores[idx] * 0.4) + (skill_score * 0.4) + (exp_score * 0.2)

        results.append({
            "job": job,
            "similarity": float(similarity_scores[idx]),
            "skill_score": round(skill_score, 3),
            "experience_score": round(exp_score, 3),
            "score": round(final_score, 3),
            "matched_skills": list(matched),
            "missing_skills": list(missing),
            "job_skills": job_skills,
            "resume_skills": resume_skills,
            "required_experience": job_exp,
            "candidate_experience": candidate_exp,
        })

    # Sort by best score
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# ─────────────────────────────────────────────────────────────────────────────
#                        TF-IDF EXPERIENCE + SKILL RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFRecommender:
    """Matches job based on TF-IDF + Skill Match + Experience Match"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.job_vectors = None
        self.jobs = []
        self.job_skills = {}

    def fit(self, jobs: List[Dict[str, Any]]) -> None:
        """Store job vectors and extract job skills"""
        self.jobs = jobs
        descriptions = [j.get("description", "") for j in jobs]
        self.job_vectors = self.vectorizer.fit_transform(descriptions)

        for j in jobs:
            self.job_skills[j["id"]] = extract_skills_service(j["description"], use_llm=False)

    def recommend(
        self, resume_text: str, candidate_exp: int = 0, top_k: int = 10, use_llm: bool = False
    ) -> List[Dict[str, Any]]:

        query_vector = self.vectorizer.transform([resume_text])
        similarity_scores = linear_kernel(query_vector, self.job_vectors).flatten()

        resume_skills = extract_skills_service(resume_text, use_llm=use_llm)

        results = []
        for idx, job in enumerate(self.jobs):

            job_skill_list = self.job_skills.get(job["id"], [])

            matched = set(job_skill_list) & set(resume_skills)
            missing = set(job_skill_list) - set(resume_skills)
            skill_score = len(matched) / max(len(job_skill_list), 1)

            job_exp = int(job.get("experience", 0))
            exp_score = min(candidate_exp / job_exp, 1) if job_exp > 0 else 1

            final_score = (similarity_scores[idx] * 0.5) + (skill_score * 0.3) + (exp_score * 0.2)

            results.append(
                {
                    "job": job,
                    "similarity": float(similarity_scores[idx]),
                    "skill_score": round(skill_score, 3),
                    "experience_score": round(exp_score, 3),
                    "score": round(final_score, 3),

                    "matched_skills": list(matched),
                    "missing_skills": list(missing),
                    "job_skills": job_skill_list,
                    "resume_skills": resume_skills,
                    "required_experience": job_exp,
                    "candidate_experience": candidate_exp,
                }
            )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]



def bucketize_recommendations(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by difficulty levels for UI"""
    buckets = {"Safe": [], "Stretch": [], "Fallback": []}

    for r in results:
        if r["score"] >= 0.70:
            buckets["Safe"].append(r)
        elif r["score"] >= 0.45:
            buckets["Stretch"].append(r)
        else:
            buckets["Fallback"].append(r)

    return buckets
