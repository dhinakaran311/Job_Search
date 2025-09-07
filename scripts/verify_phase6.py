# scripts/verify_phase6.py
"""
Manual verification script. Run this after installing requirements and setting .env.
It prints outputs for both LLM (if key present) and fallback behavior so you can confirm things work.
"""

from backend.services.skill_extractor import extract_skills
from backend.services.llm import job_summary_1line, upskilling_plan_bulleted
import os

sample_resume = """
John Doe
Experienced ML engineer. Built ETL pipelines and trained models using Python, PyTorch, Docker and AWS.
Worked on NLP, computer vision, and REST APIs.
Contact: john@example.com
"""

print("=== Using skill_extractor (LLM if enabled / fallback lexicon otherwise) ===")
skills = extract_skills(sample_resume)
print("Extracted skills:", skills)

print("\n=== LLM Job summary (may be empty if no key) ===")
summary = job_summary_1line("Looking for ML engineer to implement training pipelines and deploy models.")
print("Job summary:", summary or "<empty — likely fallback/no-key>")

print("\n=== LLM Upskilling plan (may be empty if no key) ===")
plan = upskilling_plan_bulleted("Machine Learning Engineer", ["docker", "aws"], context="2 years Python")
print("Upskilling plan:\n", plan or "<empty — likely fallback/no-key>")

print("\nDone. If outputs are empty and you have set OPENAI_API_KEY, ensure the key is valid and the SDK is installed.")
