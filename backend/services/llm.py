# import os
# import json
# from typing import List, Optional

# try:
#     from openai import OpenAI, APIError
# except ImportError:
#     from typing import Any
#     OpenAI = Any
#     APIError = Exception

# _CLIENT: Optional[OpenAI] = None
# _MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# def _get_client() -> Optional[OpenAI]:
#     """Initialize and return a singleton OpenAI client."""
#     global _CLIENT
#     if OpenAI is None:
#         return None
#     if _CLIENT is None:
#         api_key = os.environ.get("OPENAI_API_KEY")
#         if not api_key:
#             return None
#         _CLIENT = OpenAI(api_key=api_key)
#     return _CLIENT

# def _call_llm(prompt: str, json_mode: bool = False) -> str:
#     """Generic helper to call the LLM and return the text response."""
#     client = _get_client()
#     if not client:
#         return ""

#     try:
#         response_format = {"type": "json_object"} if json_mode else {"type": "text"}
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             model=_MODEL_NAME,
#             response_format=response_format,
#         )
#         content = chat_completion.choices[0].message.content
#         return content or ""
#     except APIError as e:
#         print(f"OpenAI API error: {e}")
#         return ""
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return ""

# def extract_skills_llm(text: str) -> List[str]:
#     """Extract skills from text using LLM, returns a list of strings."""
#     prompt = (
#         "Extract the technical skills from the following text. "
#         "Return the skills as a JSON list of strings. Example: [\"python\", \"react\", \"sql\"]. "
#         "If no skills are found, return an empty list. Text: \n"
#         f'"""{text}"""'
#     )
    
#     response_text = _call_llm(prompt, json_mode=True)
#     if not response_text:
#         return []

#     try:
#         data = json.loads(response_text)
#         for key in data:
#             if isinstance(data[key], list):
#                 return [str(skill).lower() for skill in data[key]]
#         return []
#     except (json.JSONDecodeError, TypeError, KeyError):
#         try:
#             skills = json.loads(response_text)
#             if isinstance(skills, list):
#                 return [str(skill).lower() for skill in skills]
#         except (json.JSONDecodeError, TypeError):
#             return []
#     return []

# def job_summary_1line(text: str) -> str:
#     """Create a one-line summary of a job description."""
#     prompt = f"Summarize the following job description in a single, concise sentence: \n" \
#              f'"""{text}"""'
#     return _call_llm(prompt).strip()

# def upskilling_plan_bulleted(job_title: str, missing_skills: List[str], context: str = "") -> str:
#     """Generate a bulleted upskilling plan."""
#     skills_str = ", ".join(missing_skills)
#     prompt = (
#         f"Create a concise, bulleted upskilling plan for a candidate aiming for the job title '{job_title}'. "
#         f"The candidate's background is: '{context}'. "
#         f"The plan should focus on acquiring these missing skills: {skills_str}. "
#         "Provide 3-5 actionable bullet points."
#     )
#     return _call_llm(prompt).strip()

# Ollama-based LLM integration
import json
from typing import List
import ollama

OLLAMA_MODEL = "mistral"  # You can change to 'llama2', 'phi3', etc.

def _call_llm(prompt: str, json_mode: bool = False) -> str:
    """Call Ollama LLM and return the response text."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


def extract_skills_llm(text: str) -> List[str]:
    """
    Extract skills from text using Ollama LLM, returns a list of strings.
    Uses safe parsing methods (json.loads and ast.literal_eval) instead of eval().
    """
    import ast
    
    prompt = (
        "Extract the technical skills from the following text. "
        "Return the skills as a JSON list of strings. Example: [\"python\", \"react\", \"sql\"]. "
        "If no skills are found, return an empty list. Text: \n"
        f'"""{text}"""'
    )
    
    response_text = _call_llm(prompt).strip()
    
    # Try JSON parsing first (safest option)
    try:
        # Clean up common formatting issues
        clean_text = response_text.strip()
        if clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        # First try: parse as JSON
        try:
            skills = json.loads(clean_text)
            if isinstance(skills, list):
                return [str(skill).lower().strip() for skill in skills]
        except json.JSONDecodeError:
            # If JSON parsing fails, try with single quotes replaced by double quotes
            try:
                skills = json.loads(clean_text.replace("'", '"'))
                if isinstance(skills, list):
                    return [str(skill).lower().strip() for skill in skills]
            except json.JSONDecodeError:
                pass
        
        # Second try: use ast.literal_eval (safer than eval)
        try:
            skills = ast.literal_eval(clean_text)
            if isinstance(skills, list):
                return [str(skill).lower().strip() for skill in skills]
        except (ValueError, SyntaxError, TypeError):
            pass
            
    except Exception as e:
        # Log the error if needed
        print(f"Error parsing skills: {e}")
    
    # If all parsing attempts fail, return an empty list
    return []
    return []


def job_summary_1line(text: str) -> str:
    """Create a one-line summary of a job description using Ollama."""
    prompt = f"Summarize the following job description in a single, concise sentence: \n\"\"\"{text}\"\"\""
    return _call_llm(prompt).strip()


def upskilling_plan_bulleted(job_title: str, missing_skills: List[str], context: str = "") -> str:
    """Generate a bulleted upskilling plan using Ollama."""
    skills_str = ", ".join(missing_skills)
    prompt = (
        f"Create a concise, bulleted upskilling plan for a candidate aiming for the job title '{job_title}'. "
        f"The candidate's background is: '{context}'. "
        f"The plan should focus on acquiring these missing skills: {skills_str}. "
        "Provide 3-5 actionable bullet points."
    )
    return _call_llm(prompt).strip()
