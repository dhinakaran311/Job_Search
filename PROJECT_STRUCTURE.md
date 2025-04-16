# Project Structure Reorganization Plan

## Current Issues
- Test files scattered in root directory (test_*.py)
- No clear separation between unit tests and integration tests
- Scripts could be better organized
- Documentation files in root
- Configuration files scattered

## Proposed Structure

```
Job_Search/
├── api/                          # FastAPI REST API
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   ├── middleware.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── routers/
│       ├── __init__.py
│       ├── jobs.py
│       ├── resume.py
│       ├── recommendations.py
│       └── skills.py
│
├── app_streamlit/                # Streamlit Web Application
│   ├── __init__.py
│   ├── app.py
│   └── data/                     # Streamlit-specific data
│
├── backend/                      # Core Business Logic
│   ├── __init__.py
│   ├── config.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── chroma.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── job_api_adzuna.py
│   │   ├── job_loader.py
│   │   └── resume_parser.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   ├── llm.py
│   │   ├── ranking.py
│   │   ├── retrieval.py
│   │   └── skill_extractor.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                        # All Test Files
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   │   ├── __init__.py
│   │   ├── test_skill_extractor.py
│   │   ├── test_resume_parser.py
│   │   └── test_recommendation.py
│   ├── integration/              # Integration tests
│   │   ├── __init__.py
│   │   ├── test_adzuna_api.py
│   │   └── test_api_endpoints.py
│   └── legacy/                   # Old test scripts (for reference)
│       ├── test_simple.py
│       ├── test_updated.py
│       ├── test_api_direct.py
│       ├── test_embedding_download.py
│       └── test.py
│
├── scripts/                      # Utility Scripts
│   ├── setup/
│   │   ├── check_dependencies.py
│   │   └── check_ollama.py
│   ├── dev/
│   │   ├── check_device.py
│   │   └── verify_phase6.py
│   └── run/
│       ├── start_api.py
│       └── start_streamlit.py
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── API.md
│   ├── SETUP.md
│   └── ARCHITECTURE.md
│
├── config/                       # Configuration Files
│   ├── pytest.ini
│   └── .env.example
│
├── data/                         # Application Data
│   ├── cache/
│   ├── resumes/
│   └── sample_jobs.json
│
├── chroma_db/                    # Vector Database Storage
│
├── logs/                         # Log Files (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md                     # Main project README
```

<!-- Updated: 2025-04-15 16:52:44 -->
<!-- Updated: 2025-04-16 13:17:48 -->