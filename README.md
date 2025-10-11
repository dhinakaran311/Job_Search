# Job_Search

**Job_Search** is an intelligent job recommender system built with Python and Streamlit. It ingests job postings from offline data or real-time APIs, extracts key skills from user resumes (using LLM or a lexicon-based approach), matches candidates to suitable jobs, and provides upskilling plans to improve employability. The app supports resume uploads, interactive recommendations, skill-based queries, and CSV exports for further analysis.

---

## Features

- **Resume Upload/Paste**: Accepts PDF, DOCX, TXT, or Markdown resumes for parsing.
- **Job Data Ingestion**:
  - Offline sample jobs from `data/sample_jobs.json`
  - Real-time jobs via Adzuna API (requires API keys)
- **Skill Extraction**: Uses LLM (Ollama/OpenAI) or lexicon-based extraction from resumes and job descriptions.
- **Job Matching & Ranking**: 
  - TF-IDF-based similarity scoring between resume and job descriptions.
  - Skill overlap analysis for deeper matching.
  - Results bucketed as "Safe", "Stretch", and "Fallback" recommendations.
- **LLM-Powered Summaries**: Optionally generates job summaries and personalized upskilling plans.
- **CSV Export**: Download recommendations for further review.
- **Streamlit UI**: Modern, interactive web app interface.
- **Logging & Controls**: Sidebar controls for job ingestion, API queries, and logs.

---

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- (Optional) Adzuna and OpenAI/Ollama API keys for live job search and LLM features

### Installation

```bash
git clone https://github.com/dhinakaran311/Job_Search.git
cd Job_Search
pip install -r requirements.txt
```

### Usage

1. **Start the Streamlit app**:
    ```bash
    streamlit run app_streamlit/app.py
    ```
2. **Ingest Jobs**:
    - Use the sidebar to ingest offline jobs or search real-time jobs via Adzuna API.
3. **Resume Input**:
    - Upload a resume file or paste resume text.
    - Specify target job role.
4. **Recommend Jobs**:
    - Click "Recommend Jobs" to view bucketed recommendations.
    - Export results to CSV as needed.

---

## File Structure

```
Job_Search/
├── app_streamlit/
│   └── app.py                  # Streamlit UI entrypoint
├── backend/
│   ├── services/               # Skill extraction, LLM interfaces
│   ├── ingestion/              # Job ingestion (Adzuna, offline)
│   ├── db/                     # Chroma DB utilities
│   └── utils/                  # Misc utilities
├── data/
│   └── sample_jobs.json        # Example job data
├── scripts/                    # Scripts for data handling
├── .gitignore
└── README.md
```

---

## Environment Variables

Create a `.env` file in your root directory for sensitive keys:
```
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_APP_KEY=your_adzuna_app_key
```

---

## How it Works

- **Job Ingestion**: Loads jobs from JSON file or Adzuna API.
- **Resume Parsing**: Extracts text from uploaded files.
- **Skill Extraction**: Applies LLM/model or keyword matching to identify skills.
- **Recommendation**: Computes similarity, skill overlap, and ranks jobs into buckets.
- **Upskilling Plan**: Suggests learning paths for missing skills (if LLM enabled).

---

## Contributing

Pull requests and issues are welcome! For major changes, please open an issue first to discuss what you would like to change.

---


## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Adzuna API](https://developer.adzuna.com/)
- [OpenAI](https://openai.com/)
- [Ollama](https://ollama.com/)

