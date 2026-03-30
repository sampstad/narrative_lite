# 📰 Narrative Extractor

A lightweight Streamlit app that extracts the key narrative elements from any news article or paragraph — **who, what, when, where, and why** — using only small, CPU-friendly NLP tools.

## What it extracts

| Section | Method |
|---|---|
| **Summary** | Extractive (top-2 entity-rich sentences) |
| **Tone** | VADER lexicon-based sentiment |
| **People / Orgs / Places** | spaCy NER (`en_core_web_sm`) |
| **Time references** | spaCy DATE/TIME entities + dateparser normalisation |
| **Key actions** | Subject-Verb-Object triples from dependency parse |
| **Causal links** | Regex on connectives (*because, led to, resulted in*, …) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
narrative_lite/
├── app.py            # Streamlit UI
├── pipeline.py       # NLP extraction logic
├── requirements.txt  # Python dependencies
├── runtime.txt       # Python version for Streamlit Cloud
└── .streamlit/
    └── config.toml   # Theme configuration
```
