"""Lightweight narrative extraction pipeline.

No GPU or large models required — runs comfortably on Streamlit Community Cloud.
Dependencies: spaCy en_core_web_sm · vaderSentiment · dateparser · beautifulsoup4
"""

import re
import html as _html

from bs4 import BeautifulSoup
import dateparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

_CAUSAL_CONNECTIVES = [
    "because",
    "caused",
    "led to",
    "resulted in",
    "due to",
    "triggered",
    "prompted",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "owing to",
    "attributed to",
]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(raw: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    if "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        raw = soup.get_text(separator=" ")
    raw = _html.unescape(raw)
    return re.sub(r"\s+", " ", raw).strip()


# ---------------------------------------------------------------------------
# Extraction helpers (all private — call extract_narrative() instead)
# ---------------------------------------------------------------------------

def _extract_entities(doc) -> dict:
    people, orgs, places = [], [], []
    seen: set = set()
    for ent in doc.ents:
        key = (ent.text.strip().lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        t = ent.text.strip()
        if ent.label_ == "PERSON":
            people.append(t)
        elif ent.label_ == "ORG":
            orgs.append(t)
        elif ent.label_ in ("GPE", "LOC"):
            places.append(t)
    return {"people": people, "organisations": orgs, "places": places}


def _extract_temporal(doc) -> list:
    results, seen = [], set()
    for ent in doc.ents:
        if ent.label_ not in ("DATE", "TIME"):
            continue
        k = ent.text.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        try:
            parsed = dateparser.parse(
                ent.text,
                settings={
                    "PREFER_DAY_OF_MONTH": "first",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                },
            )
            normalized = parsed.strftime("%Y-%m-%d") if parsed else None
        except Exception:
            normalized = None
        results.append({"text": ent.text.strip(), "normalized": normalized})
    return results


def _extract_svo(doc) -> list:
    """Extract subject-verb-object triples from the dependency parse."""
    triples, seen = [], set()
    for sent in doc.sents:
        for token in sent:
            if token.dep_ not in ("nsubj", "nsubjpass"):
                continue
            verb = token.head
            if verb.pos_ != "VERB":
                continue
            obj = next(
                (c for c in verb.children if c.dep_ in ("dobj", "attr", "pobj", "oprd")),
                None,
            )
            if obj is None:
                continue
            key = (token.lemma_, verb.lemma_, obj.lemma_)
            if key in seen:
                continue
            seen.add(key)
            triples.append({"subject": token.text, "verb": verb.text, "object": obj.text})
            if len(triples) >= 6:
                return triples
    return triples


def _extract_causal(text: str) -> list:
    """Find cause-effect pairs using connective patterns."""
    results = []
    for connective in _CAUSAL_CONNECTIVES:
        pattern = re.compile(
            rf"([^.?!\n]{{5,150}}?)\s+{re.escape(connective)}\s+([^.?!\n]{{5,150}})",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            results.append(
                {
                    "cause": m.group(1).strip(),
                    "connective": connective,
                    "effect": m.group(2).strip(),
                }
            )
            if len(results) >= 5:
                return results
    return results


def _extractive_summary(doc, n: int = 2) -> str:
    """Score sentences by entity density + opening-position bias; return top-n."""
    scored = []
    for i, sent in enumerate(doc.sents):
        score = len(sent.ents) + (1.5 if i == 0 else 0.0)
        scored.append((score, sent.text.strip()))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(t for _, t in scored[:n])


def _get_sentiment(text: str) -> dict:
    c = _sia.polarity_scores(text)["compound"]
    label = "POSITIVE" if c >= 0.05 else ("NEGATIVE" if c <= -0.05 else "NEUTRAL")
    return {"label": label, "score": round(c, 3)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_narrative(text: str, nlp) -> dict:
    """Run the full lightweight pipeline and return a structured result dict.

    Parameters
    ----------
    text:
        Raw input (plain text or HTML).
    nlp:
        Pre-loaded spaCy Language model (use ``spacy.load("en_core_web_sm")``).

    Returns
    -------
    dict
        Keys: ``summary``, ``sentiment``, ``entities``, ``temporal``,
        ``actions``, ``causal``.
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)
    return {
        "summary":   _extractive_summary(doc),
        "sentiment": _get_sentiment(cleaned),
        "entities":  _extract_entities(doc),
        "temporal":  _extract_temporal(doc),
        "actions":   _extract_svo(doc),
        "causal":    _extract_causal(cleaned),
    }
