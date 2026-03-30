"""Narrative Extractor — Streamlit front-end."""

import spacy
import streamlit as st

from pipeline import extract_narrative

st.set_page_config(
    page_title="Narrative Extractor",
    page_icon="📰",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model (loaded once, cached for the process lifetime)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading language model…")
def _load_nlp():
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_SENTIMENT_STYLE: dict = {
    "POSITIVE": ("🟢", "#d4edda", "#155724"),
    "NEGATIVE": ("🔴", "#f8d7da", "#721c24"),
    "NEUTRAL":  ("⚪", "#e2e3e5", "#383d41"),
}


def _sentiment_badge(label: str, score: float) -> str:
    icon, bg, fg = _SENTIMENT_STYLE[label]
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 12px;'
        f'border-radius:14px;font-size:0.85rem;font-weight:600;">'
        f'{icon} {label} ({score:+.2f})</span>'
    )


def _render_results(result: dict) -> None:
    sent = result["sentiment"]

    # ── Summary ──────────────────────────────────────────────────────────────
    st.markdown("### Summary")
    st.info(result["summary"] or "_Insufficient text for a summary._", icon="📝")
    st.markdown(_sentiment_badge(sent["label"], sent["score"]), unsafe_allow_html=True)
    st.write("")
    st.divider()

    # ── Entities ─────────────────────────────────────────────────────────────
    st.markdown("### Entities")
    ents = result["entities"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**👤 People**")
        st.markdown(
            "\n".join(f"- {p}" for p in ents["people"]) if ents["people"] else "_none_"
        )
    with c2:
        st.markdown("**🏢 Organisations**")
        st.markdown(
            "\n".join(f"- {o}" for o in ents["organisations"])
            if ents["organisations"]
            else "_none_"
        )
    with c3:
        st.markdown("**📍 Places**")
        st.markdown(
            "\n".join(f"- {pl}" for pl in ents["places"]) if ents["places"] else "_none_"
        )
    st.divider()

    # ── Time references ───────────────────────────────────────────────────────
    st.markdown("### Time References")
    if result["temporal"]:
        for t in result["temporal"]:
            norm = f" → `{t['normalized']}`" if t["normalized"] else ""
            st.markdown(f"- **{t['text']}**{norm}")
    else:
        st.caption("None found.")
    st.divider()

    # ── Key actions ───────────────────────────────────────────────────────────
    st.markdown("### Key Actions")
    if result["actions"]:
        for a in result["actions"]:
            st.markdown(f"- *{a['subject']}* **{a['verb']}** {a['object']}")
    else:
        st.caption("None found.")
    st.divider()

    # ── Causal links ──────────────────────────────────────────────────────────
    st.markdown("### Causal Links")
    if result["causal"]:
        for c in result["causal"]:
            st.markdown(
                f'<div style="border-left:3px solid #4A90E2;padding:6px 12px;'
                f'margin:6px 0;border-radius:0 6px 6px 0;background:#f0f4ff;">'
                f'<small style="color:#888;">via <em>{c["connective"]}</em></small><br>'
                f'<strong>{c["cause"]}</strong>'
                f'<span style="color:#4A90E2;margin:0 8px;">→</span>'
                f'{c["effect"]}'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("None found.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📰 Narrative Extractor")
    st.caption(
        "Paste a news article or paragraph to extract its key narrative elements — "
        "who, what, when, where, and why."
    )

    nlp = _load_nlp()

    with st.form("analyse_form"):
        text_input = st.text_area(
            "Article text",
            height=220,
            placeholder="Paste your article or paragraph here…",
            help="Plain text or HTML are both supported.",
        )
        submitted = st.form_submit_button("Analyse ✨", type="primary")

    if submitted:
        text = text_input.strip()
        if len(text) < 30:
            st.warning("Please enter at least a sentence or two.", icon="⚠️")
        else:
            with st.spinner("Extracting narrative…"):
                result = extract_narrative(text, nlp)
            _render_results(result)

    st.markdown("---")
    st.caption("Powered by spaCy · VADER · dateparser")


if __name__ == "__main__":
    main()
