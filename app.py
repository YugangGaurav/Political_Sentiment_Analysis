import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from src.pipeline import get_engine, analyze_text
from src.reddit import fetch_reddit_posts

st.set_page_config(
    page_title="Political Sentiment Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 15% 0%, rgba(99,102,241,.12), transparent 28%),
      radial-gradient(circle at 85% 10%, rgba(14,165,233,.09), transparent 26%),
      #080b12;
    color: #eef2ff;
}
.block-container {max-width: 1400px; padding-top: 2rem;}
[data-testid="stSidebar"] {
    background: rgba(10,13,22,.96);
    border-right: 1px solid rgba(255,255,255,.07);
}
.hero {
    padding: 28px 30px;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(20,25,40,.92), rgba(12,16,28,.78));
    box-shadow: 0 20px 70px rgba(0,0,0,.25);
    margin-bottom: 22px;
}
.eyebrow {color:#8b93ff; font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;}
.hero h1 {font-size:2.4rem; line-height:1.05; margin:.35rem 0 .7rem;}
.hero p {color:#aeb7ca; max-width:850px; font-size:1rem;}
.card {
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(16,20,32,.82);
    border-radius: 18px;
    padding: 18px;
    min-height: 118px;
}
.metric-label {color:#8993aa; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;}
.metric-value {font-size:1.75rem; font-weight:800; margin-top:8px;}
.muted {color:#8f9ab1;}
.sentiment-positive {color:#34d399;}
.sentiment-negative {color:#fb7185;}
.sentiment-neutral {color:#94a3b8;}
.badge {
    display:inline-block; padding:5px 10px; border-radius:999px;
    font-size:.72rem; font-weight:800; letter-spacing:.04em;
    background:rgba(255,255,255,.07);
}
.section-title {font-size:1.15rem; font-weight:800; margin:18px 0 10px;}
[data-testid="stTextArea"] textarea {
    background:#0c111c !important;
    border:1px solid rgba(255,255,255,.1) !important;
    border-radius:14px !important;
    color:#f8fafc !important;
}
div.stButton > button {
    border-radius:12px;
    font-weight:700;
}
.small-note {font-size:.78rem; color:#77839a;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading NLP engine…")
def load_engine():
    return get_engine()

engine = load_engine()

with st.sidebar:
    st.markdown("### ◉ Political Intelligence")
    st.caption("NLP • ML • Reddit analytics")
    page = st.radio(
        "Workspace",
        ["Overview", "Analyze Text", "Reddit Monitor", "Model Lab"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Engine**")
    st.markdown("`TF-IDF` 1–2 grams")
    st.markdown("`Logistic Regression`")
    st.markdown("`VADER` + `TextBlob`")
    st.markdown("`3-class sentiment`")
    st.divider()
    st.caption("Research dashboard • Reddit retrieval is best-effort and may be restricted by Reddit.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">Political Sentiment Intelligence</div>
  <h1>Understand the mood behind political discourse.</h1>
  <p>Analyze text with a calibrated NLP pipeline, inspect model agreement, and monitor recent Reddit submissions without exposing model internals to the user.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="section-title">System overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("10,500+", "TweetEval samples"),
        ("3", "Sentiment classes"),
        ("1–2", "TF-IDF n-grams"),
        ("3", "Model signals"),
    ]
    for col,(value,label) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("01","Clean","Negation-aware preprocessing"),
        ("02","Represent","TF-IDF unigrams + bigrams"),
        ("03","Predict","ML + VADER + TextBlob"),
        ("04","Audit","Confidence + agreement"),
    ]
    for col,(n,title,desc) in zip(cols,steps):
        col.markdown(f'<div class="card"><span class="badge">{n}</span><h3>{title}</h3><span class="muted">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick analysis</div>', unsafe_allow_html=True)
    sample = st.text_area("Try a political statement", "The policy sounds promising, but the implementation has been disappointing.", height=110)
    if st.button("Analyze now", type="primary", use_container_width=True):
        result = analyze_text(engine, sample)
        st.session_state["last_result"] = result
        st.rerun()

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        cls = r["sentiment"].lower()
        st.markdown(f'<div class="card"><div class="metric-label">Result</div><div class="metric-value sentiment-{cls}">{r["sentiment"]}</div><span class="muted">Confidence: {r["confidence"]:.0%} • {r["status"]}</span></div>', unsafe_allow_html=True)

elif page == "Analyze Text":
    st.markdown('<div class="section-title">Text analyzer</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Political text",
        placeholder="Paste a Reddit post, political statement, headline, or social-media text…",
        height=190,
    )
    if st.button("Run sentiment analysis", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            with st.spinner("Analyzing…"):
                r = analyze_text(engine, text)
            st.markdown(f'<div class="card"><div class="metric-label">Final sentiment</div><div class="metric-value sentiment-{r["sentiment"].lower()}">{r["sentiment"]}</div><span class="muted">Confidence {r["confidence"]:.0%} • {r["status"]}</span></div>', unsafe_allow_html=True)
            st.markdown("### Model signals")
            a,b,c = st.columns(3)
            a.metric("ML model", f'{r["ml_prediction"]}', f'{r["ml_confidence"]:.0%} confidence')
            b.metric("VADER", f'{r["vader_prediction"]}', f'{r["vader_compound"]:+.2f}')
            c.metric("TextBlob", f'{r["textblob_prediction"]}', f'{r["textblob_polarity"]:+.2f}')
            st.info(r["agreement_summary"])
            with st.expander("Show preprocessing"):
                st.code(r["cleaned_text"])

elif page == "Reddit Monitor":
    st.markdown('<div class="section-title">Reddit monitor</div>', unsafe_allow_html=True)
    left,right = st.columns([3,1])
    with left:
        subs = st.multiselect("Communities", ["politics","politicaldiscussion","worldnews"], default=["politics","politicaldiscussion","worldnews"])
    with right:
        limit = st.number_input("Posts / community", 5, 30, 15)
    if st.button("Refresh Reddit", type="primary", use_container_width=True):
        with st.spinner("Retrieving recent public submissions…"):
            df, status = fetch_reddit_posts(subs, int(limit), engine)
        st.session_state["reddit_df"] = df
        st.session_state["reddit_status"] = status

    if "reddit_df" in st.session_state:
        df = st.session_state["reddit_df"]
        st.caption(st.session_state.get("reddit_status",""))
        if df.empty:
            st.warning("No live Reddit submissions were returned. This can happen when Reddit restricts unauthenticated requests.")
        else:
            counts = df["sentiment"].value_counts().reindex(["Positive","Neutral","Negative"], fill_value=0)
            a,b,c = st.columns(3)
            a.metric("Positive", int(counts["Positive"]))
            b.metric("Neutral", int(counts["Neutral"]))
            c.metric("Negative", int(counts["Negative"]))
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={"x":"Sentiment","y":"Posts"},
                title="Current sentiment distribution",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            display_cols = ["subreddit","title","sentiment","confidence","score","num_comments","timestamp"]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Click Refresh Reddit to retrieve a fresh batch.")

elif page == "Model Lab":
    st.markdown('<div class="section-title">Model lab</div>', unsafe_allow_html=True)
    st.write("The UI uses the same core modeling choices documented in the notebook. For portfolio transparency, model metrics should come from the notebook's held-out evaluation rather than being invented in the dashboard.")
    metrics = pd.DataFrame([
        {"Model":"Logistic Regression","Role":"Primary supervised model","Probability":"Yes"},
        {"Model":"Calibrated LinearSVC","Role":"Benchmark","Probability":"Calibrated"},
        {"Model":"MultinomialNB","Role":"Benchmark","Probability":"Yes"},
        {"Model":"VADER","Role":"Lexicon signal","Probability":"Score"},
        {"Model":"TextBlob","Role":"Lexicon signal","Probability":"Polarity"},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.info("For the most reliable portfolio presentation, show the exact accuracy/F1 values produced by the notebook's current Run All output here rather than hard-coding metrics.")
