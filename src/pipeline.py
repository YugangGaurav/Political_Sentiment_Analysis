import re
import html
import urllib.request
import numpy as np
import pandas as pd
import emoji
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

LABELS = ["Negative", "Neutral", "Positive"]
lemmatizer = WordNetLemmatizer()

CONTRACTION_MAP = {
    "don't":"do not","doesn't":"does not","didn't":"did not","can't":"cannot",
    "won't":"will not","he's":"he is","it's":"it is","i'm":"i am",
    "they're":"they are","we're":"we are","you're":"you are",
    "isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not",
    "haven't":"have not","hasn't":"has not","hadn't":"had not"
}
NEGATION_START = {"not","never","neither","nor","cannot","barely","hardly","scarcely"}
PRESERVED_WORDS = {"no","without","none","nothing","not","never","neither","nor"}

def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text, flags=re.I)
    text = re.sub(r"(\w+)n't\b", r"\1 not", text, flags=re.I)
    return text

def split_hashtag(match):
    tag = match.group(1)
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", tag)
    return " ".join(words) if words else tag

def clean_political_text(text, apply_neg_tag=True):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = expand_contractions(text)
    text = emoji.demojize(text, delimiters=(" "," "))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[ru]/\w+", " ", text)
    text = re.sub(r"#(\w+)", split_hashtag, text).lower()
    text = re.sub(r"[^a-z_\s]", " ", text)
    tokens = word_tokenize(text)
    active_stopwords = set(stopwords.words("english")) - PRESERVED_WORDS
    processed, neg_active, dist = [], False, 0
    for token in tokens:
        if token in NEGATION_START:
            neg_active, dist = True, 0
            processed.append(token)
            continue
        if token in active_stopwords:
            continue
        lemma = lemmatizer.lemmatize(token)
        if apply_neg_tag and neg_active:
            dist += 1
            if dist <= 2:
                processed.append(f"not_{lemma}")
            else:
                neg_active = False
                processed.append(lemma)
        else:
            processed.append(lemma)
    return " ".join(processed)

def _download_tweeteval():
    base = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
    texts = urllib.request.urlopen(base+"train_text.txt", timeout=20).read().decode("utf-8").splitlines()
    labels = [int(x) for x in urllib.request.urlopen(base+"train_labels.txt", timeout=20).read().decode("utf-8").splitlines()]
    mapping = {0:"Negative",1:"Neutral",2:"Positive"}
    raw = pd.DataFrame({"text":texts,"sentiment":[mapping[x] for x in labels]})
    parts = []
    for label in LABELS:
        parts.append(raw[raw.sentiment == label].sample(n=3500, random_state=42))
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

def _train():
    df = _download_tweeteval()
    df["cleaned"] = df["text"].map(clean_political_text)
    df = df[df.cleaned.str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df.cleaned, df.sentiment, test_size=.2, random_state=42, stratify=df.sentiment
    )
    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=.85, max_features=12000)
    Xt = vec.fit_transform(X_train)
    Xv = vec.transform(X_test)
    model = LogisticRegression(C=1.5, max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(Xt, y_train)
    pred = model.predict(Xv)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro")
    }
    return {"vectorizer":vec, "model":model, "vader":SentimentIntensityAnalyzer(), "metrics":metrics}

def get_engine():
    return _train()

def _vader_distribution(compound):
    scores = np.array([-compound*1.6, (1-abs(compound))*.9, compound*1.6])
    exp_s = np.exp(scores - np.max(scores))
    return exp_s / exp_s.sum()

def analyze_text(engine, text):
    cleaned = clean_political_text(text)
    if not cleaned:
        return {"sentiment":"Neutral","confidence":.33,"status":"Low Confidence","ml_prediction":"Neutral","ml_confidence":.33,"vader_prediction":"Neutral","vader_compound":0.0,"textblob_prediction":"Neutral","textblob_polarity":0.0,"agreement_summary":"Empty text input","cleaned_text":""}
    feat = engine["vectorizer"].transform([cleaned])
    probs = engine["model"].predict_proba(feat)[0]
    classes = list(engine["model"].classes_)
    i = int(np.argmax(probs))
    ml_pred, ml_conf = classes[i], float(probs[i])
    vc = float(engine["vader"].polarity_scores(text)["compound"])
    vp = "Positive" if vc >= .05 else ("Negative" if vc <= -.05 else "Neutral")
    tb = float(TextBlob(text).sentiment.polarity)
    tp = "Positive" if tb > .05 else ("Negative" if tb < -.05 else "Neutral")
    fused = .70 * probs + .30 * _vader_distribution(vc)
    fi = int(np.argmax(fused))
    final, conf = classes[fi], float(fused[fi])
    signals = [ml_pred, vp, tp]
    counts = {x:signals.count(x) for x in LABELS}
    winner = max(counts, key=counts.get)
    agreement = f"{counts[winner]}/3 {winner}-supporting signals" if counts[winner] >= 2 else "Divided signals across models (mixed sentiment)"
    if (ml_pred != "Negative" and vc <= -.50) or (ml_pred != "Positive" and vc >= .50):
        status = "Low (Model Disagreement)"
    elif conf >= .55: status = "High"
    elif conf >= .45: status = "Medium"
    else: status = "Low"
    return {
        "sentiment":final,"confidence":round(conf,3),"status":status,
        "ml_prediction":ml_pred,"ml_confidence":round(ml_conf,3),
        "vader_prediction":vp,"vader_compound":round(vc,3),
        "textblob_prediction":tp,"textblob_polarity":round(tb,3),
        "agreement_summary":agreement,"cleaned_text":cleaned
    }
