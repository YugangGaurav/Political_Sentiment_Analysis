# Political Sentiment Intelligence UI

This folder adds a polished Streamlit interface around the existing
`Political_Sentiment_Analysis.ipynb` research notebook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Design

- Dark analytical dashboard
- Overview / Analyze Text / Reddit Monitor / Model Lab workspaces
- Confidence and model-agreement presentation
- Interactive Plotly charts
- Graceful handling of Reddit HTTP restrictions
- No API keys hard-coded into source

## Important

The UI intentionally does **not** claim Reddit is a guaranteed continuous stream.
It performs periodic retrieval of recent public submissions. Reddit may restrict
unauthenticated requests, in which case the UI reports the failure instead of
fabricating live data.

The notebook remains the source of truth for research experiments and detailed
benchmarking. The UI is the product layer.
