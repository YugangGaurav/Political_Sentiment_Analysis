# 🗳️ Political Sentiment Analysis

An NLP and Machine Learning project for analyzing sentiment in political and social-media text. The system classifies text into **Positive, Negative, and Neutral** sentiment categories and includes model-based and lexicon-based sentiment analysis.

## 🚀 Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YugangGaurav/Political_Sentiment_Analysis/blob/main/Political_Sentiment_Analysis.ipynb)

Click the button above to open the notebook directly in Google Colab.

## 🚀 Open as Political Sentiment Analysis web
[(https://colab.research.google.com/github/YugangGaurav/Political_Sentiment_Analysis/blob/main/Political_Sentiment_Analysis.ipynb)](https://politicalsentimentanalysis-cozitsyug.streamlit.app/)

Click the button above to open the website.

## 📌 Project Overview

Political discussions contain complex language, criticism, sarcasm, praise, and mixed opinions. This project applies Natural Language Processing and Machine Learning techniques to identify the overall sentiment expressed in political text.

### Main objectives

- Clean and preprocess social-media text
- Convert text into numerical features using TF-IDF
- Train and compare sentiment-classification models
- Analyze Positive, Negative, and Neutral sentiment
- Compare Machine Learning predictions with lexicon-based methods
- Test the classifier on new political text
- Analyze recent public Reddit submissions for near-real-time sentiment analysis

## 🧠 Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- NLTK
- VADER Sentiment
- TextBlob
- Matplotlib
- Seaborn
- TF-IDF
- Linear SVM / Logistic Regression / Naive Bayes

## 🔄 NLP Pipeline

```text
Raw Text
   ↓
Text Cleaning
   ↓
Tokenization & Normalization
   ↓
Negation-Aware Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Model
   ↓
Sentiment Prediction
   ↓
Positive / Negative / Neutral
```

## 📊 Model Evaluation

The notebook evaluates sentiment models using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The final model should be selected based on validation performance rather than assuming that one particular algorithm is always best.

## 🌐 Reddit Analysis

The project can retrieve recent public submissions from selected political/news-related Reddit communities and apply the trained sentiment model to the retrieved text.

The Reddit component is intended for **near-real-time analysis through periodic retrieval**, rather than a guaranteed continuous streaming connection.

## ⚠️ Limitations

Political sentiment analysis is challenging because political text often contains:

- Sarcasm
- Irony
- Negation
- Mixed sentiment
- Context-dependent language
- Domain-specific terminology

Therefore, model predictions should be treated as automated estimates rather than definitive interpretations of political opinion.

## ▶️ Running the Project

### Google Colab

1. Click **Open in Colab** above.
2. Allow the required notebook permissions.
3. Run the cells from top to bottom.
4. Install any dependencies requested by the notebook.
5. Run the training and evaluation sections.
6. Use the live-analysis section to test new political text.

### Local Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/YugangGaurav/Political_Sentiment_Analysis.git
cd Political_Sentiment_Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
Political_Sentiment_Analysis.ipynb
```

## 📁 Repository Structure

```text
Political_Sentiment_Analysis/
│
├── Political_Sentiment_Analysis.ipynb
├── README.md
└── requirements.txt
```

## 👨‍💻 Author

**Yugang Gaurav**

GitHub: [YugangGaurav](https://github.com/YugangGaurav)

## 📄 License

This project is intended for educational and research purposes.
