# Naive Bayes Review Sentiment Classification

A binary text-classification project that predicts whether a customer review is positive (`Liked = 1`) or negative (`Liked = 0`).

## Dataset overview

- Rows: 1000
- Columns: `Review`, `Liked`
- Positive reviews (1): 500
- Negative reviews (0): 500
- Missing values in the two required columns after cleaning: 0

## Project files

- `reviews.csv` – dataset
- `train_model.py` – preprocessing, model comparison, evaluation and artifact saving
- `app.py` – Streamlit user interface
- `text_preprocessing.py` – shared text-cleaning function used by training and deployment
- `model.pkl` and `vectorizer.pkl` – trained best model and matching vectorizer
- `model_name.txt` – selected model name
- `evaluation_results.json` – metrics and classification reports
- `requirements.txt` – pinned Python dependencies for reproducible model loading

## Text preprocessing

Text is lowercased, punctuation and digits are removed, English stopwords are removed, and words are stemmed with NLTK's Porter Stemmer. The vectorizer performs this preprocessing consistently for training and new reviews.

## Models compared

1. Bernoulli Naive Bayes with binary Bag of Words features
2. Multinomial Naive Bayes with Bag of Words features
3. Multinomial Naive Bayes with TF-IDF features

The best model is selected by F1 score, with accuracy used as a tie-breaker. Evaluation uses an 80/20 stratified train/test split and random state 42.

## Run locally

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and evaluate
python train_model.py

# 4. Launch the web app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload all project files.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy. Keep `app.py`, `model.pkl`, `vectorizer.pkl`, and `model_name.txt` in the repository root.

## Important

The model is a learning/demo project trained on a relatively small dataset of 1,000 reviews. Its predictions may not generalize to every product, service, or writing style. Do not treat the confidence score as certainty.
