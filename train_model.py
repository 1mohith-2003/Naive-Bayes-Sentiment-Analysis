"""Train and compare Naive Bayes sentiment classifiers for customer reviews."""
import json
import re
import pickle
import pandas as pd
from text_preprocessing import clean_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = "reviews.csv"
RANDOM_STATE = 42
def main():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["Review", "Liked"]).copy()
    data["Review"] = data["Review"].astype(str)
    data["Liked"] = data["Liked"].astype(int)

    print("Dataset shape:", data.shape)
    print("\nFirst 10 rows:\n", data.head(10))
    print("\nClass counts:\n", data["Liked"].value_counts().sort_index())
    print("\nMissing values:\n", data.isna().sum())
    print("\nReview length summary:\n", data["Review"].str.len().describe())

    X_train, X_test, y_train, y_test = train_test_split(
        data["Review"], data["Liked"], test_size=0.20,
        random_state=RANDOM_STATE, stratify=data["Liked"]
    )

    candidates = {
        "BernoulliNB + Binary CountVectorizer": (
            CountVectorizer(preprocessor=clean_text, binary=True, ngram_range=(1, 2)),
            BernoulliNB(alpha=1.0)
        ),
        "MultinomialNB + CountVectorizer": (
            CountVectorizer(preprocessor=clean_text, ngram_range=(1, 2)),
            MultinomialNB(alpha=1.0)
        ),
        "MultinomialNB + TF-IDF": (
            TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True),
            MultinomialNB(alpha=0.5)
        )
    }

    all_results = []
    fitted = {}
    for name, (vectorizer, classifier) in candidates.items():
        train_features = vectorizer.fit_transform(X_train)
        test_features = vectorizer.transform(X_test)
        classifier.fit(train_features, y_train)
        predictions = classifier.predict(test_features)

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
            "Classification Report": classification_report(
                y_test, predictions, target_names=["Negative", "Positive"],
                zero_division=0
            )
        }
        all_results.append(result)
        fitted[name] = (vectorizer, classifier)

        print(f"\n===== {name} =====")
        for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            print(f"{metric}: {result[metric]:.4f}")
        print("Confusion matrix:\n", result["Confusion Matrix"])
        print(result["Classification Report"])

    best = max(all_results, key=lambda item: (item["F1 Score"], item["Accuracy"]))
    best_vectorizer, best_classifier = fitted[best["Model"]]

    with open("model.pkl", "wb") as file:
        pickle.dump(best_classifier, file)
    with open("vectorizer.pkl", "wb") as file:
        pickle.dump(best_vectorizer, file)
    with open("model_name.txt", "w") as file:
        file.write(best["Model"])
    with open("evaluation_results.json", "w") as file:
        json.dump(all_results, file, indent=2)

    print("\nBest model:", best["Model"])
    print("Saved model.pkl, vectorizer.pkl, model_name.txt and evaluation_results.json")

if __name__ == "__main__":
    main()
