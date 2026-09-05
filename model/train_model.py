"""
Spam Spoiler - model training script
-------------------------------------
Trains the e-mail classifier used by the web app.

The project report (Email_Final_PPT.pdf) specifies an LSTM-based GRU deep
learning model over four classes: Normal, Fraudulent, Threatening and
Suspicious. A GRU/LSTM needs a large labelled corpus and real training time
to be worth using, so this script ships two things:

1. A lightweight TF-IDF + Logistic Regression classifier (fast, deterministic,
   trains in under a second) that the Flask app uses by default so the demo
   works out of the box.
2. train_lstm_gru.py (same folder) - the actual Keras LSTM/GRU network
   described in the report, trainable on the same labelled data, for anyone
   who wants to reproduce that part of the project for real.

Both scripts train on emails.csv in this folder. Replace/extend that file
with your own labelled data (columns: text,label) to improve real-world
accuracy - the bundled rows are illustrative examples, not a research-grade
corpus.
"""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "emails.csv")
VEC_PATH = os.path.join(HERE, "vectorizer.pkl")
MODEL_PATH = os.path.join(HERE, "classifier.pkl")


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    print("Held-out evaluation:")
    print(classification_report(y_test, clf.predict(X_test_vec)))

    joblib.dump(vectorizer, VEC_PATH)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved vectorizer -> {VEC_PATH}")
    print(f"Saved classifier -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
