"""
Spam Spoiler - LSTM/GRU model (matches the algorithm named in the report)
--------------------------------------------------------------------------
This reproduces the "LSTM based GRU" classifier described in the project:
tokenize -> pad sequences -> embedding -> LSTM/GRU stack -> dense layers
-> softmax over the four classes (Normal, Fraudulent, Threatening,
Suspicious).

Not used by the Flask app by default (it needs TensorFlow installed and a
lot more labelled data than the ~90 demo rows in emails.csv to be accurate)
but it's here so the project structurally matches what the slides describe.
To use it instead of the TF-IDF model, train it, then point app.py's
classify_email() at this model + its tokenizer instead of classifier.pkl.

Install first:  pip install tensorflow --break-system-packages
Run:            python train_lstm_gru.py
"""
import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "emails.csv")
MAX_WORDS = 3000
MAX_LEN = 120


def main():
    df = pd.read_csv(DATA_PATH)
    labels = sorted(df["label"].unique())
    label_to_id = {label: i for i, label in enumerate(labels)}

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    sequences = tokenizer.texts_to_sequences(df["text"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = to_categorical(df["label"].map(label_to_id), num_classes=len(labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # LSTM based GRU: an LSTM layer feeding a GRU layer, then dense layers
    # with batch normalization, as described in the "Classifier Module"
    # section of the report.
    model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        LSTM(64, return_sequences=True),
        GRU(32),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=8)

    model.save(os.path.join(HERE, "lstm_gru_model.h5"))
    with open(os.path.join(HERE, "tokenizer.json"), "w") as f:
        f.write(tokenizer.to_json())
    with open(os.path.join(HERE, "labels.json"), "w") as f:
        json.dump(labels, f)
    print("Saved lstm_gru_model.h5, tokenizer.json, labels.json")


if __name__ == "__main__":
    main()
