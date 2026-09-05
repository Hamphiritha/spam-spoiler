> Note: free hosting sleeps after inactivity — the first load may take ~30–50 seconds to wake up.
> click here : [https://spam-spoiler.onrender.com](https://spam-spoiler.onrender.com)

# Spam Spoiler — E-mail Forensics Web App

A recreation of the "Spam Spoiler" project from the presentation: a Flask
web app that classifies e-mail into **Normal, Fraudulent, Threatening or
Suspicious**, with the same page flow as the slides (Home → User login →
set Gmail credentials → Spam Detection; Admin → Training phase → Feature
extraction).

## Setup

```bash
cd spam_spoiler
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt --break-system-packages
python app.py
```

Open **http://localhost:5000**. The classifier model files
(`model/vectorizer.pkl`, `model/classifier.pkl`) are already trained and
included, so the app works immediately — no extra setup needed.

## Pages (matches the report's screenshots)

| Route | Page |
|---|---|
| `/` | Home / dashboard hero |
| `/register`, `/login_user` | User sign-up and login |
| `/userhome` | "User: <name>" + set Gmail button |
| `/setting` | Set Gmail ID & app password |
| `/train_data` | Admin training phase (sample labelled e-mails) |
| `/process3` | Feature extraction (TF-IDF table) |
| `/spam_detect` | Inbox table with per-message classification |

## What's real vs. simulated

- **Real**: user accounts (SQLite + hashed passwords), the TF-IDF +
  Logistic Regression classifier, the four-class labelling, IMAP fetch
  code that will pull your actual Gmail inbox if you provide real
  credentials in Settings.
- **Simulated for the demo**: the training/feature-extraction pages run
  against a small (~90-row) hand-written dataset in `model/emails.csv`,
  not the full Enron corpus used in the original research — swap that CSV
  for a bigger labelled set and rerun `model/train_model.py` to improve
  accuracy.
- **Gmail login**: to fetch a *real* inbox, generate a Google **App
  Password** (Gmail no longer allows plain-password IMAP) at
  https://myaccount.google.com/apppasswords and enter it in Settings. If
  no credentials are set, or the IMAP login fails, the Spam Detection page
  falls back to a demo inbox so the page always has something to show.
- **Deep learning**: `model/train_lstm_gru.py` reproduces the report's
  actual "LSTM based GRU" architecture in Keras/TensorFlow (not wired into
  the Flask app by default — install `tensorflow` and swap it in if you
  want the real deep-learning version instead of the fast TF-IDF model).

## Turn it into a desktop app

Instead of running `python app.py` and opening a browser tab, you can run it
as its own window (no address bar, no "localhost:5000" typing):

```bash
pip install -r requirements-desktop.txt
python desktop_app.py
```

This opens Spam Spoiler in a native window using pywebview. Close the window
to quit — no need to Ctrl+C a terminal.

### Build a real Windows .exe (double-click to launch, no Python needed)

Once `desktop_app.py` works, package it with PyInstaller:

```bash
pyinstaller --onefile --name "SpamSpoiler" ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "model;model" ^
  desktop_app.py
```

(On macOS/Linux, replace the `;` in `--add-data` with `:`.)

This creates `dist\SpamSpoiler.exe`. Double-click it and the app opens in its
own window — you can copy that one file to another Windows computer without
installing Python there at all. First launch may take a few seconds while it
unpacks.

## Project structure

```
spam_spoiler/
├── app.py                  # Flask routes
├── requirements.txt
├── model/
│   ├── emails.csv          # demo labelled dataset
│   ├── train_model.py      # trains the TF-IDF + LogisticRegression model (used by app)
│   ├── train_lstm_gru.py   # trains the report's actual LSTM/GRU network (optional)
│   ├── vectorizer.pkl      # pre-trained (included)
│   └── classifier.pkl      # pre-trained (included)
├── templates/               # Jinja HTML pages
├── static/style.css
└── instance/spam_spoiler.db # created on first run
```
