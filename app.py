"""
Spam Spoiler - Flask app
------------------------
Recreates the web flow from the project report:
Home -> User (register/login) -> Set Gmail creds -> Spam Detection
                Admin -> Training phase -> Feature extraction (TF-IDF)

Run:
    pip install -r requirements.txt --break-system-packages
    python app.py
Then open http://localhost:5000
"""
import os
import sqlite3
import imaplib
import email
from email.header import decode_header
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "spam_spoiler.db")
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# ---------------------------------------------------------------- database
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            gmail_id TEXT,
            gmail_app_password TEXT
        )
    """)
    conn.commit()
    conn.close()


# ------------------------------------------------------------- classifier
_vectorizer = None
_classifier = None


def load_classifier():
    global _vectorizer, _classifier
    if _vectorizer is None:
        _vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        _classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
    return _vectorizer, _classifier


def classify_email(text: str) -> str:
    vectorizer, clf = load_classifier()
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


# Demo inbox shown when there is no live Gmail connection yet, so the
# Spam Detection page always has something to display (mirrors the
# sample inbox shown in the report's screenshots).
DEMO_INBOX = [
    {"from": "Jes Re <jerslinkathy2001@gmail.com>", "subject": "test",
     "message": "SO BE INFORMED THAT THE AMOUNT TO BE PAID NOW", "date": "2023-04-03 13:46:53"},
    {"from": "Nisha Priya <2001nishapriya@gmail.com>", "subject": "test",
     "message": "I AM A GIRL OF 25 YEARS AND I AM LOOKING FOR INVESTMENT OPPORTUNITY", "date": "2023-04-03 13:26:51"},
    {"from": "Nisha Priya <2001nishapriya@gmail.com>", "subject": "test",
     "message": "As soon as your information is received by my attorney", "date": "2023-04-03 13:25:24"},
    {"from": "Nisha Priya <2001nishapriya@gmail.com>", "subject": "test",
     "message": "I AM THE SON OF THE LATE PRESIDENT", "date": "2023-04-03 13:18:19"},
    {"from": "HR Team <hr@example.com>", "subject": "Meeting notes",
     "message": "Attached is the invoice for last month's consulting hours.", "date": "2023-04-03 09:02:11"},
]


def fetch_gmail_inbox(gmail_id, app_password, limit=15):
    """Fetch recent inbox messages over IMAP. Returns None on failure so the
    caller can fall back to demo data instead of crashing the page."""
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(gmail_id, app_password)
        imap.select("inbox")
        status, data = imap.search(None, "ALL")
        if status != "OK":
            return None
        ids = data[0].split()[-limit:]
        messages = []
        for msg_id in reversed(ids):
            _, msg_data = imap.fetch(msg_id, "(RFC822)")
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            subject, enc = decode_header(msg.get("Subject", ""))[0]
            if isinstance(subject, bytes):
                subject = subject.decode(enc or "utf-8", errors="ignore")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            messages.append({
                "from": msg.get("From", ""),
                "subject": subject or "(no subject)",
                "message": body[:200],
                "date": msg.get("Date", ""),
            })
        imap.logout()
        return messages
    except Exception:
        return None


# --------------------------------------------------------------- helpers
def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login_user"))
        return view(*args, **kwargs)
    return wrapped


# ----------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or not password:
            flash("Please fill in both fields.")
            return redirect(url_for("register"))
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )
            conn.commit()
            flash("Account created, please log in.")
            return redirect(url_for("login_user"))
        except sqlite3.IntegrityError:
            flash("That username is already taken.")
            return redirect(url_for("register"))
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/login_user", methods=["GET", "POST"])
def login_user():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            return redirect(url_for("userhome"))
        flash("Invalid username or password.")
        return redirect(url_for("login_user"))
    return render_template("login_user.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/userhome")
@login_required
def userhome():
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (session["username"],)).fetchone()
    conn.close()
    return render_template("userhome.html", user=user)


@app.route("/setting", methods=["GET", "POST"])
@login_required
def setting():
    if request.method == "POST":
        gmail_id = request.form["gmail_id"].strip()
        gmail_password = request.form["gmail_password"]
        conn = get_db()
        conn.execute(
            "UPDATE users SET gmail_id = ?, gmail_app_password = ? WHERE username = ?",
            (gmail_id, gmail_password, session["username"]),
        )
        conn.commit()
        conn.close()
        flash("Gmail credentials saved.")
        return redirect(url_for("userhome"))
    return render_template("setting.html")


@app.route("/spam_detect")
@login_required
def spam_detect():
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (session["username"],)).fetchone()
    conn.close()

    inbox = None
    live = False
    if user["gmail_id"] and user["gmail_app_password"]:
        inbox = fetch_gmail_inbox(user["gmail_id"], user["gmail_app_password"])
        live = inbox is not None

    if inbox is None:
        inbox = DEMO_INBOX

    results = []
    unread = 0
    for i, mail in enumerate(inbox, start=1):
        label = classify_email(mail["message"])
        if label != "normal":
            unread += 1
        results.append({**mail, "sno": i, "label": label})

    return render_template("spam_detect.html", emails=results, unread=unread, live=live)


@app.route("/train_data")
def train_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "emails.csv"))
    sample = df.sample(min(8, len(df)), random_state=1).to_dict("records")
    return render_template("train_data.html", rows=sample)


@app.route("/process3")
def process3():
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_csv(os.path.join(MODEL_DIR, "emails.csv"))
    cols = ["word_freq_make", "word_freq_address", "word_freq_all",
            "word_freq_3d", "word_freq_free", "word_freq_money"]
    vocab = [c.replace("word_freq_", "") for c in cols]
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    matrix = vectorizer.fit_transform(df["text"]).toarray()
    table = pd.DataFrame(matrix, columns=cols).round(2)
    describe = table.describe().round(4)
    return render_template(
        "process3.html",
        table_rows=table.head(6).to_dict("records"),
        cols=cols,
        describe_rows=describe.reset_index().to_dict("records"),
    )


# Run once at import time so the database exists whether this file is
# started directly (python app.py) or loaded by a production server like
# gunicorn (which never executes the __main__ block below).
init_db()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
