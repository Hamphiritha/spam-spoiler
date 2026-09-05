"""
Spam Spoiler - Desktop launcher
--------------------------------
Runs the Flask app in a background thread and opens it in its own native
window using pywebview, so it behaves like a normal desktop app instead of
something you open in a browser tab.

Run it directly:
    python desktop_app.py

Or build a real .exe out of it - see the "Build a Windows .exe" section in
README.md.
"""
import threading
import time
import webview

from app import app, init_db

HOST = "127.0.0.1"
PORT = 5000


def run_flask():
    init_db()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    time.sleep(1.2)  # give Flask a moment to start before the window loads it

    webview.create_window(
        "Spam Spoiler",
        f"http://{HOST}:{PORT}",
        width=1200,
        height=800,
        min_size=(900, 600),
    )
    webview.start()
