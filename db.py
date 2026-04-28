import sqlite3
import os
import json
from datetime import datetime

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "users.db")

os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)


# ==============================
# INIT DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    # RESULTS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        prediction TEXT,
        score INTEGER,
        type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # SAFELY ADD detailed_data COLUMN IF IT DOES NOT EXIST
    try:
        cursor.execute("ALTER TABLE results ADD COLUMN detailed_data TEXT")
    except sqlite3.OperationalError:
        pass  # Column likely already exists

    conn.commit()
    conn.close()


# ==============================
# USER HELPERS
# ==============================
def create_user(email, hashed_password):
    """Insert a new user. Raises sqlite3.IntegrityError if email already exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, hashed_password)
    )
    conn.commit()
    conn.close()


def get_user_by_email(email):
    """Return a sqlite3.Row for the given email, or None if not found."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user


# ==============================
# RESULT HELPERS
# ==============================
def save_result(user_email, prediction, score, result_type, detailed_data=None):
    """Insert a new assessment result for the given user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    data_str = json.dumps(detailed_data) if detailed_data else None
    
    cursor.execute("""
        INSERT INTO results (user_email, prediction, score, type, detailed_data)
        VALUES (?, ?, ?, ?, ?)
    """, (user_email, prediction, score, result_type, data_str))
    conn.commit()
    conn.close()


def get_results_by_email(user_email):
    """Return a list of result dicts for the given user, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prediction, score, type, created_at, detailed_data
        FROM results
        WHERE user_email = ?
        ORDER BY created_at DESC
    """, (user_email,))
    rows = cursor.fetchall()
    conn.close()

    results_list = []
    for row in rows:
        data_json = None
        if "detailed_data" in row.keys() and row["detailed_data"]:
            try:
                data_json = json.loads(row["detailed_data"])
            except:
                pass
                
        results_list.append({
            "prediction": row["prediction"],
            "score":      row["score"],
            "type":       row["type"],
            "date":       _fmt_date(row["created_at"]),
            "detailed_data": data_json
        })

    return results_list


def _fmt_date(raw):
    """Turn '2026-04-26 03:04:12' into '26 Apr 2026, 03:04'."""
    if not raw:
        return ""
    try:
        dt = datetime.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S")
        # %#d strips leading zero on Windows; %-d on Linux/Mac
        import sys
        day_fmt = "%#d" if sys.platform == "win32" else "%-d"
        return dt.strftime(f"{day_fmt} %b %Y, %H:%M")
    except Exception:
        return str(raw)
