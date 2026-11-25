import sqlite3
from backend.config import DB_PATH


def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "track_id INTEGER, zone_id TEXT NOT NULL, zone_name TEXT NOT NULL, "
            "severity TEXT NOT NULL, screenshot TEXT, "
            "object_type TEXT NOT NULL DEFAULT 'person', "
            "class_name TEXT NOT NULL DEFAULT 'Person')"
        )
        # Backward-compatible migration for databases created by the earlier version.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "object_type" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN object_type TEXT NOT NULL DEFAULT 'person'")
        if "class_name" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN class_name TEXT NOT NULL DEFAULT 'Person'")
        conn.commit()


def add_event(timestamp, track_id, zone, screenshot, object_type="person", class_name="Person"):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO events(timestamp,track_id,zone_id,zone_name,severity,screenshot,object_type,class_name) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (timestamp, track_id, zone["id"], zone["name"], zone["severity"], screenshot, object_type, class_name),
        )
        conn.commit()
        return cur.lastrowid


def get_events(limit=100):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()]
