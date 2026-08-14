"""
SQLite-backed store for the face recognition system: embeddings, per-camera
desired state (enabled/recognition), recognition events (sightings), an index of
deliberately-taken attendance photos, and key-value settings.

For embeddings it replaces the pickle index: writes are atomic transactions,
individual people can be listed/deleted, and a version counter lets a
long-running recognizer detect new enrollments and hot-reload without a
restart. The version counter tracks the face index ONLY — camera config,
events, and settings do not bump it. WAL journal mode allows the enroll and
recognize processes to share the file concurrently.
"""
import pickle
import sqlite3
import threading
from pathlib import Path

import numpy as np

from src.config import config

EMB_DIM = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY,
    person_id  INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('version', '0');
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY,
    enabled     INTEGER NOT NULL DEFAULT 1,
    recognition INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    person     TEXT NOT NULL,
    camera     INTEGER NOT NULL,
    similarity REAL NOT NULL,
    snapshot   BLOB,
    liveness   REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- Deliberately taken attendance photos. Separate from `events` because events
-- are pruned to the newest 500 on every insert: an attendance record must not
-- be evicted by ordinary sightings. The image itself is a file on disk, not a
-- blob — `path` is relative to the capture root so the folder can be moved.
CREATE TABLE IF NOT EXISTS captures (
    id         INTEGER PRIMARY KEY,
    person     TEXT NOT NULL,
    camera     INTEGER NOT NULL,
    path       TEXT NOT NULL,
    similarity REAL,
    liveness   REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class FaceStore:
    def __init__(self, db_path=None):
        db_path = config.paths.database if db_path is None else db_path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + our own lock: the web service touches the
        # store from both its processing thread and request handler threads.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        # events tables from before the anti-spoofing feature lack `liveness`
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "liveness" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN liveness REAL")
        self.conn.commit()

    def version(self):
        """Monotonic counter bumped on every write — cheap to poll for changes."""
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        return int(row[0])

    def _bump_version(self):
        self.conn.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='version'")

    def add_embedding(self, name, vector):
        vec = np.asarray(vector, dtype=np.float32).reshape(EMB_DIM)
        with self._lock, self.conn:  # one atomic transaction
            self.conn.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
            (person_id,) = self.conn.execute(
                "SELECT id FROM people WHERE name = ?", (name,)).fetchone()
            self.conn.execute(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                (person_id, vec.tobytes()))
            self._bump_version()

    def load_all(self):
        """Returns (labels, embeddings): parallel list and (N, 512) float32 array."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT p.name, e.vector FROM embeddings e "
                "JOIN people p ON p.id = e.person_id ORDER BY e.id").fetchall()
        labels = [name for name, _ in rows]
        if rows:
            embeddings = np.frombuffer(
                b"".join(vec for _, vec in rows), dtype=np.float32
            ).reshape(len(rows), EMB_DIM).copy()
        else:
            embeddings = np.zeros((0, EMB_DIM), dtype=np.float32)
        return labels, embeddings

    def count(self):
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def list_people(self):
        """Returns [(name, n_samples), ...] sorted by name."""
        with self._lock:
            return self.conn.execute(
                "SELECT p.name, COUNT(e.id) FROM people p "
                "LEFT JOIN embeddings e ON e.person_id = p.id "
                "GROUP BY p.id ORDER BY p.name").fetchall()

    def delete_person(self, name):
        """Remove a person and all their embeddings. Returns True if they existed."""
        with self._lock, self.conn:
            cur = self.conn.execute("DELETE FROM people WHERE name = ?", (name,))
            if cur.rowcount:
                self._bump_version()
        return cur.rowcount > 0

    # -- camera desired state ------------------------------------------------

    def camera_config(self, cam_id):
        """(enabled, recognition) for a camera; creates a default row on first ask."""
        with self._lock:
            row = self.conn.execute(
                "SELECT enabled, recognition FROM cameras WHERE id = ?", (cam_id,)).fetchone()
            if row is None:
                with self.conn:
                    self.conn.execute("INSERT OR IGNORE INTO cameras (id) VALUES (?)", (cam_id,))
                return True, True
            return bool(row[0]), bool(row[1])

    def set_camera_config(self, cam_id, enabled=None, recognition=None):
        with self._lock, self.conn:
            self.conn.execute("INSERT OR IGNORE INTO cameras (id) VALUES (?)", (cam_id,))
            if enabled is not None:
                self.conn.execute("UPDATE cameras SET enabled = ? WHERE id = ?",
                                  (int(enabled), cam_id))
            if recognition is not None:
                self.conn.execute("UPDATE cameras SET recognition = ? WHERE id = ?",
                                  (int(recognition), cam_id))

    # -- key-value settings --------------------------------------------------

    def get_setting(self, key, default=None):
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_setting(self, key, value):
        with self._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                              (key, str(value)))

    # -- recognition events --------------------------------------------------

    def add_event(self, person, camera, similarity, snapshot=None, liveness=None,
                  keep=None):
        """Record a sighting; prunes the table to the most recent `keep` rows.
        `liveness` is P(live face) from the anti-spoofing model, None if the
        check didn't run."""
        keep = config.events.keep if keep is None else keep
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO events (person, camera, similarity, snapshot, liveness) "
                "VALUES (?, ?, ?, ?, ?)",
                (person, int(camera), float(similarity), snapshot,
                 None if liveness is None else float(liveness)))
            self.conn.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY id DESC LIMIT ?)", (keep,))

    def list_events(self, limit=50):
        """Most recent sightings, newest first, without the snapshot blobs."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, person, camera, similarity, liveness, created_at FROM events "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "person": r[1], "camera": r[2], "similarity": r[3],
                 "liveness": r[4], "created_at": r[5]} for r in rows]

    def event_snapshot(self, event_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT snapshot FROM events WHERE id = ?", (event_id,)).fetchone()
        return row[0] if row else None

    # -- deliberate attendance captures ---------------------------------------

    def add_capture(self, person, camera, path, similarity=None, liveness=None):
        """Index a photo already written to disk. Never pruned, and deliberately
        does NOT bump the version counter — that tracks the face index only, and
        bumping it here would make every recognizer reload on each photo."""
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO captures (person, camera, path, similarity, liveness) "
                "VALUES (?, ?, ?, ?, ?)",
                (person, int(camera), str(path),
                 None if similarity is None else float(similarity),
                 None if liveness is None else float(liveness)))
        return cur.lastrowid

    def list_captures(self, limit=50):
        """Most recent attendance photos, newest first."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, person, camera, path, similarity, liveness, created_at "
                "FROM captures ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "person": r[1], "camera": r[2], "path": r[3],
                 "similarity": r[4], "liveness": r[5], "created_at": r[6]}
                for r in rows]

    def capture_path(self, capture_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT path FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return row[0] if row else None

    def migrate_from_pickle(self, pkl_path):
        """One-time import of the legacy pickle index. Returns rows imported."""
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        labels = data["labels"]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        with self._lock, self.conn:
            for name, vec in zip(labels, embeddings):
                self.conn.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
                (person_id,) = self.conn.execute(
                    "SELECT id FROM people WHERE name = ?", (name,)).fetchone()
                self.conn.execute(
                    "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                    (person_id, vec.tobytes()))
            self._bump_version()
        return len(labels)
