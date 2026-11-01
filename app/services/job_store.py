from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock
from uuid import UUID

from app.schemas.jobs import JobResponse
from app.core.config import settings


class PersistentJobStore:
    """Small SQLite-backed job metadata store for durable API state.

    This persists job records, not processing execution. A separate worker and
    lease/retry protocol are still required for production job orchestration.
    """

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def __setitem__(self, job_id: UUID, job: JobResponse) -> None:
        payload = job.model_dump_json()
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO jobs (id, payload, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (str(job_id), payload),
            )
            connection.commit()

    def get(self, job_id: UUID, default=None):
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM jobs WHERE id = ?", (str(job_id),)
            ).fetchone()
        if row is None:
            return default
        return JobResponse.model_validate_json(row[0])

    def __len__(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return int(row[0])


job_store = PersistentJobStore(settings.job_db_path)
