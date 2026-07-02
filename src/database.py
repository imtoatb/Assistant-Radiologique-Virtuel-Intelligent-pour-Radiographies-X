from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _cases_id_is_integer(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "cases"):
        return False
    for column in conn.execute("PRAGMA table_info(cases)").fetchall():
        if column["name"] == "id":
            return "INT" in str(column["type"]).upper()
    return False


def _migrate_case_ids_to_text(conn: sqlite3.Connection) -> None:
    """Migrate old SQLite databases from numeric case IDs to text case IDs."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.executescript(
        """
        ALTER TABLE cases RENAME TO cases_old;
        ALTER TABLE prompts RENAME TO prompts_old;
        ALTER TABLE runs RENAME TO runs_old;
        ALTER TABLE evaluations RENAME TO evaluations_old;

        DROP INDEX IF EXISTS uq_prompts_name_version;
        DROP INDEX IF EXISTS uq_runs_case_prompt;
        """
    )
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        """
        INSERT INTO cases (id, image_path, source, ground_truth_label, split, notes)
        SELECT CAST(id AS TEXT), image_path, source, ground_truth_label, split, notes
        FROM cases_old
        """
    )
    conn.execute(
        """
        INSERT INTO prompts (id, prompt_name, prompt_version, prompt_text, created_at)
        SELECT id, prompt_name, prompt_version, prompt_text, created_at
        FROM prompts_old
        """
    )
    conn.execute(
        """
        INSERT INTO runs (
            id, case_id, prompt_id, image_path, model_name,
            prediction_json, predicted_class, confidence, latency_ms, created_at
        )
        SELECT
            id, CAST(case_id AS TEXT), prompt_id, image_path, model_name,
            prediction_json, predicted_class, confidence, latency_ms, created_at
        FROM runs_old
        """
    )
    conn.execute(
        """
        INSERT INTO evaluations (id, run_id, ground_truth_label, correct, error_type, created_at)
        SELECT id, run_id, ground_truth_label, correct, error_type, created_at
        FROM evaluations_old
        """
    )
    conn.executescript(
        """
        DROP TABLE evaluations_old;
        DROP TABLE runs_old;
        DROP TABLE prompts_old;
        DROP TABLE cases_old;
        """
    )
    conn.execute("PRAGMA foreign_keys = ON")


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    if _cases_id_is_integer(conn):
        _migrate_case_ids_to_text(conn)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


def seed_cases(db_path: str | Path = DEFAULT_DB, csv_path: str | Path = DEFAULT_DB) -> int:
    init_db(db_path)
    conn = connect(db_path)
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO cases (id, image_path, source, ground_truth_label, split, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["case_id"],
                    row["image_path"],
                    row["source"],
                    row["label"],
                    row["split"],
                    row.get("notes", ""),
                ),
            )
            inserted += cursor.rowcount
    conn.commit()
    conn.close()
    return inserted


def insert_prompt(
    db_path: str | Path = DEFAULT_DB,
    prompt_name: str = "",
    prompt_version: str = "",
    prompt_text: str = "",
) -> int:
    init_db(db_path)
    conn = connect(db_path)
    existing = conn.execute(
        "SELECT id FROM prompts WHERE prompt_name = ? AND prompt_version = ?",
        (prompt_name, prompt_version),
    ).fetchone()
    if existing:
        conn.close()
        return int(existing["id"])

    cursor = conn.execute(
        """
        INSERT INTO prompts (prompt_name, prompt_version, prompt_text)
        VALUES (?, ?, ?)
        """,
        (prompt_name, prompt_version, prompt_text),
    )
    prompt_id = int(cursor.lastrowid)
    conn.commit()
    conn.close()
    return prompt_id


def insert_run(db_path: str | Path, case_id: str, image_path: str, prediction: dict[str, Any], prompt_id: int) -> int:
    init_db(db_path)
    conn = connect(db_path)

    values = (
        str(case_id),
        prompt_id,
        image_path,
        prediction.get("model_name"),
        json.dumps(prediction, ensure_ascii=False),
        prediction.get("predicted_class"),
        float(prediction.get("confidence", 0.0)),
        int(prediction.get("latency_ms", 0)),
    )

    conn.execute(
        """
        INSERT INTO runs (
            case_id, prompt_id, image_path, model_name,
            prediction_json, predicted_class, confidence, latency_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(case_id, prompt_id) DO UPDATE SET
            image_path = excluded.image_path,
            model_name = excluded.model_name,
            prediction_json = excluded.prediction_json,
            predicted_class = excluded.predicted_class,
            confidence = excluded.confidence,
            latency_ms = excluded.latency_ms,
            created_at = CURRENT_TIMESTAMP
        """,
        values,
    )

    run_id = conn.execute(
        "SELECT id FROM runs WHERE case_id = ? AND prompt_id = ?",
        (str(case_id), prompt_id),
    ).fetchone()["id"]

    conn.commit()
    conn.close()
    return int(run_id)


def insert_evaluation(db_path: str | Path, run_id: int, ground_truth: str, predicted: str) -> None:
    correct = int(predicted == ground_truth)
    error_type = None if correct else f"predicted_{predicted}_expected_{ground_truth}"

    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO evaluations (
            run_id, ground_truth_label, correct, error_type
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(run_id) DO UPDATE SET
            ground_truth_label = excluded.ground_truth_label,
            correct = excluded.correct,
            error_type = excluded.error_type,
            created_at = CURRENT_TIMESTAMP
        """,
        (run_id, ground_truth, correct, error_type),
    )
    conn.commit()
    conn.close()


def get_runs(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_cases(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM cases").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prompts(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM prompts ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_evaluations(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT
            r.case_id,
            r.id AS run_id,
            e.id AS evaluation_id,
            p.id AS prompt_id,
            p.prompt_name,
            p.prompt_version,
            e.ground_truth_label,
            r.predicted_class,
            r.confidence,
            r.latency_ms AS latency
        FROM runs r
        JOIN evaluations e ON e.run_id = r.id
        JOIN prompts p ON p.id = r.prompt_id
        ORDER BY p.prompt_name, p.prompt_version, r.case_id
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
