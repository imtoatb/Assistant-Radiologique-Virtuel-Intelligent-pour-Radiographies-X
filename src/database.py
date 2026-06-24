from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "medical_ai_evidence.sqlite"
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "abnormal_lung_cases.csv"


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    """Crée les tables depuis schema.sql si elles n'existent pas encore."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


def seed_radios(
    db_path: str | Path = DEFAULT_DB,
    csv_path: str | Path = DEFAULT_CSV,
) -> int:
    """Charge le CSV des radiographies dans la table radios.
    Ignore les lignes déjà présentes (INSERT OR IGNORE).
    Retourne le nombre de lignes vraiment insérées.
    """
    init_db(db_path)
    conn = connect(db_path)
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO radios(id, image_path, source, ground_truth_label, split, notes)
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


def insert_run(
    db_path: str | Path,
    case_id: str,
    image_path: str,
    prediction: dict,
) -> int:
    """Log une inférence du modèle dans la table runs.
    Retourne l'id du run (utilisé ensuite par insert_evaluation).
    """
    init_db(db_path)
    conn = connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO runs(case_id, image_path, model_name, prompt_version,
                         prediction_json, predicted_class, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            image_path,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0)),
            int(prediction.get("latency_ms", 0)),
        ),
    )
    run_id: int = cursor.lastrowid  # type: ignore[assignment]
    conn.commit()
    conn.close()
    return run_id


def insert_evaluation(
    db_path: str | Path,
    run_id: int,
    ground_truth: str,
    predicted: str,
    comment: str = "",
) -> None:
    """Log si le modèle s'est trompé ou pas sur ce run."""
    init_db(db_path)
    correct = 1 if predicted == ground_truth else 0
    error_type: str | None = None
    if not correct:
        error_type = f"predicted_{predicted}_expected_{ground_truth}"
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO evaluations(run_id, ground_truth_label, correct, error_type, reviewer_comment)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, ground_truth, correct, error_type, comment),
    )
    conn.commit()
    conn.close()


def get_runs(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne tous les runs loggés, du plus récent au plus ancien."""
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_evaluations(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne toutes les évaluations avec les infos du run associé."""
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT e.*, r.predicted_class, r.case_id, r.model_name
        FROM evaluations e
        JOIN runs r ON e.run_id = r.id
        ORDER BY e.created_at DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_radios(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne toutes les radiographies du catalogue."""
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM radios").fetchall()
    conn.close()
    return [dict(r) for r in rows]
