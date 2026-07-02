from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "database.sqlite"

# connecte à la base de données SQLite et retourne un objet Connection
def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON") # active les clées étrangères
    return conn


def init_db(db_path: str | Path = DEFAULT_DB) -> None:
    """Crée les tables depuis schema.sql si elles n'existent pas encore """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()

def seed_cases( db_path: str | Path = DEFAULT_DB, csv_path: str | Path = DEFAULT_DB) -> int:
    """Charge le CSV des radiographies dans la table cases
    Ignore les lignes déjà présentes (INSERT OR IGNORE)
    Retourne le nombre de lignes vraiment insérées
    """
    init_db(db_path)
    conn = connect(db_path)
    inserted = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO cases (id, image_path, source, ground_truth_label, split, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["case_id"]),
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


def insert_case(db_path: str | Path = DEFAULT_DB, image_path: str = "", source: str = "upload", ground_truth_label: str | None = None, split: str = "upload", notes: str = "") -> int:
    """Insère une nouvelle radiographie (ex: upload utilisateur) dans la table cases
    Retourne l'id de la case créée (utilisé ensuite par insert_run)
    """
    init_db(db_path)
    conn = connect(db_path)
    cursor = conn.execute(
        """
        INSERT INTO cases (image_path, source, ground_truth_label, split, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (image_path, source, ground_truth_label, split, notes),
    )
    case_id: int = cursor.lastrowid
    conn.commit()
    conn.close()
    return case_id


def insert_prompt( db_path: str | Path = DEFAULT_DB, prompt_name: str = "", prompt_version: str = "", prompt_text: str = "") -> int:
    """Insère un prompt dans la table prompts si il n'existe pas déjà
    Retourne l'id du prompt (utilisé ensuite par insert_run)
    """
    init_db(db_path)
    conn = connect(db_path)
    existing = conn.execute(
        "SELECT id FROM prompts WHERE prompt_name = ? AND prompt_version = ?",
        (prompt_name, prompt_version),
    ).fetchone()
    if existing:
        conn.close()
        return existing["id"]
    cursor = conn.execute(
        """
        INSERT INTO prompts (prompt_name, prompt_version, prompt_text)
        VALUES (?, ?, ?)
        """,
        (prompt_name, prompt_version, prompt_text),
    )
    prompt_id: int = cursor.lastrowid
    conn.commit()
    conn.close()
    return prompt_id


def insert_run(db_path: str | Path, case_id: int, image_path: str, prediction: dict, prompt_id: int ) -> int:
    """Insère ou met à jour le run correspondant au couple case_id / prompt_id."""
    init_db(db_path)
    conn = connect(db_path)

    values = (
        case_id,
        prompt_id,
        image_path,
        prediction.get("model_name"),
        json.dumps(prediction, ensure_ascii=False),
        prediction.get("predicted_class"),
        float(prediction.get("confidence", 0.0)),
        int(prediction.get("latency_ms", 0)),
    )

    conn.execute("""
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
    """, values)

    run_id = conn.execute(
        "SELECT id FROM runs WHERE case_id = ? AND prompt_id = ?",
        (case_id, prompt_id)
    ).fetchone()["id"]

    conn.commit()
    conn.close()
    return run_id


def insert_evaluation(db_path: str | Path, run_id: int, ground_truth: str, predicted: str) -> None:
    """Insère ou met à jour l'évaluation du run."""
    correct = int(predicted == ground_truth)
    error_type = None if correct else f"predicted_{predicted}_expected_{ground_truth}"

    conn = connect(db_path)

    conn.execute("""
        INSERT INTO evaluations (
            run_id, ground_truth_label, correct, error_type
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(run_id) DO UPDATE SET
            ground_truth_label = excluded.ground_truth_label,
            correct = excluded.correct,
            error_type = excluded.error_type,
            created_at = CURRENT_TIMESTAMP
    """, (run_id, ground_truth, correct, error_type))

    conn.commit()
    conn.close()


# --- Fonctions de lecture de la base de données --- 

def get_runs(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne tous les runs loggés, du plus récent au plus ancien """
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cases(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne toutes les radiographies de la table cases """
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM cases").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_prompts(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne tous les prompts, du plus récent au plus ancien """
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM prompts ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_evaluations(db_path: str | Path = DEFAULT_DB) -> list[dict]:
    """Retourne les évaluations avec les informations des runs et des prompts."""
    conn = connect(db_path)

    rows = conn.execute("""
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
    """).fetchall()

    conn.close()
    return [dict(row) for row in rows]