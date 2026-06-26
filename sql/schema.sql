CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY,
    image_path TEXT NOT NULL,
    source TEXT,
    ground_truth_label TEXT,
    split TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER,
    prompt_id INTEGER, 
    image_path TEXT,
    model_name TEXT,
    prediction_json TEXT,
    predicted_class TEXT,
    confidence REAL,
    latency_ms INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(case_id) REFERENCES cases(id), -- to link the case table
    FOREIGN KEY(prompt_id) REFERENCES prompts(id) -- to link the prompt table
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    ground_truth_label TEXT,
    correct INTEGER,
    error_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);
