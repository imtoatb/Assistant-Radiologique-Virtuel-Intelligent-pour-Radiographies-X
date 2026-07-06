# Pulmonar - Assistant Radiologue IA

Prototype pedagogique d'IA multimodale pour l'analyse prudente de radiographies thoraciques frontales, appuyé sur MedGemma-4b-it.

## Avertissement

**Prototype pédagogique. Non destine au diagnostic. Validation par un professionnel qualifié requise.**

Ce projet n'est pas un dispositif médical. Il ne pose aucun diagnostic. Aucune donnée patient réelle n'est utilisée. Toute sortie doit être relue par un professionnel de sante.

## Objectif

Recevoir une radiographie thoracique frontale et retourner un JSON structuré :
qualité de l'image, classe predite (`normal`, `suspected_opacity`, `uncertain`), confiance, observations visuelles, justification courte, limites et avertissement obligatoire.

Le projet compare une baseline par prompting a une version ameliorée (prompt renforce, seuil d'incertitude, garde-fous).

## Architecture (pipeline)

1. Upload de l'image via l'application web (FastAPI).
2. Validation de l'image (`src/image_validator.py`) : format, taille, niveaux de gris, contraste, ressemblance thorax.
3. Anonymisation (`src/anonymizer.py`) : suppression des metadonnées EXIF.
4. Inference (`src/inference.py`) : toy baseline, baseline pixel logistique, ou MedGemma-4b-it.
5. Garde-fous JSON (`src/guardrails.py`) : validation du schema, warning obligatoire, repli sur `uncertain`.
6. Sauvegarde en base SQLite (`src/database.py`, `sql/schema.sql`).

## Prerequis

- Windows, Python **3.12** (ne pas utiliser 3.14 : pas de whéels stables pour bitsandbytes et une partie du stack ML).
- GPU NVIDIA pour MedGemma (teste sur RTX 4050 Laptop 6 Go, quantification 4-bit). Le CPU suffit pour la baseline toy et les tests.

## Installation

```
py -3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip

# torch CUDA en premier : il n'est pas dans requirements.txt pour eviter que pip
# tire la version CPU par defaut depuis PyPI
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

python -m pip install -r requirements.txt
```

Verification GPU :
```
python -c "import torch; print(torch.cuda.is_available())"
```
Doit afficher `True`.

Note : si pip renvoie une erreur de certificat SSL sur un venv Python 3.12.0 fraichement crée, c'est le certifi vendored de pip qui est trop ancien. Corriger en amorcant une fois :
```
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Lancer l'application web

Depuis la racine du projet, avec le venv actif :
```
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Puis ouvrir http://localhost:8000/

L'interface propose plusieurs modèles via le menu deroulant : **Pixel Baseline** (regression logistique, rapide), **Reference / Ameliore (jouet)** (regle de test), **MedGemma Baseline** (prompt v0) et **MedGemma Ameliore** (prompt v10, le meilleur trouve sur RSNA).

## Importer le dataset RSNA

A faire **avant** toute évaluation réelle : les commandes d'évaluation ci-dessous s'appuient sur `data/cases.csv` et `data/database.sqlite`, produits par cet import. Necessite un compte Kaggle authentifie ayant accepte les regles de la competition `rsna-pneumonia-detection-challenge` :
```
python scripts/import_rsna_pneumonia.py --max-cases 100
python scripts/setup_db.py --all
```
(Le smoke eval ci-dessous, lui, tourne sur un dataset synthetique deja fourni et ne necessite pas cet import.)

## Lancer l'évaluation

Smoke eval (rapide, predicteur toy, dataset synthetique, verifie que la chaine fonctionne) :
```
python eval/run_evaluation.py --mode toy --out-dir eval/outputs --db-path data/database.sqlite
```

Campagne réelle sur MedGemma (comparaison de prompts sur les vraies radios, sauvegarde en base). Comparaison finale baseline v0 vs meilleur prompt v10 sur les 100 cas du holdout :
```
python scripts/run_prompt_evaluation.py --mode baseline --prompt-version 0 --case-ids-file data/rsna_holdout_sample_100.txt --cases-path data/cases.csv --db-path data/database.sqlite
python scripts/run_prompt_evaluation.py --mode improved --prompt-version 10 --case-ids-file data/rsna_holdout_sample_100.txt --cases-path data/cases.csv --db-path data/database.sqlite
```
`--mode` accepte `baseline`, `improved` ou `full` (baseline + improved sur le meme echantillon) ; `--prompt-version` selectionne `prompts/{mode}_prompt_{version}.txt` ; `--case-ids-file` lit une liste de case_id depuis un fichier. Par defaut `data/cases.csv` / `data/database.sqlite` pointent vers le dataset RSNA (le dataset actif). Pour relancer une analyse sur le dataset Kaggle pilote, passer `--cases-path data/cases_kaggle_dataset.csv --db-path data/database_kaggle_dataset.sqlite`.

## Lancer les tests

```
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q
```

## Structure du depot

- `app/` : application FastAPI + frontend HTML.
- `api/` : API minimale historique.
- `src/` : logique metier (inference, validation, anonymisation, garde-fous, base, metriques).
- `prompts/` : prompt baseline et versions ameliorées.
- `scripts/` : import de datasets, entrainement des baselines, evaluation, batch.
- `eval/` : sorties d'évaluation et registre d'erreurs.
- `data/` : images, base SQLite, logs.
- `docs/` : appel d'offre, architecture, protocole d'évaluation, ethique et limites.
- `tests/` : smoke test de conformite du depot.

## Modèles

- **toy baseline** : règle simple sur le nom de fichier et le signal image (pour les tests et la reproductibilité).
- **baseline pixel** : regression logistique sur features image (`models/pixel_baseline_rsna.joblib` pour RSNA, `models/pixel_baseline.joblib` pour le pilote Kaggle).
- **MedGemma-4b-it** : VLM medical, quantifie 4-bit, prompt charge depuis `prompts/` (v0 baseline, v10 meilleur prompt ameliore).

## Sources et licences des données

Deux datasets réels ont ete utilises, avec des bases separées (`data/*_kaggle_dataset.*` vs `data/database.sqlite` + `data/cases.csv`) pour ne jamais melanger les deux :

- **Kaggle `fkarimovv/abnormal-lung`** (dataset pilote) : utilise en premier car simple et rapide a importer (`scripts/import_abnormal_lung_dataset.py`, pas d'authentification particuliere), pour demarrer et valider tout le pipeline (baseline, prompts ameliores, garde-fous, base de données) sans attendre. 932 images, resultats dans `data/database_kaggle_dataset.sqlite` (notebook `04_Database_visualisation.ipynb`).
- **RSNA Pneumonia Detection Challenge** (dataset officiel du cahier des charges) : importe via `scripts/import_rsna_pneumonia.py` (`kagglehub`, necessite un compte Kaggle authentifie ayant accepte les regles de la competition). Images DICOM converties en PNG, labels `normal`/`suspected_opacity` derives du champ `Target`. Resultats dans `data/database.sqlite` (notebook `05_Database_visualisation_rsna.ipynb`).

MedGemma-4b-it est soumis a des conditions d'usage specifiques - voir la model card Hugging Face (`google/medgemma-4b-it`).

## Ethique et limites

Voir `docs/ethique_et_limites.md`. Points cles : aucune donnée patient réelle, warning present partout (UI, JSON, README, rapport), pas de diagnostic definitif, repli sur `uncertain` en cas d'incertitude ou de mauvaise qualite d'image.
