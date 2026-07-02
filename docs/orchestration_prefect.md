# Orchestration avec Prefect

Prefect remplace Airflow pour ce projet car il fonctionne plus simplement sous Windows.

## Pipeline orchestre

```text
import_abnormal_lung_dataset optionnel
-> import_rsna_pneumonia optionnel
-> generate_cases_csv
-> setup_database
-> train_pixel_baseline optionnel
-> run_evaluation MedGemma
-> compute_metrics optionnel
-> verify_outputs
```

Le flow par defaut lance MedGemma avec le prompt improved :

```text
generate_cases_csv
-> setup_database
-> train_pixel_baseline
-> run_evaluation en mode improved
-> compute_metrics
-> verify_outputs
```

Le flow est defini dans :

```text
orchestration/radiology_flow.py
```

## Installation

```powershell
python -m pip install -r requirements-orchestration.txt
```

## Lancement simple

```powershell
python orchestration\radiology_flow.py --max-cases 20
```

Cette commande lance MedGemma. Pour un premier test, limite plutot a 1 ou 2 cas :

```powershell
python orchestration\radiology_flow.py --skip-training --max-cases 1
```

## Lancement rapide sans reentrainement

```powershell
python orchestration\radiology_flow.py --skip-training --max-cases 5
```

## Smoke test sans MedGemma

Le mode `toy` sert seulement a tester l'orchestration rapidement :

```powershell
python orchestration\radiology_flow.py --skip-training --eval-mode toy --prompt-version 0 --max-cases 5
```

## Lancer seulement la preparation

```powershell
python orchestration\radiology_flow.py --skip-training --eval-mode none --skip-metrics
```

## Ajouter RSNA au pipeline

```powershell
python orchestration\radiology_flow.py --with-rsna --clean-imports --rsna-max-cases 150
```

RSNA peut echouer si Kaggle n'est pas connecte ou si les regles du challenge n'ont pas ete acceptees.

Pour continuer le pipeline meme si RSNA echoue :

```powershell
python orchestration\radiology_flow.py --with-rsna --continue-on-import-error --skip-training --max-cases 5
```

Pour corriger l'erreur Kaggle :

```text
1. Revoquer le token colle dans une conversation ou un document.
2. Aller sur Kaggle > Account > Create New API Token.
3. Lancer : python scripts\configure_kaggle_token.py --check
4. Coller le nouveau token quand le script le demande.
5. Accepter les regles du challenge rsna-pneumonia-detection-challenge.
6. Relancer le flow avec --with-rsna.
```

## Lancer une evaluation MedGemma

Prompt baseline :

```powershell
python orchestration\radiology_flow.py --skip-training --eval-mode baseline --prompt-version 0 --max-cases 5
```

Prompt improved :

```powershell
python orchestration\radiology_flow.py --skip-training --eval-mode improved --prompt-version 1 --max-cases 5
```

Comparaison baseline + improved :

```powershell
python orchestration\radiology_flow.py --skip-training --eval-mode full --prompt-version 1 --sample-n 20
```

## Interface Prefect optionnelle

Dans un premier terminal :

```powershell
prefect server start
```

Dans un deuxieme terminal :

```powershell
python orchestration\radiology_flow.py --max-cases 20
```

Puis ouvrir :

```text
http://127.0.0.1:4200
```

## Sorties attendues

- `data/cases.csv`
- `data/database.sqlite`
- `models/pixel_baseline.joblib`
- `eval/pixel_baseline_report.json`

RSNA reste optionnel car il depend de Kaggle. MedGemma est le mode par defaut du pipeline. Le mode `toy` est reserve aux smoke tests.
