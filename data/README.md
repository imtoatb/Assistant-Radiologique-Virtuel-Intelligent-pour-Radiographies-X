# Données

Ce dossier contient un jeu **synthétique jouet** destiné à tester l'architecture, les logs, les métriques et l'interface. Il ne s'agit pas d'un dataset médical réel.

Pour un vrai projet, utiliser un dataset autorisé comme RSNA Pneumonia, CheXpert, MIMIC-CXR ou NIH ChestXray, en respectant les licences et les conditions d'accès.

## Datasets réels utilisés

- `brutes_kaggle/` : dataset Kaggle `fkarimovv/abnormal-lung`, utilisé en premier car simple à importer (pas d'authentification particulière). Sert de pilote pour valider tout le pipeline. Résultats dans `database_kaggle_dataset.sqlite` / `cases_kaggle_dataset.csv`.
- `brutes_rsa/` : dataset RSNA Pneumonia Detection Challenge, le dataset officiel demandé par le cahier des charges. Importé via `scripts/import_rsna_pneumonia.py` (nécessite un compte Kaggle authentifié). Résultats dans `database.sqlite` / `cases.csv` (les fichiers actifs par défaut).

Les deux jeux de données restent dans des bases séparées pour ne jamais mélanger les images ou les runs des deux datasets.

## `synthetic_cases.csv`

Colonnes :

- `case_id`
- `image_path`
- `source`
- `label`
- `split`
- `quality`
- `notes`

## Images synthétiques

Les images dans `sample_images/` imitent grossièrement une radiographie thoracique uniquement pour vérifier les flux de code. Elles ne doivent pas être utilisées pour évaluer une performance médicale.
