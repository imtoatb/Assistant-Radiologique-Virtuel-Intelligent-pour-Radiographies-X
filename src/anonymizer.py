"""
anonymizer.py
=============
Anonymisation des images entrantes avant tout traitement.

Ce module supprime toutes les métadonnées embarquées dans l'image
(EXIF, IPTC, XMP, commentaires) qui pourraient contenir des informations
patient (nom, date de naissance, établissement, numéro de sécurité sociale…).

Même si les images sont en JPEG/PNG et non en DICOM, les appareils modernes
et certains logiciels PACS exportent des métadonnées dans les fichiers JPEG.

IMPORTANT : Ce module ne modifie pas le fichier original.
Il retourne une image PIL propre et/ou sauvegarde une copie anonymisée.

Usage :
    from src.anonymizer import anonymize_image, anonymize_to_path

    # Retourne une PIL.Image sans métadonnées
    clean_image = anonymize_image("path/to/image.jpg")

    # Sauvegarde une copie anonymisée sur disque
    anonymize_to_path("path/to/image.jpg", "path/to/clean_image.jpg")

    # Vérifie si une image contient des métadonnées sensibles
    report = audit_metadata("path/to/image.jpg")
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image


# Champs EXIF connus pour contenir des infos patient ou établissement
_SENSITIVE_EXIF_TAGS = {
    0x013B: "Artist",
    0x8298: "Copyright",
    0x9286: "UserComment",
    0x927C: "MakerNote",
    0x9C9B: "XPAuthor",
    0x9C9C: "XPComment",
    0x9C9D: "XPKeywords",
    0x9C9E: "XPSubject",
    0x9C9F: "XPTitle",
    0x0131: "Software",
    0x013E: "WhitePoint",
    0x0132: "DateTime",
    0x9003: "DateTimeOriginal",
    0x9004: "DateTimeDigitized",
    0x013C: "HostComputer",
    0x010F: "Make",
    0x0110: "Model",
    0x0112: "Orientation",
}


def anonymize_image(path: str | Path) -> Image.Image:
    """
    Charge une image et retourne une copie PIL sans aucune métadonnée.

    La technique consiste à re-encoder l'image en mémoire via PIL,
    ce qui supprime tous les chunks de métadonnées (EXIF, IPTC, XMP,
    commentaires PNG, chunks tEXt…).

    L'image elle-même (pixels) n'est pas modifiée.
    """
    path = Path(path)
    original = Image.open(path)

    # Conserver le mode original (L pour gris, RGB, RGBA…)
    mode = original.mode

    # Re-créer l'image depuis les données pixel brutes uniquement
    # getdata() retourne les valeurs de pixels sans aucune métadonnée
    clean = Image.new(mode, original.size)
    clean.putdata(list(original.getdata()))

    return clean


def anonymize_to_path(
    source: str | Path,
    destination: str | Path | None = None,
    quality: int = 95,
) -> Path:
    """
    Sauvegarde une copie anonymisée de l'image sur disque.

    Si destination n'est pas fourni, sauvegarde dans le même dossier
    avec le suffixe _anon avant l'extension.

    Retourne le chemin de destination.
    """
    source = Path(source)

    if destination is None:
        destination = source.parent / f"{source.stem}_anon{source.suffix}"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    clean = anonymize_image(source)

    # Sauvegarde sans passer de paramètre exif/info
    save_kwargs: dict[str, Any] = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    clean.save(destination, **save_kwargs)
    return destination


def audit_metadata(path: str | Path) -> dict[str, Any]:
    """
    Analyse les métadonnées présentes dans l'image et retourne un rapport.

    Utile pour vérifier avant/après anonymisation, et pour le rapport
    de conformité du prototype.

    Retourne :
        has_sensitive_metadata (bool)  : True si des métadonnées sensibles ont été trouvées
        found_tags (list[str])         : noms des tags détectés
        exif_count (int)               : nombre total de champs EXIF présents
        has_png_text (bool)            : True si des chunks texte PNG sont présents
        summary (str)                  : résumé lisible
    """
    path = Path(path)
    image = Image.open(path)

    found_tags: list[str] = []
    exif_count = 0
    has_png_text = False

    # ── EXIF (JPEG principalement) ──────────────────────────────────────────
    try:
        exif_data = image._getexif()  # type: ignore[attr-defined]
        if exif_data:
            exif_count = len(exif_data)
            for tag_id, value in exif_data.items():
                if tag_id in _SENSITIVE_EXIF_TAGS:
                    tag_name = _SENSITIVE_EXIF_TAGS[tag_id]
                    if value and str(value).strip():
                        found_tags.append(f"EXIF:{tag_name}={str(value)[:50]}")
    except (AttributeError, Exception):
        pass

    # ── Métadonnées PNG (chunks tEXt, zTXt, iTXt) ─────────────────────────
    try:
        info = image.info or {}
        text_keys = [k for k in info if isinstance(info[k], str) and info[k].strip()]
        if text_keys:
            has_png_text = True
            for k in text_keys:
                found_tags.append(f"PNG:{k}={str(info[k])[:50]}")
    except Exception:
        pass

    has_sensitive = bool(found_tags) or exif_count > 5

    if not found_tags and exif_count == 0:
        summary = "Aucune métadonnée détectée."
    elif found_tags:
        summary = f"{len(found_tags)} champ(s) potentiellement sensible(s) détecté(s) : {', '.join(t.split('=')[0] for t in found_tags)}"
    else:
        summary = f"{exif_count} champs EXIF présents (aucun identifié comme sensible)."

    return {
        "has_sensitive_metadata": has_sensitive,
        "found_tags": found_tags,
        "exif_count": exif_count,
        "has_png_text": has_png_text,
        "summary": summary,
    }