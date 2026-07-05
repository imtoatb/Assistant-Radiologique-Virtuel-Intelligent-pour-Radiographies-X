import sys
sys.path.append(".")
from src.anonymizer import audit_metadata, anonymize_to_path

path = "data/brutes_kaggle/ABNORMAL-LUNG_00001_suspected_opacity_001.jpg"

print("aVANT anonymisation")
avant = audit_metadata(path)
print(avant["summary"])
print("Tags trouvés :", avant["found_tags"])
print("Nombre de champs EXIF :", avant["exif_count"])

clean = anonymize_to_path(path, "tmp_test_anon.jpg")

print("\n=après anonymisation")
apres = audit_metadata(clean)
print(apres["summary"])
print("Tags trouvés :", apres["found_tags"])
print("Nombre de champs EXIF :", apres["exif_count"])

import os; os.remove("tmp_test_anon.jpg")