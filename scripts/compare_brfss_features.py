from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
XPT = DATA / "raw" / "LLCP2016.XPT"

print("=== EXISTING DATASET FEATURES ===")

for filename in ["raw/features.csv", "raw/targets.csv"]:
    path = DATA / filename
    if path.exists():
        df = pd.read_csv(path, nrows=3)
        print(f"\n{filename}")
        print("Columns:", df.columns.tolist())
    else:
        print(f"\nNot found: {filename}")

print("\n=== BRFSS VARIABLE CANDIDATES ===")

_, meta = pyreadstat.read_xport(str(XPT), metadataonly=True)

keywords = [
    "DIABETE", "BMI", "BPHIGH", "TOLDHI", "CHOLCHK",
    "SMOKE", "CVDSTRK", "CVDINFR", "EXER",
    "FRUIT", "VEGET", "HLTHPLN", "MEDCOST",
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "DIFFWALK",
    "SEX", "AGEG", "EDUCA", "INCOME"
]

for name in meta.column_names:
    label = meta.column_names_to_labels.get(name, "")
    if any(k in name.upper() or k in label.upper()
           for k in keywords):
        print(f"{name}: {label}")

print("\nAudit complete. No data or models were changed.")
