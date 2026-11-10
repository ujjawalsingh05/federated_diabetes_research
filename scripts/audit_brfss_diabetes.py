from pathlib import Path
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "raw" / "LLCP2016.XPT"

print("Loading the diabetes response column...")
df, meta = pyreadstat.read_xport(
    str(FILE),
    usecols=["DIABETE3"]
)

print("\nVariable label:")
print(meta.column_names_to_labels.get("DIABETE3", "No label found"))

print("\nRaw response counts:")
print(df["DIABETE3"].value_counts(dropna=False).sort_index())

print("\nResponse labels:")
for code, label in meta.variable_value_labels.get("DIABETE3", {}).items():
    print(f"{code}: {label}")

print("\nAudit complete. No files were modified.")
