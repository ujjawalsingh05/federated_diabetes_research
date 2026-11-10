from pathlib import Path
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
xpt_path = ROOT / "data" / "raw" / "LLCP2016.XPT"

if not xpt_path.exists():
    raise FileNotFoundError(f"File not found: {xpt_path}")

print("Reading metadata only; patient records will not be loaded...")
_, meta = pyreadstat.read_xport(
    str(xpt_path),
    metadataonly=True
)

print(f"\nVariables found: {len(meta.column_names)}")

keywords = [
    "DIABET", "PREDIAB", "BMI", "BPHIGH", "BPMEDS",
    "CHOL", "SMOKE", "PHYSHLTH", "MENTHLTH",
    "GENHLTH", "EXER", "FRUIT", "VEGET", "AGE",
    "SEX", "EDUCA", "INCOME", "STROKE", "HEART",
    "WALK", "HEALTH"
]

for name in meta.column_names:
    label = meta.column_names_to_labels.get(name, "")
    if any(word in name.upper() or word in label.upper()
           for word in keywords):
        print(f"\n{name}: {label}")
        value_labels = meta.variable_value_labels.get(name, {})
        if value_labels:
            print("  Response labels:", value_labels)

