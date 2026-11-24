import pandas as pd
import pyreadstat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XPT = ROOT / "data" / "raw" / "LLCP2015.XPT"
OUT_CSV = ROOT / "data" / "processed" / "brfss_positive_records.csv"

print("Loading 2015 BRFSS dataset (this may take 1-2 minutes)...")
columns = [
    "DIABETE3", "PREDIAB1", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5",
    "SMOKE100", "CVDSTRK3", "_MICHD", "_TOTINDA", "_FRTLT1", "_VEGLT1",
    "_RFDRHV5", "HLTHPLN1", "MEDCOST", "GENHLTH", "MENTHLTH", "PHYSHLTH",
    "DIFFWALK", "SEX", "_AGEG5YR", "EDUCA", "INCOME2"
]
df, _ = pyreadstat.read_xport(str(XPT), usecols=columns)

# --- 1. TARGET SELECTION ---
is_diabetic = df['DIABETE3'] == 1.0
is_prediabetic_1 = df['DIABETE3'] == 4.0
is_prediabetic_2 = df['PREDIAB1'] == 1.0
df_pos = df[is_diabetic | is_prediabetic_1 | is_prediabetic_2].copy()

print(f"Initial positive candidate records: {len(df_pos)}")

# --- 2. FEATURE TRANSFORMATION ---
print("Transforming BRFSS features to match the 21-feature UCI format...")

# [CRITICAL FIX] Initialize with the correct index to prevent Pandas from dropping rows
transformed = pd.DataFrame(index=df_pos.index)

transformed['Diabetes_binary'] = 1.0

transformed['HighBP'] = df_pos['_RFHYPE5'].map({1.0: 0.0, 2.0: 1.0})
transformed['HighChol'] = df_pos['TOLDHI2'].map({1.0: 1.0, 2.0: 0.0})
# Fixed CholCheck logic: 1=checked in 5 yrs, 2/3=not checked in 5 yrs
transformed['CholCheck'] = df_pos['_CHOLCHK'].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})
transformed['BMI'] = df_pos['_BMI5'] / 100.0
transformed['Stroke'] = df_pos['CVDSTRK3'].map({1.0: 1.0, 2.0: 0.0})
transformed['HeartDiseaseorAttack'] = df_pos['_MICHD'].map({1.0: 1.0, 2.0: 0.0})
transformed['PhysActivity'] = df_pos['_TOTINDA'].map({1.0: 1.0, 2.0: 0.0})
transformed['Fruits'] = df_pos['_FRTLT1'].map({1.0: 1.0, 2.0: 0.0})
transformed['Veggies'] = df_pos['_VEGLT1'].map({1.0: 1.0, 2.0: 0.0})
transformed['HvyAlcoholConsump'] = df_pos['_RFDRHV5'].map({1.0: 0.0, 2.0: 1.0})
transformed['AnyHealthcare'] = df_pos['HLTHPLN1'].map({1.0: 1.0, 2.0: 0.0})
transformed['NoDocbcCost'] = df_pos['MEDCOST'].map({1.0: 1.0, 2.0: 0.0})
transformed['GenHlth'] = df_pos['GENHLTH'].where(df_pos['GENHLTH'] <= 5.0)
transformed['DiffWalk'] = df_pos['DIFFWALK'].map({1.0: 1.0, 2.0: 0.0})
transformed['Sex'] = df_pos['SEX'].map({1.0: 1.0, 2.0: 0.0})
transformed['Age'] = df_pos['_AGEG5YR'].where(df_pos['_AGEG5YR'] <= 13.0)
transformed['Education'] = df_pos['EDUCA'].where(df_pos['EDUCA'] <= 6.0)
transformed['Income'] = df_pos['INCOME2'].where(df_pos['INCOME2'] <= 8.0)
transformed['Smoker'] = df_pos['SMOKE100'].map({1.0: 1.0, 2.0: 0.0})

transformed['MentHlth'] = df_pos['MENTHLTH'].replace({88.0: 0.0})
transformed['MentHlth'] = transformed['MentHlth'].where(transformed['MentHlth'] <= 30.0)

transformed['PhysHlth'] = df_pos['PHYSHLTH'].replace({88.0: 0.0})
transformed['PhysHlth'] = transformed['PhysHlth'].where(transformed['PhysHlth'] <= 30.0)

# --- 3. CLEANING AND SAVING ---
print("Dropping rows with true missing or refused answers...")
final_df = transformed.dropna()
print(f"Final valid positive records successfully mapped: {len(final_df)}")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)
print(f"Saved extracted positive records to: {OUT_CSV}")