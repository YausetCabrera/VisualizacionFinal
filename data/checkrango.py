import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "rentamedia-sc-3.csv"

df = pd.read_csv(file_path, sep=",", encoding="utf-8", skipinitialspace=True)

# Asegurar tipo numérico por si acaso
df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

# Eliminar nulos por seguridad
values = df["OBS_VALUE"].dropna()

print("N válidos:", len(values))
print("N nulos:", df["OBS_VALUE"].isna().sum())

print("\nMIN:", values.min())
print("MAX:", values.max())

print("\nResumen estadístico:")
print(values.describe())

print("\nPercentiles:")
print(values.quantile([0.01, 0.05, 0.5, 0.95, 0.99]))