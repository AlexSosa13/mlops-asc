import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Crear directorio para guardar los splits si no existe
os.makedirs("data_splits", exist_ok=True)

# 2. Cargar el dataset original
print("Cargando diabetes.csv...")
df = pd.read_csv("diabetes.csv")

# 3. Primera división: Separar el 10% para TEST (y dejar el 90% para Train+Val)
# Usamos stratify=df["Outcome"] para mantener la proporción de clases
df_temp, df_test = train_test_split(
    df, 
    test_size=0.10, 
    stratify=df["Outcome"], 
    random_state=42
)

# 4. Segunda división: Del 90% restante, queremos un 70% (Train) y un 20% (Validación).
# La proporción matemática para sacar un 20% del total a partir de un 90% es 20/90.
df_train, df_val = train_test_split(
    df_temp, 
    test_size=(20/90), 
    stratify=df_temp["Outcome"], 
    random_state=42
)

# 5. Comprobación de tamaños
print("\n=== TAMAÑOS DE LOS DATASETS ===")
print(f"Total original: {len(df)} filas (100%)")
print(f"Train:          {len(df_train)} filas ({len(df_train)/len(df)*100:.1f}%)")
print(f"Validación:     {len(df_val)} filas ({len(df_val)/len(df)*100:.1f}%)")
print(f"Test:           {len(df_test)} filas ({len(df_test)/len(df)*100:.1f}%)")

# 6. Guardar los datasets en archivos CSV separados
df_train.to_csv("data_splits/train.csv", index=False)
df_val.to_csv("data_splits/val.csv", index=False)
df_test.to_csv("data_splits/test.csv", index=False)

print("\n✅ Datasets guardados correctamente en la carpeta 'data_splits/'")