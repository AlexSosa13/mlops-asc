import pandas as pd
import requests
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar el dataset de Test (10% de datos invisibles para el modelo)
print("Cargando dataset de test...")
df_test = pd.read_csv("data_splits/test.csv")
X_test = df_test.drop("Outcome", axis=1)
y_test = df_test["Outcome"]  # La "verdad" que usaremos para evaluar

# 2. Preparar el payload para la API (formato Pandas split)
payload = {
    "dataframe_split": X_test.to_dict(orient="split")
}

# 3. Llamar a la API de MLflow (corriendo en otra terminal)
print("Enviando los datos de Test a la API de MLflow...")
try:
    response = requests.post(
        url="http://127.0.0.1:5000/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        # Extraer las predicciones de la respuesta JSON
        preds = response.json()["predictions"]
        
        # 4. Calcular métricas reales comparando las predicciones con y_test
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print("\n✅ === MÉTRICAS FINALES SOBRE DATASET DE TEST (10%) ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}  <-- Métrica principal médica")
        print(f"F1-Score:  {f1:.4f}")
        
    else:
        print(f"❌ Error en la API. Código: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Error de conexión: Asegúrate de haber levantado el modelo 'champion' en el puerto 5000.")