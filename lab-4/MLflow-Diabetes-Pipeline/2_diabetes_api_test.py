import requests
import json
import pandas as pd

# 1. Cargar unas cuantas filas reales del dataset para la prueba
df = pd.read_csv("diabetes.csv")
X_test = df.drop("Outcome", axis=1).head(3)

# 2. Preparar el payload JSON esperado por MLflow para modelos basados en Pandas
# Utilizamos orient="split" que genera un diccionario con "columns" y "data"
payload = {
    "dataframe_split": X_test.to_dict(orient="split")
}

# 3. Hacer la petición POST al endpoint invocations
print("Enviando datos a la API de MLflow...")
try:
    response = requests.post(
        url="http://127.0.0.1:5000/invocations",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    if response.status_code == 200:
        preds = response.json()
        print("\n✅ === PREDICCIONES DE LA API ===")
        print(f"Resultados (0 = No Diabetes, 1 = Diabetes): {preds['predictions']}")
    else:
        print(f"❌ Error en la API. Código de estado: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Error de conexión: ¿El servidor de MLflow sigue corriendo en el puerto 5000?")