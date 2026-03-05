import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Configurar MLflow con SQLite local
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_experiment")

# 2. Cargar el dataset
print("Cargando dataset...")
df = pd.read_csv("diabetes.csv")

# Separar variables predictoras (X) de la variable objetivo (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split simple para este experimento de prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenamiento del modelo base
print("Entrenando modelo básico de Regresión Logística...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluación básica
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 4. Tracking y Registro en MLflow
model_name = "Diabetes_Model_Basic"

with mlflow.start_run(run_name="Basic_LR_Test") as run:
    # Registrar la métrica de prueba
    mlflow.log_metric("accuracy", acc)
    
    # Proporcionar un ejemplo de entrada es buena práctica para registrar la firma (signature) del modelo
    input_example = X_test[:2]
    
    # Registrar el modelo y subirlo al Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        registered_model_name=model_name,
    )

print(f"✅ Entrenamiento finalizado. Accuracy: {acc:.4f}")
print(f"✅ Modelo registrado en MLflow con el nombre: '{model_name}'")