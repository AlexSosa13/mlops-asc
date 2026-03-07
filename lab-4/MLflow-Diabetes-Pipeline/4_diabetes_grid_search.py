import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools

# 1. Configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("diabetes_grid_search")

# 2. Cargar los splits de datos (Train y Validation)
print("Cargando datos de train y validation...")
df_train = pd.read_csv("data_splits/train.csv")
df_val = pd.read_csv("data_splits/val.csv")

X_train = df_train.drop("Outcome", axis=1)
y_train = df_train["Outcome"]

X_val = df_val.drop("Outcome", axis=1)
y_val = df_val["Outcome"]

# 3. Definir el Grid de Hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Generar todas las combinaciones posibles
keys, values = zip(*param_grid.items())
hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Iniciando Grid Search con {len(hyperparameter_combinations)} combinaciones...")

# 4. Bucle principal del Grid Search con MLflow Tracking
for i, params in enumerate(hyperparameter_combinations):
    run_name = f"RF_Run_{i+1}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"\nEntrenando {run_name} con parámetros: {params}")
        
        # Registrar parámetros en MLflow
        mlflow.log_params(params)
        
        # Entrenar el modelo
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        
        # Predecir sobre el conjunto de VALIDACIÓN
        y_pred = model.predict(X_val)
        
        # Calcular métricas
        acc = accuracy_score(y_val, y_pred)
        # Importante: zero_division=0 evita warnings si el modelo predice todo ceros al principio
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Registrar el modelo (sin subirlo al Model Registry todavía, eso lo haremos en la Fase 4)
        input_example = X_train.head(2)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example
        )
        
        print(f"Resultados -> Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

print("\n✅ Grid Search finalizado. Puedes ver los resultados ejecutando 'mlflow ui' (o 'uv run mlflow ui').")