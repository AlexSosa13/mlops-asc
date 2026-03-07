import mlflow
from mlflow.tracking import MlflowClient

# 1. Configurar MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# 2. Obtener el ID del experimento
experiment = client.get_experiment_by_name("diabetes_grid_search")
experiment_id = experiment.experiment_id

# 3. Buscar los mejores runs ordenados por Recall (y desempatar con F1-Score)
print("Buscando los mejores modelos en MLflow...")
runs_df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.recall DESC", "metrics.f1_score DESC"],
    max_results=2 # Nos quedamos solo con los 2 mejores
)

model_name = "Diabetes_RF_Model"
print(f"\n=== SELECCIÓN DE LOS MEJORES MODELOS (Criterio: Mayor Recall) ===")

# 4. Registrar los modelos y asignarles Alias
for i, row in runs_df.iterrows():
    run_id = row['run_id']
    recall = row['metrics.recall']
    f1 = row['metrics.f1_score']
    acc = row['metrics.accuracy']
    
    # Extraer los parámetros (esto te servirá para tu Entregable 1)
    n_estimators = row['params.n_estimators']
    max_depth = row['params.max_depth']
    min_samples_split = row['params.min_samples_split']
    
    print(f"\n🏆 Puesto {i+1}:")
    print(f"Parámetros: n_estimators={n_estimators}, max_depth={max_depth}, min_samples={min_samples_split}")
    print(f"Métricas  -> Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")
    
    # Construir la URI del modelo basada en el run_id
    model_uri = f"runs:/{run_id}/model"
    
    # Registrar el modelo en el Model Registry
    registered_model = mlflow.register_model(model_uri, model_name)
    
    # Asignar alias ('champion' al primero, 'staging' al segundo)
    alias = "champion" if i == 0 else "staging"
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=registered_model.version
    )
    
    print(f"✅ Registrado como '{model_name}' (Versión {registered_model.version}) con Alias: '{alias}'")

print("\n¡Selección y registro completados con éxito!")