# Entregables - Laboratorio 4 - Alejandro Sosa Corral

## Entregable 1: Tabla de parámetros y métricas de los modelos registrados

Tras realizar un Grid Search exhaustivo con un `RandomForestClassifier` y evaluar múltiples combinaciones de hiperparámetros, se seleccionaron los dos mejores modelos basándonos en criterios médicos. 

En el diagnóstico de diabetes, un falso negativo (no detectar la enfermedad) tiene un coste mucho mayor para la salud del paciente que un falso positivo. Por ello, la métrica elegida para determinar a los ganadores fue el **Recall (Sensibilidad)**, desempatando con el F1-Score para mantener un equilibrio razonable con la precisión.

| Alias | Puesto | n_estimators | max_depth | min_samples_split | Recall (Val) | F1-Score (Val) | Accuracy (Val) |
|-------|--------|--------------|-----------|-------------------|--------------|----------------|----------------|
| `@champion` | 1 | 200 | 10 | 2 | 0.5926 | 0.6214 | 0.7468 |
| `@staging` | 2 | 200 | None | 2 | 0.5741 | 0.6139 | 0.7468 |

Se decidió registrar estos modelos y no los de otros *runs* porque, aunque otras combinaciones pudieran tener un `Accuracy` ligeramente superior, estas configuraciones maximizaban la detección de pacientes positivos (Recall), lo cual es el requerimiento principal en este contexto de salud.

---

## Entregable 2: Script de obtención de métricas de la API
*(El código de este entregable se encuentra en el archivo `MLflow-Diabetes-Pipeline/6_diabetes_test_evaluation.py` del repositorio).*

---

## Entregable 3: Conclusiones

Tras evaluar nuestro modelo campeón (Random Forest con `n_estimators=200`, `max_depth=10`, `min_samples_split=2`) contra el 10% del dataset reservado para test a través de la API de MLflow, obtuvimos los siguientes resultados:

* **Accuracy:** 0.7662
* **Precision:** 0.6552
* **Recall:** 0.7037
* **F1-Score:** 0.6786

**Conclusiones:**
1. **Generalización del modelo:** El modelo ha demostrado una excelente capacidad de generalización. Al comparar las métricas de Test con las de Validación, observamos que el Recall aumentó de 0.59 a 0.70, y el Accuracy pasó de 0.74 a 0.76. Esto confirma la ausencia de *overfitting* (sobreajuste).
2. **Trade-off de métricas:**  Logramos cumplir el objetivo médico principal superando la barrera del 70% en Recall en datos no vistos, aceptando a cambio una precisión del 65.5% (lo que significa que algunos pacientes sanos serán clasificados erróneamente como diabéticos y requerirán pruebas adicionales, un margen de error aceptable en triaje médico).
3. **Ciclo de vida con MLflow:** La herramienta MLflow ha demostrado ser vital para estructurar el experimento. Nos permitió pasar rápidamente de entrenar múltiples combinaciones (Grid Search) a seleccionar, registrar y finalmente exponer como API el modelo ganador, todo ello manteniendo la trazabilidad absoluta del código, los parámetros y los datos (gracias al esquema de divisiones de train/val/test).