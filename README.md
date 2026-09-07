# Examen Aplicado de Machine Learning I
**Estudiante:** Alejandro Contreras  
**Institución:** Universidad Mayor — Escuela de Ingeniería  
**Asignatura:** Machine Learning I  
**Evaluación:** Examen Aplicado (30%)

---

## 1. Descripción del Dataset
* **Dataset:** California Housing
* **Fuente:** Scikit-Learn / StatLib (Pace & Barry, 1997)
* **URL:** [California Housing Dataset Docs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
* **Observaciones:** 20.640 registros
* **Predictores:** 8 variables numéricas continuas
* **Variable Objetivo:** `MedHouseVal` (Valor mediano de la vivienda distrital en cientos de miles de USD)
* **Tipo de Tarea:** Regresión Supervisada

---

## 2. Metodología Resumida
1. **Auditoría y EDA:** Verificación de integridad de datos (0.0% nulos), mitigación de valores atípicos mediante *capping* IQR en predictores clave y análisis de asimetría (*skewness* = 0.9765).
2. **Preprocesamiento sin Data Leakage:** División estratificada 80/20 (`random_state=42`) previa a transformaciones. Construcción de `ColumnTransformer` con `SimpleImputer(strategy='median')` y `StandardScaler()`, ajustado solo sobre `X_train`.
3. **Aprendizaje No Supervisado:** Reducción dimensional con PCA ($k=5$ componentes, $89.75\%$ de varianza explicada). Agrupamiento con K-Means ($K=3$), evaluado por inercia y coeficiente de Silhouette.
4. **Modelado Supervisado:** Optimización con validación cruzada ($cv=5$) sobre Ridge Regression y Random Forest Regressor, midiendo tiempos de entrenamiento.
5. **Diagnóstico y Evaluación:** Medición de métricas estrictas en test (RMSE, MAE, $R^2$, MAPE a 4 decimales), análisis de residuos y estudio de las 10 peores predicciones.

---

## 3. Resultados del Modelo Ganador
El modelo seleccionado por mejor desempeño y capacidad de capturar relaciones no lineales complejas fue **Random Forest Regressor**:

| Estimador | RMSE | MAE | $R^2$ | MAPE | Tiempo Entrenamiento (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Tuned)** | **0.5053** | **0.3275** | **0.8062** | **0.1874** | **18.4210** |
| Ridge Regression (Tuned) | 0.7456 | 0.5332 | 0.5758 | 0.3142 | 0.0412 |

*Los resultados detallados se encuentran exportados en `metricas_modelos.csv`.*

---

## 4. Enlace al Video de Presentación
* **Video Explicativo (≤ 8 min):** `[INSERTAR_AQUÍ_ENLACE_YOUTUBE_O_DRIVE]`  
*(Cobertura: Dataset y Justificación 1 min, EDA 2 min, PCA + K-Means 1 min, Modelos y Tabla 2.5 min, Conclusiones 1.5 min)*

---

## 5. Declaración de Uso de IA Generativa
> **Declaración Obligatoria:**  
> En cumplimiento con los estándares de integridad académica de la Universidad Mayor, se declara que se utilizó asistencia de Inteligencia Artificial Generativa (asistente conversacional de lenguaje) como soporte para la estructuración conceptual del flujo metodológico, optimización de sintaxis en scripts de visualización y revisión de redacción de los análisis ejecutivos. El procesamiento de datos, validación técnica, ejecución de modelos e interpretación de resultados fueron desarrollados, supervisados y verificados por el autor del trabajo.

---

## 6. Instrucciones de Reproducibilidad
Para clonar el repositorio y ejecutar el pipeline en un entorno local:

```bash
# 1. Clonar el repositorio
git clone [https://github.com/alejandro-engr/ML1_ExamenAplicado_Contreras_Alejandro.git](https://github.com/alejandro-engr/ML1_ExamenAplicado_Contreras_Alejandro.git)
cd ML1_ExamenAplicado_Contreras_Alejandro

# 2. Instalar dependencias exactas
pip install -r requirements.txt

# 3. Abrir el notebook ejecutado
jupyter notebook examen_ml1.ipynb