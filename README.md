# Predicción de Recompra B2B (Retail)

Este repositorio contiene la pipeline analítica completa para predecir la probabilidad de recompra de clientes B2B en un entorno e-commerce de retail.

## 🛠️ Estructura del Proyecto

```text
├── data/
│   ├── raw/                  # Datos brutos originales (no trackeados en git)
│   └── processed/            # Datos limpios, modelos serializados y entregables
├── notebooks/
│   ├── 1_EDA_y_Limpieza.ipynb      # Extracción, depuración e ingeniería de características
│   └── 2_Modelo_Recompra.ipynb     # Modelado predictivo, tuning y MLOps
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Guía de Ejecución

Sigue estos pasos en estricto orden para reproducir el modelo predictivo.

### 1. Preparación del Entorno
Instala las dependencias necesarias. Se recomienda usar un entorno virtual (`venv` o `conda`):
```bash
pip install -r requirements.txt
```

### 2. Ingesta de Datos (Data Sourcing)
El archivo original contiene los registros transaccionales históricos.
1. Descarga o sitúa el archivo original `online_retail_II.xlsx` (o el equivalente).
2. **Cópialo dentro de la carpeta `data/raw/`**. 

*Nota: Por políticas de privacidad y peso (LFS), el directorio `data/raw/` está excluido de Git.*

### 3. Ejecución de la Pipeline

#### Paso A: Depuración e Ingeniería de Datos
Abre y ejecuta **`notebooks/1_EDA_y_Limpieza.ipynb`** de principio a fin.
* **Qué hace:** Carga las múltiples hojas del Excel, limpia valores nulos/negativos, elimina outliers y calcula métricas RFM y Cohortes.
* **Qué genera:** Guarda el archivo `data/processed/df_model.parquet`, un dataset tabular maestro listo para que los algoritmos de Machine Learning lo ingieran.

#### Paso B: Modelado Predictivo y Extracción de Negocio
Abre y ejecuta **`notebooks/2_Modelo_Recompra.ipynb`** de principio a fin.
* **Qué hace:** Ingiere el Parquet, ejecuta el particionado, realiza validación cruzada entre varios algoritmos, optimiza XGBoost y Regresión Logística, extrae la curva Lift, analiza interpretabilidad con SHAP values y etiqueta a toda la base de clientes.
* **Qué genera:**
  1. `data/processed/best_xgboost_model.pkl`: El modelo predictivo serializado y optimizado para usar en producción.
  2. `data/processed/scaler.pkl`: El preprocesador (escalador) necesario para transformar futuros datos.
  3. `data/processed/predicciones_clientes_recompra.csv`: El **entregable de negocio final**, que contiene el ID de cada cliente, su probabilidad matemática de recompra y el segmento asignado (Alta Probabilidad, Dudoso, Riesgo de Fuga).

## 📊 Interpretabilidad
Ambos cuadernos están fuertemente documentados. Los gráficos estadísticos (Distribuciones RFM, Curvas ROC, Curva Lift y Feature Importance vía SHAP) incluyen conclusiones analíticas diseñadas para lectura directa por parte de Stakeholders y negocio.
