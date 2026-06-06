# 🛒 Motor Predictivo de Recompra B2B/B2C (Online Retail II)

Este repositorio contiene la arquitectura de datos, el flujo de procesamiento ETL, la ingeniería de variables y el motor predictivo de Machine Learning diseñado para **estimar la probabilidad de recompra** de los clientes de la compañía. 

El proyecto toma como punto de partida un histórico transaccional a nivel de factura (diciembre 2009 – diciembre 2011) y lo transforma en un activo estratégico para la toma de decisiones comerciales y la optimización del Retorno de Inversión (ROI) en campañas de retención.

---

## 🛠️ Estructura del Proyecto

```text
├── data/
│   ├── raw/                           # Dataset transaccional bruto original (Excluido de Git)
│   └── processed/                     # Datasets curados y listos para modelar (Parquet/CSV)
├── notebooks/
│   ├── 1_EDA_y_Limpieza.ipynb         # Depuración, división temporal e ingeniería de variables
│   └── 2_Modelo_Recompra.ipynb        # Modelado predictivo, calibración, interpretabilidad y exportación
├── modelos_produccion/                # Modelos y escaladores serializados para producción (.joblib)
├── images/                            # Gráficos e insights de negocio extraídos de los notebooks
├── Documentacion_Proyecto.md          # Documentación detallada en lenguaje de negocio para stakeholders
├── README.md                          # Guía técnica e instrucciones del repositorio
├── requirements.txt                   # Dependencias de Python necesarias
└── .gitignore                         # Exclusiones de archivos temporales y datos sensibles
```

---

## 🚀 Metodología y Arquitectura de la Solución

El proyecto está diseñado bajo un enfoque pragmático centrado en el negocio, implementado a lo largo de dos fases de desarrollo estructuradas:

### Fase 1: Limpieza, Robustez y Modelado del Perfil del Cliente
* **Depuración Transaccional:** Filtrado de registros sin identificar (Guest Checkouts), eliminación de transacciones duplicadas y aislamiento de transacciones con precios/cantidades nulas o negativas (devoluciones).
* **División Temporal Inteligente (Evitando Data Leakage):** Para asegurar una validación empírica robusta, se determinó una fecha de corte en **junio de 2011**:
  - **Ventana de Observación (Pasado):** 18 meses previos para calcular las variables del cliente.
  - **Ventana de Predicción (Futuro):** 6 meses posteriores para definir si el cliente efectivamente recompra (Target = 1) o se fuga (Target = 0).
* **Ingeniería de Características (ADN del Cliente):** 
  - Métricas de comportamiento **RFM**: Recencia, Frecuencia y Monetary (con capping en el percentil 99 para mitigar outliers de mayoristas).
  - **Índice de Salud (Recency-Tenure Ratio):** Métrica clave que relativiza la inactividad del cliente frente a su ciclo de vida en la empresa.
  - **Categorización Temática (NLP):** Procesamiento de lenguaje natural (TF-IDF y K-Means clustering) sobre las descripciones de las facturas para segmentar las preferencias de los usuarios en 8 clústeres temáticos.
  - **Dinámicas de Compra:** Tendencia de gasto reciente (Momentum), ratios de devolución e internacionalidad.

### Fase 2: Modelado Predictivo, Calibración e Interpretabilidad
* **Entrenamiento y Validación:** Particionado estratificado train/test (80/20) y validación cruzada de 5 pliegues (Stratified K-Fold).
* **Modelos Evaluados:** Regresión Logística (L1/L2), K-Nearest Neighbors (KNN), Random Forest y XGBoost.
* **Optimización (Tuning):** Búsqueda por rejilla (*GridSearchCV*) para controlar el sobreajuste (overfitting). El modelo **XGBoost Optimizado** (restringido a `max_depth: 3` y `learning_rate: 0.01`) se coronó como el campeón absoluto al maximizar la métrica de referencia (**AUC-ROC**).
* **Calibración Bayesiana (Platt Scaling):** Ajuste sigmoidal para alinear las probabilidades del modelo con la realidad, métrica indispensable para estimar de forma precisa el ROI.
* **Interpretabilidad Explicable (XAI):** Uso de valores **SHAP** y Coeficientes para abrir la "caja negra" del modelo y determinar las palancas que disparan el riesgo de fuga (`Recency_Tenure_Ratio`) o fomentan la retención (`Total_Quantity`).

---

## 📈 Entregables Estratégicos de Negocio

El pipeline analítico genera tres salidas tangibles y accionables:

1. **Preprocesador y Modelo en Producción (`modelos_produccion/`):**
   - `scaler_rfm.joblib`: Escalador estandarizado matemático entrenado con los datos históricos.
   - `modelo_recompra_lr.joblib`: Algoritmo clasificador optimizado y calibrado listo para su despliegue en producción.
2. **Listado de Clientes CRM (`data/processed/predicciones_clientes_recompra.csv`):**
   - Contiene la base de clientes identificada con su probabilidad de recompra estimada, el ticket medio (AOV), el **Ingreso Esperado (Expected Revenue = Probabilidad × AOV)** y la clasificación en tres segmentos clave de acción CRM: *Alta Probabilidad*, *Dudoso (Incentivar)* y *Riesgo de Fuga*.
3. **Curva de Ganancias Acumuladas (Lift):**
   - Demuestra que apuntando las campañas solo al **30% de los clientes** con mayor propensión estimada por el modelo, capturamos a la gran mayoría de los recompradores potenciales, maximizando drásticamente el presupuesto de marketing.

---

## 🛠️ Guía de Instalación y Ejecución

### 1. Requisitos Previos
Clona el repositorio e instala las dependencias de Python (preferiblemente en un entorno virtual limpio):
```bash
# Crear entorno virtual
python -m venv .venv
source .venv/Scripts/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Sourcing de Datos
1. Descarga el archivo de origen `online_retail_II.xlsx`.
2. Crea el directorio `data/raw/` y deposita el archivo dentro.

### 3. Ejecución del Flujo
Ejecuta secuencialmente los cuadernos de Jupyter en la carpeta `notebooks/`:
- **`1_EDA_y_Limpieza.ipynb`**
- **`2_Modelo_Recompra.ipynb`**

---

## 📅 Hoja de Ruta (Evolución Analítica)

### Corto Plazo (Hacia la versión 1.5): Mejoras Incrementales
* **A/B Testing en Campañas:** Desplegar pilotos controlados utilizando la segmentación CRM actual para medir el dinero rescatado y refinar los costes de adquisición/retención.
* **Feature Engineering Detallado:** Desarrollar variables sobre temporalidad semanal (días laborales vs. fines de semana) y varianza de los ciclos de compra del usuario.
* **Optimización Bayesiana:** Implementar optimización inteligente sobre los hiperparámetros para mejorar el rendimiento predictivo sin aumentar la complejidad del modelo.

### Largo Plazo (Hacia la versión 2.0): Transformación Analítica
* **Modelos de Valor del Cliente (CLV):** Evolucionar del modelo binario de clasificación a un modelo de regresión que estime los ingresos totales en libras que aportará cada cliente.
* **Modelado de Secuencias (Deep Learning):** Integrar arquitecturas recurrentes (LSTM) o transformers tabulares para estimar el ciclo de compra en tiempo real del usuario.
* **Motores de Recomendación Interconectados:** Enlazar la propensión a la recompra con recomendaciones hiper-personalizadas de productos específicos (Cross-Sell / Up-Sell) para cada usuario VIP en riesgo de fuga.
