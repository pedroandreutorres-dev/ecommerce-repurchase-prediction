# 🛒 Motor Predictivo de Recompra B2B/B2C (Online Retail II)

Este repositorio contiene la arquitectura de datos, el flujo de procesamiento analítico y el motor predictivo de Machine Learning diseñado para **estimar la probabilidad de que un cliente repita compra**. 

El objetivo final del proyecto no es construir un algoritmo de caja negra, sino **convertir los datos en decisiones de negocio accionables**, permitiendo al equipo de Marketing desplegar acciones de *fidelización* (para clientes con alta propensión) y acciones de *retención/recuperación* (para aquellos con baja propensión).

---

## 🛠️ Estructura del Proyecto

```text
├── data/
│   ├── raw/                           # Dataset transaccional bruto original (Online Retail II)
│   └── processed/                     # Datasets curados y listos para modelar (Nivel Cliente)
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

## 🚀 Metodología y Decisiones Analíticas Clave

El proyecto está diseñado bajo un enfoque pragmático y orientado al retorno de inversión (ROI), aplicando criterios rigurosos para la limpieza e ingeniería de datos:

### Fase 1: Construcción del Perfil de Cliente (EDA y Limpieza)
El dataset original se presenta a nivel de ticket (factura). El primer reto fue pivotarlo para construir una **visión única de cliente**:

* **Tratamiento de Clientes Anónimos:** Se identificaron registros con `Customer IDs` nulos (Guest Checkouts). Al no tener historial identificable que permita construir una radiografía o medir si recompran o no, **se eliminaron del modelado** para evitar introducir ruido estadístico.
* **Gestión Bifurcada de Devoluciones:** Las transacciones con cantidades negativas (devoluciones) se excluyeron del cálculo de volumen de facturación pura (`Monetary`) para no contaminarlo. Sin embargo, se utilizaron de forma inteligente para crear un `Ratio de Devoluciones`, una variable predictora clave que actúa como señal de insatisfacción.
* **División Temporal (Prevención de Data Leakage):** Para construir la variable objetivo de forma honesta, se definió una fecha de corte estricta común para todos los usuarios.
  - **Ventana Input (Input Features):** Todo lo ocurrido antes de la fecha de corte se utilizó para extraer el historial del cliente.
  - **Ventana Target:** El comportamiento en los meses posteriores sirvió para etiquetar matemáticamente si hubo recompra (1) o fuga (0).
* **Ingeniería de Características (Más allá del RFM):** Además de Recencia, Frecuencia y Valor Monetario, se calcularon métricas avanzadas como la distancia temporal entre compras, variedad de productos comprados, ratios de inactividad (`Recency-Tenure Ratio`) y categorización temática mediante NLP sobre las descripciones de factura.

### Fase 2: Modelado Predictivo, Evaluación y Negocio
* **Benchmark Algorítmico:** Se evaluaron modelos de Regresión Logística, KNN, Random Forest y XGBoost, validando su rendimiento con Validación Cruzada Estratificada.
* **Métrica de Evaluación Principal:** Siguiendo el rigor del caso de negocio, el modelo se optimizó priorizando la métrica de referencia **AUC-ROC** y la evaluación directa de la **Matriz de Confusión**. Esto permite un análisis global robusto antes de decantarse por afinar `Precision` o `Recall` bajo escenarios específicos de coste financiero.
* **El Campeón - Random Forest Optimizado:** Tras la optimización de hiperparámetros mediante *GridSearchCV*, el modelo `Random Forest` superó a sus competidores demostrando la mayor estabilidad, generalización y área bajo la curva (AUC).
* **Interpretabilidad Explicable (XAI):** Se utilizó la librería **SHAP** para asegurar que el modelo fuera transparente y prescriptivo. Identificamos que el *Hábito de Compra (Frecuencia)* y la *Inversión Acumulada (Monetary)* son las principales palancas de lealtad, dictando qué palancas tocar desde Marketing.

---

## 📈 Entregables Estratégicos de Negocio

El pipeline analítico genera salidas tangibles listas para ser ingeridas por el equipo de negocio:

1. **Preprocesador y Modelo en Producción (`modelos_produccion/`):**
   - El ecosistema serializado (`scaler_rfm.joblib` y `modelo_recompra_rf.joblib`) listo para su despliegue en ingeniería.
2. **Listado de Clientes Scoring:**
   - Base de clientes identificada con su probabilidad exacta de recompra y segmentada en grupos de acción CRM de alto impacto.
3. **Perspectiva de Presentación Ejecutiva:**
   - Insights de negocio visuales que escapan del argot técnico, soportados por la **Curva de Ganancias Acumuladas (Lift)**, la cual demuestra cómo focalizar presupuestos en los deciles de mayor propensión multiplica drásticamente el ROI frente a campañas aleatorias a toda la base de datos.

---

## 🛠️ Guía de Instalación y Ejecución

### 1. Requisitos Previos
Clona el repositorio e instala las dependencias de Python en un entorno virtual limpio:
```bash
python -m venv .venv
source .venv/Scripts/activate  # En Windows: .venv\Scripts\activate
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

## 📅 Hoja de Ruta Analítica (Próximos Pasos)

El proyecto asienta unos cimientos sólidos, pero la madurez analítica nos exige plantear los siguientes vectores de evolución:

* **Evolución a CLV (Customer Lifetime Value):** Transicionar del actual modelo de clasificación binaria (recompra sí/no) a un modelo de regresión puro que estime los ingresos totales en libras esperados durante el ciclo de vida del usuario.
* **Motores de Recomendación Interconectados:** Enlazar la predicción de fuga de este modelo con recomendaciones hiper-personalizadas (Cross-Sell / Up-Sell). No solo predecir *quién* se va, sino *qué* producto ofrecerle para retenerlo.
* **Refinamiento Financiero de Umbrales (Thresholds):** Ajustar el punto de corte probabilístico de decisión introduciendo una matriz de costes real de las campañas. Así se optimizaría matemáticamente el punto exacto de equilibrio entre el coste de invitar erróneamente a alguien (Falso Positivo) frente al coste de perder a un cliente valioso por no impactarle (Falso Negativo).
