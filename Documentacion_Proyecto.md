# Proyecto Analítico: Predicción de Recompra y Estrategia de Fidelización

**Rol:** Senior Data Scientist
**Objetivo:** Desarrollar un modelo que prediga la probabilidad de que un cliente vuelva a comprar, permitiendo al equipo de negocio dirigir sus campañas de retención y fidelización de manera rentable e inteligente.

A lo largo de este documento detallamos el proceso end-to-end, desde la obtención y purificación de los datos brutos hasta la implementación del motor predictivo, asegurando siempre que cada decisión matemática tenga una traducción directa y accionable para el negocio.

---

## 1. Introducción y Contexto del Negocio

Como principal Data Scientist de la compañía minorista online, se nos presentó el reto de trabajar con el histórico de transacciones del conjunto de datos **Online Retail II** (diciembre 2009 - diciembre 2011). 

El objetivo primordial no es simplemente crear un algoritmo predictivo, sino **transformar datos en decisiones comerciales**. Debemos identificar qué clientes tienen alta probabilidad de volver a comprar (para afianzar su lealtad con incentivos premium) y qué clientes están en riesgo de fuga (para impactarlos con campañas de reactivación eficientes). Siguiendo el rigor analítico, evitamos enfoques puramente teóricos y centramos la evaluación en métricas de coste de oportunidad y Retorno de Inversión (ROI).

---

## 2. Fase 1: Limpieza de Datos y Construcción del "Pasado"

Los datos crudos, a nivel de ticket, requerían un proceso de limpieza exhaustivo para pivotarlos a nivel de cliente.

### 2.1. Depuración del Histórico
1. **Eliminación de "Guest Checkouts":** Se aislaron los registros con `Customer ID` nulos. Al ser compras anónimas, es matemáticamente imposible construir un historial longitudinal o etiquetar si el cliente recompró en el futuro. Conservarlos solo inyectaría ruido, por lo que fueron eliminados sistemáticamente.
2. **Gestión Bifurcada de Devoluciones:** Limpiamos errores del sistema y transacciones anómalas. Las cantidades negativas (devoluciones) no se restaron burdamente del volumen de compra, sino que fueron excluidas del cálculo de facturación pura (`Monetary`) para no falsear los ingresos reales. Paralelamente, se utilizaron de forma inteligente para derivar un "Ratio de Devoluciones", un potente indicador predictivo de insatisfacción.

### 2.2. Diseño del Marco Temporal (Prevención de Data Leakage)
Para evitar el error más grave en Machine Learning ("Fuga de Datos" o *Data Leakage*), establecemos una fecha de corte estricta (**junio de 2011**) idéntica para todos los usuarios:
* **Ventana de Observación (Input):** Todo lo ocurrido hasta junio de 2011. Usada exclusivamente para calcular el historial de consumo del cliente.
* **Ventana de Predicción (Target):** Los últimos meses del dataset. Usada como "oráculo" para ver quién efectivamente volvió a comprar (Target = 1) y quién se fugó (Target = 0).

Esta división generó un target excepcionalmente saludable (aproximadamente un **52% de retención frente a un 48% de fugas**), proporcionando un terreno ideal para el modelado algorítmico sin sufrir los problemas crónicos de las clases desbalanceadas.

---

## 3. Ingeniería de Características: El ADN del Cliente

Transformamos las facturas en un dataset donde cada fila es un cliente único perfilado mediante *Feature Engineering*:

1. **Variables RFM Expandidas:** Recencia (días desde la última compra), Frecuencia (número de tickets) y Valor Monetario (gasto total).
2. **Estilo de Vida (NLP):** Categorización temática aplicando procesamiento de lenguaje natural (TF-IDF y K-Means) sobre las descripciones de los artículos.
3. **Dinámicas de Compra:** Frecuencia de devoluciones, variedad de artículos, y el *Momentum* (Tendencia de gasto reciente).

### 3.1. La Variable Estrella: Índice de Salud (Recency-Tenure Ratio)
El tiempo sin comprar (`Recency`) es engañoso: 60 días inactivo es grave para alguien que compra semanalmente, pero normal para un mayorista. Para solucionarlo, creamos el **Recency-Tenure Ratio** (Recencia / Antigüedad). Si el ratio se acerca a 1, significa que el cliente ha pasado casi todo su ciclo vital inactivo, disparando las alarmas de riesgo.

![Heatmap Correlaciones](images/1_EDA_y_Limpieza_img_2.png)
*Insight Visual:* Confirmamos matemáticamente la fuerte correlación positiva entre `Frequency` y `Monetary`. La rentabilidad sostenida se basa en la recurrencia, no en una única cesta masiva.

---

## 4. Diagnóstico de Negocio (Business Storytelling)

Antes de predecir el futuro, diagnosticamos el macroentorno comercial:

### 4.1. Estacionalidad Extrema (El Efecto Q4)
![Evolución Mensual](images/2_Modelo_Recompra_img_0.png)
La facturación de la empresa tiene un pico monumental en el último trimestre. Perder a un cliente justo antes del Q4 tiene un coste de oportunidad enorme. Las campañas predictivas de retención deben ser agresivas entre los meses de agosto y octubre.

### 4.2. Análisis de Pareto (Concentración del Riesgo)
![Curva de Pareto](images/2_Modelo_Recompra_img_1.png)
Confirmamos que apenas un **21.6% de los clientes genera el 80% de los ingresos**. Esto justifica nuestra arquitectura predictiva: no podemos lanzar campañas de descuento masivas "para todos" (destruyendo el margen), sino usar el algoritmo como radar para detectar y blindar exclusivamente a este núcleo VIP.

---

## 5. El Motor Predictivo (Machine Learning)

Sometimos a cuatro familias algorítmicas (Regresión Logística, KNN, Random Forest, XGBoost) a una rigurosa competición matemática.

### 5.1. Selección de Métricas (AUC-ROC y Matriz de Confusión)
Por indicación directa de la casuística de negocio, evitamos guiarnos ciegamente por el *Accuracy* global. Utilizamos el **Área bajo la Curva ROC (AUC)** como métrica reina, ya que evalúa la capacidad pura del modelo para separar a los recompradores de los fugitivos. Paralelamente, utilizamos la **Matriz de Confusión** como validador final de los aciertos y errores tangibles del algoritmo.

### 5.2. El Campeón: Random Forest Optimizado
Tras una optimización exhaustiva mediante *GridSearchCV*, el modelo **Random Forest** se coronó como el campeón indiscutible, demostrando la mayor estabilidad, generalización y AUC frente al sobreajuste detectado inicialmente en arquitecturas como XGBoost.

### 5.3. Estrategia de Umbrales (Precision vs Recall)
Adecuamos el modelo de probabilidades a la cuenta de resultados de Marketing:
Basándonos en nuestra curva de Precisión-Recall, abandonamos el corte clásico del 0.50 y establecimos **dos umbrales financieros**:
* **Umbral 0.70 (Alta Propensión):** Maximiza la *Precisión*. Usado para acciones caras (ej. regalos físicos), donde un Falso Positivo (gastar en alguien que no iba a comprar de todas formas) destruye el ROI.
* **Umbral 0.40 (Acciones de Reactivación):** Maximiza el *Recall*. Usado para campañas de bajo coste (ej. email automatizado), donde atrapar un Verdadero Positivo (recuperar al cliente) compensa con creces los envíos inútiles.

---

## 6. Interpretabilidad (XAI): Las Palancas de Retención

Un modelo "caja negra" no es útil para la dirección. Utilizando Valores **SHAP** sobre el Random Forest, abrimos el motor para extraer de forma transparente qué variables detonan realmente las decisiones del algoritmo:

![SHAP Values](images/2_Modelo_Recompra_img_10.png)

1. **El Hábito como Rey (`Frequency`):** El predictor de lealtad más potente. Fomentar visitas recurrentes genera un anclaje a largo plazo inquebrantable.
2. **Valor Bruto Acumulado (`Monetary`):** Barrera psicológica de salida. La alta inversión económica pasada reduce drásticamente el riesgo de fuga.
3. **El Asesino Silencioso (`Recency_Tenure_Ratio`):** El principal factor de riesgo. Si esta métrica de inactividad relativa aumenta, la probabilidad matemática de recuperación se hunde de inmediato.

---

## 7. Hoja de Ruta Analítica (Madurez y Próximos Pasos)

Este proyecto representa un Producto Mínimo Viable analítico sumamente robusto. Sin embargo, proponemos los siguientes vectores de evolución para demostrar madurez analítica en el ecosistema comercial de la empresa:

1. **Customer Lifetime Value (CLV):** Transicionar del actual modelo de clasificación binaria (recompra sí/no) a un modelo de regresión puro que estime directamente los ingresos totales (en libras) esperados a lo largo del ciclo de vida del usuario.
2. **Motores de Recomendación (Cross-Sell Interconectado):** Si el modelo detecta que un cliente VIP está en riesgo crítico de fuga, no basta con enviar una alerta genérica. El siguiente paso tecnológico es conectar el algoritmo con un sistema de filtrado colaborativo que decida automáticamente los 3 artículos "gancho" específicos con mayor probabilidad de retener a *ese* usuario concreto.
3. **A/B Testing en Vivo:** Desplegar las predicciones actuales sobre un segmento aislado de la base de datos (grupo de tratamiento) y comparar la retención económica contra un grupo de control. Esta será la prueba irrefutable del Lift generado por el algoritmo.
