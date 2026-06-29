# Cobranza Inteligente Home Credit

Sistema analítico de priorización de cobranza basado en modelamiento predictivo, scoring económico y optimización de acciones bajo restricciones operativas.

Este proyecto fue diseñado como una demo técnica profesional para banca, retail financiero, fintechs o equipos de riesgo/cobranza. El objetivo es transformar información histórica de pagos y créditos en una cartera priorizada de gestión, estimando la probabilidad de regularización, la recuperación esperada y el valor neto esperado de contactar a cada cliente/crédito.

---

## 1. Objetivo del proyecto

El proyecto responde a una pregunta de negocio concreta:

> Dado un presupuesto operativo limitado de cobranza, ¿a qué clientes conviene gestionar primero, mediante qué tipo de acción y con qué recuperación esperada?

Para ello, el pipeline construye eventos históricos de cobranza, genera variables temporales, entrena un modelo predictivo tipo Hurdle Model, estima recuperación esperada, recomienda acciones y construye un plan priorizado bajo restricciones de presupuesto/capacidad.

---

## 2. Alcance funcional

El pipeline cubre las siguientes etapas:

1. Carga de tablas crudas.
2. Construcción de eventos de cobranza y features temporales.
3. Construcción de targets prospectivos de regularización y recuperación.
4. Tratamiento de variables bajo una política CMF-friendly.
5. Entrenamiento de modelos predictivos.
6. Scoring de cartera.
7. Motor de decisión y recomendación de acciones.
8. Priorización de cartera.
9. Benchmark de estrategias.
10. Curva de recuperación esperada.
11. Optimización presupuestaria.
12. Reportes de métricas, ejecución y trazabilidad regulatoria.

---

## 3. Enfoque metodológico

### 3.1 Problema predictivo

El problema se modela como un Hurdle Model:

1. Clasificación: estimar la probabilidad de regularización dentro de un horizonte definido.
2. Regresión condicional: estimar el monto recuperado condicionado a recuperación positiva/regularización.
3. Recuperación esperada: combinar probabilidad y monto esperado para obtener una expectativa económica por cliente/crédito.

La formulación conceptual es:

```text
E[recuperación | X] = P(regulariza | X) × E[monto recuperado | regulariza = 1, X]
```

### 3.2 Modelo utilizado

La versión actual utiliza:

- `LightGBMClassifier` para la probabilidad de regularización.
- `LightGBMRegressor` para el monto recuperado condicional.
- Calibración de probabilidades.
- Separación por cliente cuando es posible para reducir fuga entre train/test.
- Transformación `log1p` del target de monto.
- Variables categóricas nativas de LightGBM, sin One-Hot Encoding productivo.
- Sin escalamiento numérico, dado que los modelos basados en árboles no requieren `StandardScaler`.

### 3.3 Métricas de negocio

El proyecto prioriza métricas útiles para cobranza, no accuracy genérica:

- ROC AUC.
- Average Precision.
- Gini.
- Brier Score.
- Precision/recall por top-k.
- Captura de recuperación por top-k.
- Recuperación esperada.
- Valor esperado neto.
- Benchmark contra estrategia aleatoria/baseline.

---

## 4. Estructura esperada del proyecto

La estructura mínima esperada es:

```text
cobranza-inteligente-home-credit/
│
├── data/
│   ├── raw/
│   │   ├── application_train.csv
│   │   ├── previous_application.csv
│   │   ├── installments_payments.csv
│   │   ├── bureau.csv                    # opcional
│   │   ├── bureau_balance.csv            # opcional
│   │   ├── POS_CASH_balance.csv          # opcional
│   │   └── credit_card_balance.csv       # opcional
│   │
│   └── processed/
│
├── outputs/
│   └── eda/
│
├── models/
│
├── scripts/
│   ├── run_pipeline.py
│   └── run_eda.py
│
├── src/
│   └── cobranza_inteligente/
│       ├── pipeline.py
│       ├── modeling.py
│       ├── decision_engine.py
│       ├── optimization.py
│       ├── eda.py
│       └── ...
│
├── requirements.txt
└── README.md
```

---

## 5. Inputs necesarios

### 5.1 Tablas obligatorias

Para correr el pipeline en modo principal sin tablas opcionales, deben existir estos archivos en `data/raw/`:

| Archivo | Descripción | Obligatorio |
|---|---|---|
| `application_train.csv` | Información principal del cliente/aplicación. | Sí |
| `previous_application.csv` | Créditos/aplicaciones previas asociadas a clientes. | Sí |
| `installments_payments.csv` | Historial de cuotas, pagos, montos y fechas relativas. | Sí |

### 5.2 Tablas opcionales

Estas tablas agregan riqueza predictiva, pero no son necesarias para correr el modo base:

| Archivo | Descripción | Obligatorio |
|---|---|---|
| `bureau.csv` | Información bureau externa. | No |
| `bureau_balance.csv` | Historial mensual de bureau. | No |
| `POS_CASH_balance.csv` | Saldos POS/cash. | No |
| `credit_card_balance.csv` | Historial de tarjeta de crédito. | No |

Para ejecutar sin estas tablas, usar:

```powershell
python scripts/run_pipeline.py --no-optional-tables
```

---

## 6. Instalación

### 6.1 Crear entorno virtual

Desde la raíz del proyecto:

```powershell
python -m venv .venv
```

Activar entorno en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 6.2 Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencias principales esperadas:

```text
pandas
numpy
scikit-learn
lightgbm
scipy
joblib
matplotlib
openpyxl
pyarrow
```

Si `lightgbm` no está instalado:

```powershell
pip install lightgbm
```

---

## 7. Cómo correr el proyecto

### 7.1 Verificar sintaxis

Antes de correr el pipeline completo:

```powershell
python -m compileall src scripts
```

### 7.2 Modo demo rápida

Este modo usa una muestra limitada de filas. Es útil para desarrollo, pruebas y validación funcional rápida.

```powershell
python scripts/run_pipeline.py --max-rows 300000 --no-optional-tables
```

Este fue el comando usado para la corrida documentada en este README.

### 7.3 Modo tablas principales completas

Este modo usa todas las filas de las tablas principales y excluye tablas opcionales.

```powershell
python scripts/run_pipeline.py --no-optional-tables
```

Advertencia: puede ser pesado para notebooks con 16 GB de RAM, especialmente por los cruces temporales y generación de features.

### 7.4 Modo completo con tablas opcionales

Este modo intenta usar todas las tablas disponibles.

```powershell
python scripts/run_pipeline.py
```

Recomendado solo si el equipo tiene suficiente memoria/RAM o si se ejecuta en servidor/cloud.

### 7.5 Guardar log de ejecución

Para ejecuciones largas, se recomienda guardar log:

```powershell
python scripts/run_pipeline.py --max-rows 300000 --no-optional-tables 2>&1 | Tee-Object -FilePath outputs/run_300k_log.txt
```

Esto muestra la salida en pantalla y además guarda un archivo de respaldo en `outputs/run_300k_log.txt`.

---

## 8. Cómo correr el EDA

El EDA permite auditar estructura de tablas, calidad de datos, missingness, outliers, integridad referencial, distribución de targets, embudo de pérdida de observaciones y alertas regulatorias.

### 8.1 EDA con muestra

```powershell
python scripts/run_eda.py --raw-dir data/raw --processed-dir data/processed --outputs-dir outputs --max-rows 300000
```

### 8.2 EDA completo

```powershell
python scripts/run_eda.py --raw-dir data/raw --processed-dir data/processed --outputs-dir outputs
```

Los outputs se guardan en:

```text
outputs/eda/
```

Archivos principales:

```text
eda_report.html
eda_resumen.json
eda_funnel.csv
eda_table_profile.csv
eda_missing_values.csv
eda_numeric_profile.csv
eda_categorical_profile.csv
eda_key_integrity.csv
eda_target_profile.csv
eda_alertas_calidad.json
```

---

## 9. Outputs generados por el pipeline

El pipeline genera archivos en `outputs/`:

| Archivo | Descripción |
|---|---|
| `cartera_priorizada.csv` | Cartera final priorizada a nivel cliente/crédito. |
| `cartera_priorizada_eventos.csv` | Priorización a nivel evento/cuota. |
| `holdout_scored.csv` | Predicciones sobre muestra holdout. |
| `metricas_modelo.csv` | Métricas del modelo en formato tabular. |
| `metricas_modelo.json` | Métricas completas del modelo. |
| `benchmark_estrategias.csv` | Comparación de estrategias de priorización. |
| `curva_recuperacion.csv` | Curva de recuperación esperada acumulada. |
| `plan_optimo_presupuesto.csv` | Plan de contactos optimizado/priorizado bajo presupuesto. |
| `resumen_optimizacion.json` | Resumen del módulo de optimización. |
| `reporte_cmf_modelo.json` | Trazabilidad de variables y política CMF-friendly. |
| `resumen_ejecucion.json` | Resumen general de la corrida. |

---

## 10. Resultados de referencia de la corrida demo

Corrida ejecutada:

```powershell
python scripts/run_pipeline.py --max-rows 300000 --no-optional-tables
```

### 10.1 Datos procesados

| Métrica | Valor |
|---|---:|
| Snapshots/eventos generados | 10.447 |
| Cartera actual priorizable | 7.193 |
| Variables numéricas finales | 66 |
| Variables categóricas finales | 11 |
| Tasa de target `regulariza_horizonte` | 27,26% |

### 10.2 Métricas predictivas

| Métrica | Valor |
|---|---:|
| ROC AUC | 0,746 |
| Average Precision | 0,364 |
| Baseline Average Precision | 0,200 |
| Lift Average Precision | 1,82 |
| Gini | 0,493 |
| Brier Score | 0,148 |
| Top 10% target rate | 44,44% |
| Top 30% recall business | 58,18% |

Interpretación: el modelo muestra señal predictiva preliminar y supera la tasa base, pero la muestra de entrenamiento es reducida; por tanto, estos resultados deben interpretarse como validación técnica de demo, no como validación productiva final.

### 10.3 Optimización presupuestaria

Con presupuesto de 500.000 unidades monetarias:

| Métrica | Valor |
|---|---:|
| Contactos seleccionados | 2.789 |
| Costo usado | 500.000 |
| Recuperación esperada | 8.662.672,71 |
| Valor neto esperado | 8.162.672,71 |
| ROI esperado promedio | 16,57 |

Nota: en esta corrida el solver MILP alcanzó límite de tiempo y el sistema utilizó una heurística greedy como fallback. Por lo tanto, este output debe interpretarse como plan priorizado bajo restricciones, no como solución MILP óptima certificada.

---

## 11. Política CMF-friendly y gobierno de variables

El pipeline incluye una política estricta de exclusión de variables sensibles, personales o proxy de alto riesgo para decisiones automatizadas de cobranza.

Variables excluidas explícitamente en la corrida demo:

```text
CNT_CHILDREN
CNT_FAM_MEMBERS
CODE_GENDER
DAYS_BIRTH
NAME_FAMILY_STATUS
```

Principios aplicados:

- No usar variables personales directas o altamente sensibles/proxy para decidir acciones de cobranza.
- No usar identificadores puros como variables predictivas.
- Mantener trazabilidad de variables excluidas y retenidas.
- Entregar recomendaciones para decisión humana.
- No ejecutar acciones irrevocables automáticamente.
- Evitar variables con riesgo de fuga temporal.

Este proyecto es una demo técnica y no constituye certificación de cumplimiento normativo.

---

## 12. Limitaciones conocidas

1. La corrida demo usa `--max-rows`, por lo que es representativa para validación técnica, no para evaluación definitiva.
2. El uso de `--no-optional-tables` reduce riqueza predictiva.
3. La muestra modelable resultante es baja para un piloto productivo.
4. El corte por filas puede romper la integridad relacional entre tablas. Para una versión más robusta se recomienda muestreo relacional por cliente/crédito.
5. En la corrida demo, el solver MILP no logró solución dentro del límite de tiempo y se usó greedy fallback.
6. El dataset Home Credit no corresponde a una operación real chilena; el proyecto debe entenderse como demostración metodológica transferible a datos reales de cobranza.

---

## 13. Posibles mejoras

Prioridades técnicas recomendadas:

1. Implementar muestreo relacional por `SK_ID_CURR` y `SK_ID_PREV` en vez de usar `head(n)` por tabla.
2. Limitar candidatos del MILP a top-k por valor esperado neto para evitar timeouts.
3. Agregar tuning más formal de LightGBM.
4. Comparar LightGBM contra Random Forest y modelos logísticos calibrados en varias semillas.
5. Revisar definición del target de recuperación monetaria para alinear regularización y recuperación parcial.
6. Incorporar tablas opcionales de manera controlada.
7. Agregar dashboard ejecutivo final en Power BI, Streamlit o Excel avanzado.


---

## 14. Comandos principales

### Activar entorno

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalar dependencias

```powershell
pip install -r requirements.txt
```

### Correr demo principal

```powershell
python scripts/run_pipeline.py --max-rows 300000 --no-optional-tables
```

### Correr tablas principales completas

```powershell
python scripts/run_pipeline.py --no-optional-tables
```

### Correr EDA demo

```powershell
python scripts/run_eda.py --raw-dir data/raw --processed-dir data/processed --outputs-dir outputs --max-rows 300000
```

### Correr EDA completo

```powershell
python scripts/run_eda.py --raw-dir data/raw --processed-dir data/processed --outputs-dir outputs
```

---

## 15. Nota final

El foco del proyecto no es solo generar predicciones, sino traducirlas en decisiones operativas accionables. La salida más importante no es la probabilidad aislada, sino la priorización económica de la cartera bajo restricciones reales de gestión.
