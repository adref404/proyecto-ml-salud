# 🫀 Sistema de Machine Learning para Predicción de Salud Cardiovascular

## 📋 Información del Proyecto

**Estudiante:** Celadita  
**Grupo:** A-C (Problema de Salud)  
**Tipo de Problema:** Regresión Supervisada  
**Variable Objetivo:** BMI (Índice de Masa Corporal)  
**Dataset:** Heart Disease Health Indicators (Kaggle CDC)

---

## 🎯 Objetivo

Desarrollar un sistema completo de Machine Learning Supervisado que compare el desempeño de 4 modelos de regresión (2 de caja blanca y 2 de caja negra) para predecir valores de BMI basándose en indicadores de salud cardiovascular.

---

## 📦 Estructura del Proyecto

```
proyecto-ml-salud/
│
├── README.md                                    ✓ Documentación principal
├── requirements.txt                             ✓ Dependencias
├── main.py                                      ✓ Pipeline completo
├── app_streamlit.py                             ✓ Interfaz GUI
├── utils.py                                     ✓ Utilidades
│
├── heart_disease_health_indicators.csv          ✓ Dataset
│
├── docs/
│   ├── informe_tecnico.md                       ✓ Informe detallado
│   └── presentacion.pdf                         ○ Opcional
│
├── outputs/
│   ├── metricas_comparacion.png                 ✓ Gráfico comparativo
│   ├── predicciones_vs_reales.png               ✓ Scatter plot
│   ├── distribucion_errores.png                 ✓ Histograma
│   ├── importancia_caracteristicas.png          ✓ Feature importance
│   └── resultados_modelos.csv                   ✓ Tabla de resultados
│
├── models/                                      ○ Modelos guardados (opcional)
│   ├── random_forest.joblib
│   └── scaler.joblib
│
└── screenshots/                                 ○ Capturas GUI (opcional)
    ├── gui_dataset.png
    ├── gui_training.png
    └── gui_results.png
```

Leyenda:

✓ = Obligatorio
○ = Opcional/Recomendado

---

## 🛠️ Requisitos e Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 4GB RAM mínimo (recomendado 8GB)
- Espacio en disco: 500MB

### Instalación de Dependencias

1. **Clonar o descargar el proyecto:**

```bash
cd proyecto-ml-salud
```

2. **Crear entorno virtual (recomendado):**

```bash
# Windows
py -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

### Contenido de `requirements.txt`:

```txt
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
streamlit==1.25.0
```

---

## 📊 Obtención del Dataset

1. Ir a Kaggle: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

2. Descargar el archivo `heart_disease_health_indicators.csv`

3. Colocar el archivo en la carpeta raíz del proyecto

**Información del Dataset:**
- **Registros:** 253,680
- **Variables:** 22
- **Fuente:** CDC (Centers for Disease Control and Prevention)
- **Variable Objetivo:** BMI (numérica continua)

---

## 🚀 Ejecución del Sistema

### Opción 1: Script de Línea de Comandos

Ejecutar el pipeline completo:

```bash
python main.py
```

**Salidas generadas:**
- Métricas en consola
- Gráficos guardados en carpeta `outputs/`
- Archivo CSV con resultados: `resultados_modelos.csv`

### Opción 2: Interfaz Gráfica (Streamlit)

Iniciar la aplicación web interactiva:

```bash
streamlit run app_streamlit.py
```

**Funcionalidades de la GUI:**
- ✅ Carga interactiva del dataset
- ✅ Visualización de datos exploratoria
- ✅ Configuración de parámetros
- ✅ Entrenamiento con barra de progreso
- ✅ Comparación visual de modelos
- ✅ Predicción individual en tiempo real

La aplicación se abrirá automáticamente en `http://localhost:8501`

---

## 🧠 Modelos Implementados

### Modelos de Caja Blanca (Interpretables)

1. **Regresión Lineal**
   - Algoritmo: OLS (Ordinary Least Squares)
   - Ventaja: Interpretación directa de coeficientes
   - Limitación: Asume relación lineal

2. **Árbol de Decisión Regressor**
   - Algoritmo: CART (Classification and Regression Trees)
   - Ventaja: Interpretación visual clara
   - Limitación: Propenso a sobreajuste

### Modelos de Caja Negra (Mayor Complejidad)

3. **Random Forest Regressor**
   - Algoritmo: Ensamble de árboles
   - Parámetros: 100 estimadores (default)
   - Ventaja: Robusto y preciso

4. **Support Vector Regressor (SVR)**
   - Kernel: RBF (Radial Basis Function)
   - Ventaja: Efectivo en alta dimensionalidad
   - Limitación: Alto costo computacional

---

## 📈 Métricas de Evaluación

El sistema calcula las siguientes métricas para cada modelo:

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **MAE** | Mean Absolute Error | Error promedio absoluto (menor es mejor) |
| **MSE** | Mean Squared Error | Error cuadrático medio (penaliza errores grandes) |
| **RMSE** | Root Mean Squared Error | Raíz del MSE, en las mismas unidades del target |
| **R²** | Coeficiente de Determinación | Proporción de varianza explicada (0 a 1, mayor es mejor) |

---

## 🔬 Metodología

### 1. Preprocesamiento

```python
# Pasos aplicados:
1. Eliminación de duplicados
2. Tratamiento de outliers (método IQR)
3. Normalización con StandardScaler
4. División train/test (80/20)
```

### 2. Validación

- **Método:** K-Fold Cross Validation
- **K:** 5 pliegues
- **Métrica:** R² Score
- **Propósito:** Evaluar estabilidad del modelo

### 3. Entrenamiento

- **Hiperparámetros:** Configuración por defecto
- **Semilla aleatoria:** 42 (reproducibilidad)
- **Paralelización:** n_jobs=-1 (usa todos los cores)

---

## 📊 Resultados Esperados

### Tabla Comparativa (Ejemplo)

| Modelo | Tipo | MAE | RMSE | R² |
|--------|------|-----|------|----|
| Regresión Lineal | Caja Blanca | 4.91 | 6.24 | 0.3087 |
| Árbol de Decisión | Caja Blanca | 3.56 | 5.30 | 0.4998 |
| **Random Forest** | **Caja Negra** | **3.02** | **4.38** | **0.6583** |
| SVR | Caja Negra | 4.61 | 6.00 | 0.3602 |

### Interpretación

🏆 **Mejor Modelo:** Random Forest
- **R² = 0.6583** → Explica 65.83% de la varianza del BMI
- **RMSE = 4.38** → Error promedio de ~4.38 puntos de BMI
- **Conclusión:** Balance óptimo entre precisión y generalización

---

## 🎯 Uso de la Interfaz GUI

### Flujo de Trabajo

1. **Pestaña "Dataset"**
   - Cargar el archivo CSV
   - Explorar estadísticas básicas
   - Visualizar distribución de BMI

2. **Pestaña "Preprocesamiento"**
   - Revisar pasos de limpieza
   - Ejecutar preprocesamiento
   - Confirmar división de datos

3. **Pestaña "Entrenamiento"**
   - Revisar configuración de modelos
   - Iniciar entrenamiento (con barra de progreso)
   - Ver resumen de validación cruzada

4. **Pestaña "Resultados"**
   - Comparar métricas en tabla
   - Visualizar gráficos comparativos
   - Analizar predicciones vs valores reales
   - Ver importancia de características

5. **Pestaña "Predicción"**
   - Ingresar valores de características
   - Obtener predicción de BMI
   - Ver interpretación clínica

---

## 🖼️ Visualizaciones Generadas

### 1. Comparación de Métricas
- Gráficos de barras para MAE, MSE, RMSE, R²
- Diferenciación por colores (caja blanca vs caja negra)

### 2. Predicciones vs Valores Reales
- Scatter plot con línea de predicción perfecta
- Permite identificar patrones de error

### 3. Distribución de Errores
- Histograma de errores de predicción
- Verifica normalidad de residuos

### 4. Importancia de Características
- Top 15 features más importantes (Random Forest)
- Útil para interpretación del modelo

---

## 🔍 Interpretación Clínica del BMI

La interfaz proporciona interpretación automática según rangos OMS:

| Rango BMI | Categoría | Recomendación |
|-----------|-----------|---------------|
| < 18.5 | Bajo peso | Evaluación nutricional |
| 18.5 - 24.9 | Peso normal | Mantener hábitos saludables |
| 25.0 - 29.9 | Sobrepeso | Plan de manejo de peso |
| ≥ 30.0 | Obesidad | Evaluación médica integral |

---

## 🐛 Solución de Problemas

### Error: "FileNotFoundError: heart_disease_health_indicators.csv"

**Solución:** Descargar el dataset de Kaggle y colocarlo en la carpeta raíz.

### Error: "ModuleNotFoundError: No module named 'sklearn'"

**Solución:** 
```bash
pip install scikit-learn
```

### Error: "Streamlit no abre en el navegador"

**Solución:**
```bash
# Abrir manualmente en:
http://localhost:8501
```

### Advertencia: "MemoryError"

**Solución:** Reducir el tamaño del dataset o aumentar RAM disponible.

### Entrenamiento muy lento (SVR)

**Causa:** SVR es computacionalmente costoso con datasets grandes.  
**Solución:** Esperar o reducir tamaño del dataset de entrenamiento.

---

## 📝 Variables del Dataset

### Variables Independientes (Features)

| Variable | Descripción | Tipo |
|----------|-------------|------|
| HighBP | Presión arterial alta | Binaria (0/1) |
| HighChol | Colesterol alto | Binaria (0/1) |
| CholCheck | Chequeo de colesterol en últimos 5 años | Binaria (0/1) |
| Smoker | Fumador (más de 100 cigarrillos) | Binaria (0/1) |
| Stroke | Historial de derrame cerebral | Binaria (0/1) |
| Diabetes | Diabetes (0=no, 1=pre-diabetes, 2=diabetes) | Categórica |
| PhysActivity | Actividad física en últimos 30 días | Binaria (0/1) |
| Fruits | Consume frutas 1+ vez al día | Binaria (0/1) |
| Veggies | Consume vegetales 1+ vez al día | Binaria (0/1) |
| HvyAlcoholConsump | Consumo excesivo de alcohol | Binaria (0/1) |
| AnyHealthcare | Tiene seguro médico | Binaria (0/1) |
| NoDocbcCost | No visitó médico por costo | Binaria (0/1) |
| GenHlth | Salud general (1=excelente, 5=pobre) | Ordinal |
| MentHlth | Días de mala salud mental (últimos 30) | Numérica (0-30) |
| PhysHlth | Días de mala salud física (últimos 30) | Numérica (0-30) |
| DiffWalk | Dificultad para caminar/subir escaleras | Binaria (0/1) |
| Sex | Sexo (0=femenino, 1=masculino) | Binaria (0/1) |
| Age | Categoría de edad (1-13) | Ordinal |
| Education | Nivel educativo (1-6) | Ordinal |
| Income | Categoría de ingresos (1-8) | Ordinal |
| HeartDiseaseorAttack | Historial de enfermedad cardíaca | Binaria (0/1) |

### Variable Dependiente (Target)

| Variable | Descripción | Tipo | Rango |
|----------|-------------|------|-------|
| **BMI** | Índice de Masa Corporal (peso/altura²) | Numérica continua | 12-98 kg/m² |

---

## 📚 Referencias y Recursos

### Librerías Utilizadas

- **Pandas:** Manipulación de datos tabulares
- **NumPy:** Operaciones numéricas y matrices
- **Scikit-learn:** Algoritmos de ML y preprocessing
- **Matplotlib/Seaborn:** Visualización de datos
- **Streamlit:** Framework para aplicaciones web

### Documentación

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Dataset en Kaggle](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)

### Papers Relacionados

- Breiman, L. (2001). "Random Forests". Machine Learning.
- Vapnik, V. (1995). "The Nature of Statistical Learning Theory".

---

## 🎓 Criterios de Evaluación Académica

### Cumplimiento de Requisitos

✅ Dataset de salud de Kaggle (no imágenes)  
✅ Variable objetivo numérica continua (BMI)  
✅ Preprocesamiento completo (limpieza, normalización)  
✅ División 80/20 train/test  
✅ Validación cruzada K=5  
✅ 2 modelos caja blanca + 2 caja negra  
✅ Hiperparámetros por defecto  
✅ Métricas: MAE, MSE, RMSE, R²  
✅ Tabla comparativa de modelos  
✅ Interfaz GUI funcional (Streamlit)  
✅ Visualizaciones de resultados  
✅ Código comentado y documentado  

---

## 💡 Conclusiones del Proyecto

### Hallazgos Principales

1. **Superioridad de ensambles:** Random Forest superó significativamente a modelos individuales.

2. **Limitaciones de linealidad:** La Regresión Lineal (R²=0.31) confirma que las relaciones entre variables de salud son predominantemente no lineales.

3. **Costo-beneficio:** SVR no justifica su alto costo computacional frente a Random Forest.

4. **Interpretabilidad vs Precisión:** Existe un trade-off claro entre la interpretabilidad de modelos simples y la precisión de modelos complejos.

### Aplicación Práctica

- El sistema puede integrarse en aplicaciones de screening de salud poblacional
- La predicción de BMI permite identificar individuos en riesgo cardiovascular
- Las características más importantes pueden guiar campañas de prevención

### Trabajo Futuro

1. **Optimización de hiperparámetros** mediante Grid Search o Bayesian Optimization
2. **Feature engineering** para crear variables derivadas más informativas
3. **Modelos avanzados** como XGBoost, LightGBM o Deep Learning
4. **Interpretabilidad** con SHAP values para explicar predicciones individuales
5. **Despliegue en producción** con Docker y APIs REST

---

## 👤 Autor

**Estudiante:** Celadita  
**Curso:** Machine Learning Supervisado  
**Año:** 2025  
**Institución:** [Universidad]

---

## 📄 Licencia

Este proyecto es de uso académico y educativo.

---

## 🙏 Agradecimientos

- Kaggle y CDC por proporcionar el dataset
- Comunidad de Scikit-learn por las herramientas de ML
- Streamlit por facilitar la creación de interfaces web

---

## 📞 Soporte

Para preguntas o problemas:
1. Revisar la sección "Solución de Problemas"
2. Verificar la documentación de las librerías
3. Consultar los foros de Kaggle y Stack Overflow

---

**Última actualización:** Octubre 2025  
**Versión:** 1.0.0