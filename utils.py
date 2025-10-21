"""
Utilidades y funciones auxiliares para el Sistema ML
Archivo: utils.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

# ============================================================================
# FUNCIONES DE GUARDADO Y CARGA DE MODELOS
# ============================================================================

def save_model(model, model_name, output_dir='models'):
    """
    Guarda un modelo entrenado en disco
    
    Args:
        model: Modelo scikit-learn entrenado
        model_name: Nombre del modelo (ej: 'random_forest')
        output_dir: Directorio de salida
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(output_dir, filename)
    
    joblib.dump(model, filepath)
    print(f"✓ Modelo guardado: {filepath}")
    
    return filepath

def load_model(filepath):
    """
    Carga un modelo desde disco
    
    Args:
        filepath: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    model = joblib.load(filepath)
    print(f"✓ Modelo cargado: {filepath}")
    return model

def save_scaler(scaler, output_dir='models'):
    """
    Guarda el scaler de normalización
    
    Args:
        scaler: StandardScaler ajustado
        output_dir: Directorio de salida
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scaler_{timestamp}.joblib"
    filepath = os.path.join(output_dir, filename)
    
    joblib.dump(scaler, filepath)
    print(f"✓ Scaler guardado: {filepath}")
    
    return filepath

# ============================================================================
# FUNCIONES DE REPORTES
# ============================================================================

def generate_report(results_df, output_dir='reports'):
    """
    Genera reporte en formato JSON y texto
    
    Args:
        results_df: DataFrame con resultados de modelos
        output_dir: Directorio de salida
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reporte JSON
    report_dict = {
        'timestamp': timestamp,
        'best_model': results_df.loc[results_df['R2_test'].idxmax(), 'Modelo'],
        'best_r2': float(results_df['R2_test'].max()),
        'best_rmse': float(results_df.loc[results_df['R2_test'].idxmax(), 'RMSE_test']),
        'models': results_df.to_dict('records')
    }
    
    json_path = os.path.join(output_dir, f'report_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"✓ Reporte JSON guardado: {json_path}")
    
    # Reporte TXT
    txt_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE RESULTADOS - SISTEMA ML SALUD CARDIOVASCULAR\n")
        f.write("="*70 + "\n\n")
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TABLA COMPARATIVA DE MODELOS\n")
        f.write("-"*70 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        best_idx = results_df['R2_test'].idxmax()
        f.write("MEJOR MODELO\n")
        f.write("-"*70 + "\n")
        f.write(f"Modelo: {results_df.loc[best_idx, 'Modelo']}\n")
        f.write(f"R² Score: {results_df.loc[best_idx, 'R2_test']:.4f}\n")
        f.write(f"RMSE: {results_df.loc[best_idx, 'RMSE_test']:.4f}\n")
        f.write(f"MAE: {results_df.loc[best_idx, 'MAE_test']:.4f}\n")
        
    print(f"✓ Reporte TXT guardado: {txt_path}")

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_input_data(data, expected_features):
    """
    Valida que los datos de entrada tengan las características esperadas
    
    Args:
        data: DataFrame con datos de entrada
        expected_features: Lista de características esperadas
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Verificar columnas
    missing_cols = set(expected_features) - set(data.columns)
    if missing_cols:
        return False, f"Faltan columnas: {missing_cols}"
    
    extra_cols = set(data.columns) - set(expected_features)
    if extra_cols:
        return False, f"Columnas adicionales no esperadas: {extra_cols}"
    
    # Verificar valores nulos
    if data.isnull().any().any():
        null_cols = data.columns[data.isnull().any()].tolist()
        return False, f"Valores nulos en columnas: {null_cols}"
    
    return True, "Datos válidos"

def check_data_drift(original_stats, new_data):
    """
    Detecta drift en la distribución de datos
    
    Args:
        original_stats: Estadísticas del dataset original (mean, std)
        new_data: Nuevos datos a verificar
        
    Returns:
        dict: Reporte de drift por característica
    """
    drift_report = {}
    
    for col in original_stats.keys():
        if col in new_data.columns:
            orig_mean = original_stats[col]['mean']
            orig_std = original_stats[col]['std']
            
            new_mean = new_data[col].mean()
            new_std = new_data[col].std()
            
            # Calcular z-score de la diferencia
            z_score = abs(new_mean - orig_mean) / orig_std if orig_std > 0 else 0
            
            drift_report[col] = {
                'original_mean': orig_mean,
                'new_mean': new_mean,
                'z_score': z_score,
                'drift_detected': z_score > 3  # Umbral de 3 desviaciones estándar
            }
    
    return drift_report

# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def predict_with_confidence(model, X, n_estimators=None):
    """
    Realiza predicción con intervalo de confianza (solo para Random Forest)
    
    Args:
        model: Modelo Random Forest entrenado
        X: Datos de entrada
        n_estimators: Número de árboles (si None, usa todos)
        
    Returns:
        tuple: (predicción, std_deviation)
    """
    if hasattr(model, 'estimators_'):
        # Random Forest - obtener predicciones de cada árbol
        predictions = np.array([tree.predict(X) for tree in model.estimators_])
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        return mean_pred, std_pred
    else:
        # Otros modelos - solo predicción puntual
        pred = model.predict(X)
        return pred, np.zeros_like(pred)

def batch_predict(model, scaler, data_path, output_path):
    """
    Realiza predicciones en batch sobre un archivo CSV
    
    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        data_path: Ruta al archivo CSV de entrada
        output_path: Ruta al archivo CSV de salida
    """
    print(f"Cargando datos desde: {data_path}")
    data = pd.read_csv(data_path)
    
    print(f"Registros a predecir: {len(data)}")
    
    # Normalizar
    data_scaled = scaler.transform(data)
    
    # Predecir
    predictions = model.predict(data_scaled)
    
    # Agregar predicciones al dataframe
    data['BMI_predicted'] = predictions
    
    # Categorizar BMI
    data['BMI_category'] = pd.cut(
        predictions,
        bins=[0, 18.5, 25, 30, 100],
        labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad']
    )
    
    # Guardar resultados
    data.to_csv(output_path, index=False)
    print(f"✓ Predicciones guardadas en: {output_path}")
    
    return data

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

def calculate_feature_statistics(df, target_col='BMI'):
    """
    Calcula estadísticas de características
    
    Args:
        df: DataFrame con datos
        target_col: Nombre de columna objetivo
        
    Returns:
        dict: Estadísticas por característica
    """
    stats = {}
    
    for col in df.columns:
        if col != target_col:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'correlation_with_target': float(df[col].corr(df[target_col]))
            }
    
    return stats

def get_top_features(model, feature_names, top_n=10):
    """
    Obtiene las características más importantes
    
    Args:
        model: Modelo con feature_importances_
        feature_names: Nombres de las características
        top_n: Número de características a retornar
        
    Returns:
        DataFrame con top características
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    else:
        return None

def interpret_bmi(bmi_value):
    """
    Interpreta el valor de BMI según clasificación OMS
    
    Args:
        bmi_value: Valor de BMI
        
    Returns:
        dict: Categoría e interpretación
    """
    if bmi_value < 18.5:
        category = "Bajo peso"
        risk = "Bajo"
        recommendation = "Evaluación nutricional. Aumentar ingesta calórica saludable."
    elif 18.5 <= bmi_value < 25:
        category = "Peso normal"
        risk = "Normal"
        recommendation = "Mantener hábitos saludables de alimentación y ejercicio."
    elif 25 <= bmi_value < 30:
        category = "Sobrepeso"
        risk = "Moderado"
        recommendation = "Implementar plan de reducción de peso mediante dieta y ejercicio."
    elif 30 <= bmi_value < 35:
        category = "Obesidad Grado I"
        risk = "Alto"
        recommendation = "Evaluación médica. Plan estructurado de pérdida de peso."
    elif 35 <= bmi_value < 40:
        category = "Obesidad Grado II"
        risk = "Muy Alto"
        recommendation = "Evaluación médica urgente. Considerar tratamiento especializado."
    else:
        category = "Obesidad Grado III"
        risk = "Extremo"
        recommendation = "Intervención médica inmediata. Considerar cirugía bariátrica."
    
    return {
        'bmi': round(bmi_value, 2),
        'category': category,
        'cardiovascular_risk': risk,
        'recommendation': recommendation
    }

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN AUXILIARES
# ============================================================================

def create_confusion_matrix_categorical(y_true, y_pred):
    """
    Crea matriz de confusión para categorías de BMI
    
    Args:
        y_true: Valores reales de BMI
        y_pred: Valores predichos de BMI
        
    Returns:
        DataFrame: Matriz de confusión
    """
    # Convertir valores continuos a categorías
    def categorize(bmi):
        if bmi < 18.5:
            return "Bajo peso"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Sobrepeso"
        else:
            return "Obesidad"
    
    y_true_cat = [categorize(y) for y in y_true]
    y_pred_cat = [categorize(y) for y in y_pred]
    
    categories = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
    
    confusion_matrix = pd.DataFrame(
        0,
        index=categories,
        columns=categories
    )
    
    for true, pred in zip(y_true_cat, y_pred_cat):
        confusion_matrix.loc[true, pred] += 1
    
    return confusion_matrix

def calculate_categorical_accuracy(y_true, y_pred):
    """
    Calcula precisión por categorías de BMI
    
    Args:
        y_true: Valores reales de BMI
        y_pred: Valores predichos de BMI
        
    Returns:
        float: Accuracy de clasificación categórica
    """
    def categorize(bmi):
        if bmi < 18.5:
            return 0
        elif bmi < 25:
            return 1
        elif bmi < 30:
            return 2
        else:
            return 3
    
    y_true_cat = [categorize(y) for y in y_true]
    y_pred_cat = [categorize(y) for y in y_pred]
    
    correct = sum([1 for true, pred in zip(y_true_cat, y_pred_cat) if true == pred])
    accuracy = correct / len(y_true_cat)
    
    return accuracy

# ============================================================================
# FUNCIONES DE PRUEBA Y VALIDACIÓN
# ============================================================================

def test_system():
    """
    Ejecuta pruebas del sistema completo
    """
    print("="*70)
    print("EJECUTANDO PRUEBAS DEL SISTEMA")
    print("="*70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Importación de librerías
    tests_total += 1
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn
        import streamlit
        print("✓ Test 1: Importación de librerías - PASSED")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Test 1: Importación de librerías - FAILED: {e}")
    
    # Test 2: Creación de datos sintéticos
    tests_total += 1
    try:
        data = pd.DataFrame({
            'HighBP': np.random.randint(0, 2, 100),
            'HighChol': np.random.randint(0, 2, 100),
            'Age': np.random.randint(1, 14, 100),
            'BMI': np.random.normal(28, 6, 100)
        })
        assert len(data) == 100
        assert 'BMI' in data.columns
        print("✓ Test 2: Creación de datos sintéticos - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 2: Creación de datos sintéticos - FAILED: {e}")
    
    # Test 3: Normalización
    tests_total += 1
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = data.drop(columns=['BMI'])
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape
        assert abs(X_scaled.mean()) < 1e-10  # Media cercana a 0
        print("✓ Test 3: Normalización - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 3: Normalización - FAILED: {e}")
    
    # Test 4: Entrenamiento de modelo simple
    tests_total += 1
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, data['BMI'], test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert score > 0  # R² debe ser positivo
        print(f"✓ Test 4: Entrenamiento de modelo - PASSED (R²={score:.4f})")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 4: Entrenamiento de modelo - FAILED: {e}")
    
    # Test 5: Predicción
    tests_total += 1
    try:
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert not np.isnan(predictions).any()
        print("✓ Test 5: Predicción - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 5: Predicción - FAILED: {e}")
    
    # Test 6: Cálculo de métricas
    tests_total += 1
    try:
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        assert mae >= 0
        assert -1 <= r2 <= 1
        print(f"✓ Test 6: Cálculo de métricas - PASSED (MAE={mae:.2f}, R²={r2:.4f})")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 6: Cálculo de métricas - FAILED: {e}")
    
    # Test 7: Interpretación de BMI
    tests_total += 1
    try:
        interpretation = interpret_bmi(28.5)
        assert 'category' in interpretation
        assert 'cardiovascular_risk' in interpretation
        assert interpretation['category'] == "Sobrepeso"
        print("✓ Test 7: Interpretación de BMI - PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 7: Interpretación de BMI - FAILED: {e}")
    
    # Test 8: Guardado de modelo
    tests_total += 1
    try:
        save_model(model, 'test_model', output_dir='test_models')
        assert os.path.exists('test_models')
        print("✓ Test 8: Guardado de modelo - PASSED")
        tests_passed += 1
        
        # Limpiar
        import shutil
        shutil.rmtree('test_models')
    except Exception as e:
        print(f"✗ Test 8: Guardado de modelo - FAILED: {e}")
    
    # Resumen
    print("\n" + "="*70)
    print(f"RESUMEN DE PRUEBAS: {tests_passed}/{tests_total} PASSED")
    print("="*70)
    
    if tests_passed == tests_total:
        print("✓ Todos los tests pasaron exitosamente")
        return True
    else:
        print(f"✗ {tests_total - tests_passed} tests fallaron")
        return False

# ============================================================================
# FUNCIÓN DE DEMO
# ============================================================================

def run_demo():
    """
    Ejecuta una demostración del sistema con datos sintéticos
    """
    print("\n" + "="*70)
    print("DEMOSTRACIÓN DEL SISTEMA ML")
    print("="*70 + "\n")
    
    # 1. Crear datos sintéticos
    print("1. Generando datos sintéticos...")
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'HighBP': np.random.randint(0, 2, n_samples),
        'HighChol': np.random.randint(0, 2, n_samples),
        'Smoker': np.random.randint(0, 2, n_samples),
        'Stroke': np.random.randint(0, 2, n_samples),
        'Diabetes': np.random.randint(0, 3, n_samples),
        'PhysActivity': np.random.randint(0, 2, n_samples),
        'Fruits': np.random.randint(0, 2, n_samples),
        'Veggies': np.random.randint(0, 2, n_samples),
        'HvyAlcoholConsump': np.random.randint(0, 2, n_samples),
        'AnyHealthcare': np.random.randint(0, 2, n_samples),
        'GenHlth': np.random.randint(1, 6, n_samples),
        'MentHlth': np.random.randint(0, 31, n_samples),
        'PhysHlth': np.random.randint(0, 31, n_samples),
        'Sex': np.random.randint(0, 2, n_samples),
        'Age': np.random.randint(1, 14, n_samples),
        'Education': np.random.randint(1, 7, n_samples),
        'Income': np.random.randint(1, 9, n_samples),
    })
    
    # Generar BMI con relaciones no lineales
    data['BMI'] = (
        25 + 
        data['Age'] * 0.3 + 
        data['HighBP'] * 2 + 
        data['PhysActivity'] * (-1.5) +
        data['GenHlth'] * 1.2 +
        np.random.normal(0, 3, n_samples)
    )
    
    print(f"   ✓ Generados {n_samples} registros sintéticos")
    
    # 2. Preprocesar
    print("\n2. Preprocesando datos...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    X = data.drop(columns=['BMI'])
    y = data['BMI']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"   ✓ Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    
    # 3. Entrenar modelos
    print("\n3. Entrenando modelos...")
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"   ✓ Regresión Lineal entrenada (R²={lr_score:.4f})")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)
    print(f"   ✓ Random Forest entrenado (R²={rf_score:.4f})")
    
    # 4. Evaluar
    print("\n4. Evaluando modelos...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)
    
    print("\n   Regresión Lineal:")
    print(f"      MAE: {mean_absolute_error(y_test, lr_pred):.4f}")
    print(f"      RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.4f}")
    print(f"      R²: {lr_score:.4f}")
    
    print("\n   Random Forest:")
    print(f"      MAE: {mean_absolute_error(y_test, rf_pred):.4f}")
    print(f"      RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f}")
    print(f"      R²: {rf_score:.4f}")
    
    # 5. Predicción de ejemplo
    print("\n5. Predicción de ejemplo...")
    ejemplo = X_test[0:1]
    bmi_real = y_test.iloc[0]
    bmi_pred = rf.predict(ejemplo)[0]
    
    print(f"   BMI Real: {bmi_real:.2f} kg/m²")
    print(f"   BMI Predicho: {bmi_pred:.2f} kg/m²")
    print(f"   Error: {abs(bmi_real - bmi_pred):.2f} kg/m²")
    
    interpretation = interpret_bmi(bmi_pred)
    print(f"\n   Categoría: {interpretation['category']}")
    print(f"   Riesgo: {interpretation['cardiovascular_risk']}")
    print(f"   Recomendación: {interpretation['recommendation']}")
    
    print("\n" + "="*70)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*70)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal de utilidades
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python utils.py [test|demo]")
        print("\nOpciones:")
        print("  test  - Ejecuta pruebas del sistema")
        print("  demo  - Ejecuta demostración con datos sintéticos")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        test_system()
    elif command == 'demo':
        run_demo()
    else:
        print(f"Comando no reconocido: {command}")
        print("Comandos disponibles: test, demo")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()