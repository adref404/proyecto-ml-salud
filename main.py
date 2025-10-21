"""
Sistema de Machine Learning Supervisado para Predicción de Salud Cardiovascular
Estudiante: Celadita (Problema de Salud - Grupo A-C)
Dataset: Heart Disease Health Indicators (Kaggle)
"""

# ============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 2. CARGA Y EXPLORACIÓN DEL DATASET
# ============================================================================

def load_and_explore_data(filepath):
    """
    Carga el dataset y realiza exploración inicial
    
    Args:
        filepath: Ruta del archivo CSV
        
    Returns:
        DataFrame con los datos cargados
    """
    print("="*70)
    print("CARGANDO DATASET")
    print("="*70)
    
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # Información básica
    print(f"\nDimensiones del dataset: {df.shape}")
    print(f"Número de filas: {df.shape[0]:,}")
    print(f"Número de columnas: {df.shape[1]}")
    
    # Primeras filas
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    # Información de tipos de datos
    print("\nInformación de columnas:")
    print(df.info())
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    return df


# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================

def preprocess_data(df, target_column='BMI'):
    """
    Preprocesa los datos: limpieza, tratamiento de outliers, normalización
    
    Args:
        df: DataFrame original
        target_column: Nombre de la columna objetivo
        
    Returns:
        X: Features preprocesadas
        y: Variable objetivo
        scaler: Objeto StandardScaler ajustado
    """
    print("\n" + "="*70)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*70)
    
    # Copiar dataframe
    df_clean = df.copy()
    
    # 1. Eliminar duplicados
    duplicates = df_clean.duplicated().sum()
    print(f"\nDuplicados encontrados: {duplicates}")
    df_clean = df_clean.drop_duplicates()
    
    # 2. Verificar valores nulos
    nulls = df_clean.isnull().sum().sum()
    print(f"Valores nulos totales: {nulls}")
    
    # 3. Tratamiento de outliers en la variable objetivo usando IQR
    Q1 = df_clean[target_column].quantile(0.25)
    Q3 = df_clean[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nTratamiento de outliers en {target_column}:")
    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Límites: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    outliers = ((df_clean[target_column] < lower_bound) | 
                (df_clean[target_column] > upper_bound)).sum()
    print(f"Outliers detectados: {outliers}")
    
    # Filtrar outliers
    df_clean = df_clean[
        (df_clean[target_column] >= lower_bound) & 
        (df_clean[target_column] <= upper_bound)
    ]
    print(f"Registros después de limpieza: {len(df_clean):,}")
    
    # 4. Separar features y target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    print(f"\nShape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    
    # 5. Normalización con StandardScaler
    print("\nAplicando StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Normalización completada ✓")
    
    return X_scaled, y, scaler


# ============================================================================
# 4. DIVISIÓN DE DATOS Y VALIDACIÓN CRUZADA
# ============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en entrenamiento y prueba
    
    Args:
        X: Features
        y: Target
        test_size: Proporción para prueba (default: 0.2 = 20%)
        random_state: Semilla para reproducibilidad
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "="*70)
    print("DIVISIÓN DE DATOS")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nConjunto de entrenamiento: {len(X_train):,} registros ({(1-test_size)*100:.0f}%)")
    print(f"Conjunto de prueba: {len(X_test):,} registros ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# 5. ENTRENAMIENTO DE MODELOS
# ============================================================================

def train_models(X_train, y_train):
    """
    Entrena 4 modelos: 2 caja blanca, 2 caja negra
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        
    Returns:
        Diccionario con modelos entrenados
    """
    print("\n" + "="*70)
    print("ENTRENAMIENTO DE MODELOS")
    print("="*70)
    
    models = {}
    
    # MODELOS DE CAJA BLANCA
    print("\n--- MODELOS DE CAJA BLANCA ---")
    
    # 1. Regresión Lineal
    print("\n1. Entrenando Regresión Lineal...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    print("   ✓ Completado")
    
    # Validación cruzada
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    print(f"   R² con CV (K=5): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 2. Árbol de Decisión
    print("\n2. Entrenando Árbol de Decisión...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    print("   ✓ Completado")
    
    # Validación cruzada
    cv_scores = cross_val_score(dt, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    print(f"   R² con CV (K=5): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # MODELOS DE CAJA NEGRA
    print("\n--- MODELOS DE CAJA NEGRA ---")
    
    # 3. Random Forest
    print("\n3. Entrenando Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print("   ✓ Completado")
    
    # Validación cruzada
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    print(f"   R² con CV (K=5): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 4. Support Vector Regressor
    print("\n4. Entrenando SVR...")
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    models['SVR'] = svr
    print("   ✓ Completado")
    
    # Validación cruzada
    cv_scores = cross_val_score(svr, X_train, y_train, cv=5, 
                                 scoring='r2', n_jobs=-1)
    print(f"   R² con CV (K=5): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\n✓ Todos los modelos entrenados exitosamente")
    
    return models


# ============================================================================
# 6. EVALUACIÓN DE MODELOS
# ============================================================================

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evalúa todos los modelos y calcula métricas
    
    Args:
        models: Diccionario con modelos entrenados
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        
    Returns:
        DataFrame con resultados
    """
    print("\n" + "="*70)
    print("EVALUACIÓN DE MODELOS")
    print("="*70)
    
    results = []
    
    for name, model in models.items():
        print(f"\nEvaluando {name}...")
        
        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Métricas en entrenamiento
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        # Métricas en prueba
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        # Guardar resultados
        results.append({
            'Modelo': name,
            'Tipo': 'Caja Blanca' if name in ['Linear Regression', 'Decision Tree'] else 'Caja Negra',
            'MAE_train': train_metrics['mae'],
            'MSE_train': train_metrics['mse'],
            'RMSE_train': train_metrics['rmse'],
            'R2_train': train_metrics['r2'],
            'MAE_test': test_metrics['mae'],
            'MSE_test': test_metrics['mse'],
            'RMSE_test': test_metrics['rmse'],
            'R2_test': test_metrics['r2']
        })
        
        # Imprimir métricas
        print(f"  Entrenamiento - MAE: {train_metrics['mae']:.4f}, "
              f"RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"  Prueba        - MAE: {test_metrics['mae']:.4f}, "
              f"RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    return results_df


# ============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================================

def plot_results(results_df, models, X_test, y_test):
    """
    Genera visualizaciones de los resultados
    
    Args:
        results_df: DataFrame con métricas
        models: Diccionario con modelos
        X_test: Features de prueba
        y_test: Target de prueba
    """
    print("\n" + "="*70)
    print("GENERACIÓN DE GRÁFICOS")
    print("="*70)
    
    # Configuración de estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Comparación de métricas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparación de Métricas por Modelo', fontsize=16, fontweight='bold')
    
    metrics = ['MAE_test', 'MSE_test', 'RMSE_test', 'R2_test']
    titles = ['MAE (Mean Absolute Error)', 'MSE (Mean Squared Error)', 
              'RMSE (Root Mean Squared Error)', 'R² (Coeficiente de Determinación)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        colors = ['#3498db' if t == 'Caja Blanca' else '#9b59b6' 
                  for t in results_df['Tipo']]
        ax.bar(results_df['Modelo'], results_df[metric], color=colors, alpha=0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Modelo')
        ax.set_ylabel(metric.split('_')[0])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metricas_comparacion.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de comparación guardado: metricas_comparacion.png")
    
    # 2. Predicciones vs Valores Reales (mejor modelo)
    best_model_name = results_df.loc[results_df['R2_test'].idxmax(), 'Modelo']
    best_model = models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción perfecta')
    plt.xlabel('Valores Reales', fontsize=12)
    plt.ylabel('Valores Predichos', fontsize=12)
    plt.title(f'Predicciones vs Valores Reales - {best_model_name}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de predicciones guardado: predicciones_vs_reales.png")
    
    # 3. Distribución de errores
    errors = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='#2ecc71')
    plt.xlabel('Error (Real - Predicho)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(f'Distribución de Errores - {best_model_name}', 
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribucion_errores.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de distribución guardado: distribucion_errores.png")
    
    # 4. Importancia de características (Random Forest)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
                 color='#e74c3c', alpha=0.7)
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Característica', fontsize=12)
        plt.title('Top 15 Características Más Importantes - Random Forest', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
        print("✓ Gráfico de importancia guardado: importancia_caracteristicas.png")


# ============================================================================
# 8. TABLA COMPARATIVA
# ============================================================================

def print_comparison_table(results_df):
    """
    Imprime tabla comparativa formateada
    
    Args:
        results_df: DataFrame con resultados
    """
    print("\n" + "="*70)
    print("TABLA COMPARATIVA DE MODELOS (Conjunto de Prueba)")
    print("="*70)
    
    # Seleccionar columnas relevantes
    display_df = results_df[['Modelo', 'Tipo', 'MAE_test', 'MSE_test', 
                              'RMSE_test', 'R2_test']].copy()
    display_df.columns = ['Modelo', 'Tipo', 'MAE', 'MSE', 'RMSE', 'R²']
    
    # Formatear números
    for col in ['MAE', 'MSE', 'RMSE']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    display_df['R²'] = display_df['R²'].apply(lambda x: f"{x:.4f}")
    
    print("\n" + display_df.to_string(index=False))
    
    # Identificar mejor modelo
    best_idx = results_df['R2_test'].idxmax()
    best_model = results_df.loc[best_idx, 'Modelo']
    best_r2 = results_df.loc[best_idx, 'R2_test']
    best_rmse = results_df.loc[best_idx, 'RMSE_test']
    
    print("\n" + "="*70)
    print(f"🏆 MEJOR MODELO: {best_model}")
    print(f"   - R² = {best_r2:.4f} (explica {best_r2*100:.2f}% de la varianza)")
    print(f"   - RMSE = {best_rmse:.4f}")
    print("="*70)


# ============================================================================
# 9. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("\n" + "="*70)
    print(" SISTEMA DE MACHINE LEARNING SUPERVISADO - REGRESIÓN")
    print(" Problema: Predicción de Salud Cardiovascular (BMI)")
    print(" Estudiante: Celadita (Grupo A-C)")
    print("="*70)
    
    # Configuración
    DATA_PATH = 'heart_disease_health_indicators.csv'  # Ruta del dataset
    TARGET_COLUMN = 'BMI'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    try:
        # 1. Cargar datos
        df = load_and_explore_data(DATA_PATH)
        
        # 2. Preprocesar
        X, y, scaler = preprocess_data(df, TARGET_COLUMN)
        
        # 3. Dividir datos
        X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
        
        # 4. Entrenar modelos
        models = train_models(X_train, y_train)
        
        # 5. Evaluar modelos
        results_df = evaluate_models(models, X_train, y_train, X_test, y_test)
        
        # 6. Tabla comparativa
        print_comparison_table(results_df)
        
        # 7. Visualizaciones
        plot_results(results_df, models, X_test, y_test)
        
        # 8. Guardar resultados
        results_df.to_csv('resultados_modelos.csv', index=False)
        print("\n✓ Resultados guardados en: resultados_modelos.csv")
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        
        return models, results_df, scaler
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo '{DATA_PATH}'")
        print("   Descarga el dataset de Kaggle y colócalo en el directorio actual")
        return None, None, None
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None, None, None


# ============================================================================
# 10. EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    models, results, scaler = main()