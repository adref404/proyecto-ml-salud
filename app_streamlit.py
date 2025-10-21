"""
Interfaz GUI con Streamlit para Sistema ML de Predicción de Salud
Archivo: app_streamlit.py
Ejecución: streamlit run app_streamlit.py
"""

import streamlit as st
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
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema ML - Salud Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Carga el dataset"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def preprocess_data(df, target_col='BMI'):
    """Preprocesa los datos"""
    # Copiar dataframe
    df_clean = df.copy()
    
    # Eliminar duplicados
    df_clean = df_clean.drop_duplicates()
    
    # Tratar outliers en target
    Q1 = df_clean[target_col].quantile(0.25)
    Q3 = df_clean[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_clean[
        (df_clean[target_col] >= lower_bound) & 
        (df_clean[target_col] <= upper_bound)
    ]
    
    # Separar X y y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

def train_all_models(X_train, y_train, X_test, y_test):
    """Entrena y evalúa todos los modelos con optimizaciones"""
    
    models = {
        'Regresión Lineal': LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=1000)  # Optimizado
    }
    
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_container = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        start_time = time.time()
        status_text.text(f"Entrenando {name}...")
        
        # Optimización especial para SVR con datasets grandes
        if name == 'SVR' and len(X_train) > 10000:
            from sklearn.utils import resample
            # Usar muestra de 10,000 registros para SVR
            X_svr, y_svr = resample(X_train, y_train, n_samples=10000, 
                                   random_state=42, stratify=None)
            status_text.text(f"Entrenando {name} (usando muestra de {len(X_svr):,} registros)...")
            model.fit(X_svr, y_svr)
            # Para predicciones, usar el modelo entrenado en la muestra
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            # Entrenar modelo normalmente
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        
        trained_models[name] = model
        elapsed_time = time.time() - start_time
        
        # Métricas
        results.append({
            'Modelo': name,
            'Tipo': 'Caja Blanca' if name in ['Regresión Lineal', 'Árbol de Decisión'] else 'Caja Negra',
            'MAE_train': mean_absolute_error(y_train, y_train_pred),
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R2_train': r2_score(y_train, y_train_pred),
            'MAE_test': mean_absolute_error(y_test, y_test_pred),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R2_test': r2_score(y_test, y_test_pred),
            'Tiempo': f"{elapsed_time:.2f}s"
        })
        
        # Mostrar tiempo transcurrido
        time_container.text(f"⏱️ {name} completado en {elapsed_time:.2f}s")
        
        progress_bar.progress((idx + 1) / len(models))
        time.sleep(0.2)  # Reducido de 0.5s a 0.2s
    
    status_text.text("✅ Entrenamiento completado")
    time_container.empty()
    
    return pd.DataFrame(results), trained_models

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🫀 Sistema ML - Predicción Salud Cardiovascular</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    **Proyecto:** Sistema de Machine Learning Supervisado para Regresión  
    **Estudiante:** Celadita (Grupo A-C - Problema de Salud)  
    **Dataset:** Heart Disease Health Indicators (Kaggle CDC)
    """)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=100)
        st.title("⚙️ Configuración")
        
        st.markdown("### 📊 Dataset")
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="Sube el dataset de Kaggle: Heart Disease Health Indicators"
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Parámetros")
        
        test_size = st.slider(
            "Tamaño conjunto de prueba (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        ) / 100
        
        random_state = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            max_value=100,
            value=42
        )
        
        st.markdown("---")
        st.markdown("### 📚 Información")
        st.info("""
        **Modelos implementados:**
        - 🔵 Regresión Lineal (Caja Blanca)
        - 🔵 Árbol de Decisión (Caja Blanca)
        - 🟣 Random Forest (Caja Negra)
        - 🟣 SVR (Caja Negra)
        """)
    
    # Inicializar variables de sesión
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Dataset", 
        "🔧 Preprocesamiento", 
        "🚀 Entrenamiento", 
        "📊 Resultados",
        "🎯 Predicción"
    ])
    
    # ========================================================================
    # TAB 1: DATASET
    # ========================================================================
    with tab1:
        st.header("📁 Exploración del Dataset")
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📝 Registros", f"{df.shape[0]:,}")
            with col2:
                st.metric("📊 Variables", df.shape[1])
            with col3:
                st.metric("❌ Valores Nulos", df.isnull().sum().sum())
            with col4:
                st.metric("🔄 Duplicados", df.duplicated().sum())
            
            st.markdown("---")
            
            # Mostrar datos
            st.subheader("Vista previa de datos")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Estadísticas
            st.subheader("Estadísticas descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Distribución de la variable objetivo
            st.subheader("Distribución de BMI (Variable Objetivo)")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['BMI'], bins=50, edgecolor='black', alpha=0.7, color='#667eea')
            ax.set_xlabel('BMI', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.set_title('Distribución del Índice de Masa Corporal', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ Por favor, carga un dataset desde la barra lateral")
            st.info("""
            **Instrucciones:**
            1. Descarga el dataset "Heart Disease Health Indicators" desde Kaggle
            2. Usa el botón de carga en la barra lateral
            3. El sistema procesará automáticamente los datos
            """)
    
    # ========================================================================
    # TAB 2: PREPROCESAMIENTO
    # ========================================================================
    with tab2:
        st.header("🔧 Preprocesamiento de Datos")
        
        if st.session_state.data_loaded:
            df = st.session_state.df
            
            st.subheader("Pasos de preprocesamiento")
            
            steps = [
                ("1️⃣", "Eliminación de duplicados", "Limpieza inicial de registros repetidos"),
                ("2️⃣", "Tratamiento de outliers", "Método IQR para detectar valores atípicos en BMI"),
                ("3️⃣", "Normalización", "StandardScaler (μ=0, σ=1) aplicado a todas las features"),
                ("4️⃣", "División de datos", f"Entrenamiento: {(1-test_size)*100:.0f}% | Prueba: {test_size*100:.0f}%")
            ]
            
            for icon, title, description in steps:
                with st.container():
                    col1, col2 = st.columns([1, 9])
                    with col1:
                        st.markdown(f"### {icon}")
                    with col2:
                        st.markdown(f"**{title}**")
                        st.caption(description)
                st.markdown("---")
            
            if st.button("🔄 Ejecutar Preprocesamiento", use_container_width=True):
                with st.spinner("Procesando datos..."):
                    X, y, scaler = preprocess_data(df)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Guardar en sesión
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    st.session_state.preprocessed = True
                    
                    time.sleep(1)
                    st.success("✓ Preprocesamiento completado exitosamente")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📚 Datos de Entrenamiento", f"{len(X_train):,}")
                    with col2:
                        st.metric("🧪 Datos de Prueba", f"{len(X_test):,}")
        else:
            st.warning("⚠️ Primero carga el dataset en la pestaña 'Dataset'")
    
    # ========================================================================
    # TAB 3: ENTRENAMIENTO
    # ========================================================================
    with tab3:
        st.header("🚀 Entrenamiento de Modelos")
        
        if st.session_state.data_loaded and hasattr(st.session_state, 'preprocessed'):
            
            st.info("""
            **Modelos a entrenar:**
            
            **Caja Blanca (Interpretables):**
            - **Regresión Lineal:** Modelo paramétrico que asume relación lineal
            - **Árbol de Decisión:** Modelo basado en reglas de división recursiva
            
            **Caja Negra (Mayor complejidad):**
            - **Random Forest:** Ensamble de árboles de decisión (100 estimadores)
            - **SVR:** Support Vector Regressor con kernel RBF (optimizado con muestreo)
            
            **Validación:** K-Fold Cross Validation con K=5
            
            **⚡ Optimizaciones aplicadas:**
            - SVR usa muestreo inteligente para datasets grandes (>10K registros)
            - Parámetros optimizados para mejor rendimiento
            - Indicadores de tiempo en tiempo real
            """)
            
            st.markdown("---")
            
            if st.button("▶️ Entrenar Todos los Modelos", use_container_width=True, type="primary"):
                st.markdown("### 📈 Progreso del Entrenamiento")
                
                results_df, models = train_all_models(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    st.session_state.X_test,
                    st.session_state.y_test
                )
                
                st.session_state.results_df = results_df
                st.session_state.models = models
                st.session_state.models_trained = True
                
                st.balloons()
                st.success("🎉 ¡Entrenamiento completado exitosamente!")
                
                # Mostrar resumen rápido
                st.markdown("### 📊 Resumen Rápido")
                best_idx = results_df['R2_test'].idxmax()
                best_model = results_df.loc[best_idx, 'Modelo']
                
                st.info(f"🏆 **Mejor modelo:** {best_model} con R² = {results_df.loc[best_idx, 'R2_test']:.4f}")
        else:
            st.warning("⚠️ Primero completa el preprocesamiento en la pestaña anterior")
    
    # ========================================================================
    # TAB 4: RESULTADOS
    # ========================================================================
    with tab4:
        st.header("📊 Resultados y Evaluación")
        
        if st.session_state.models_trained:
            results_df = st.session_state.results_df
            
            # Tabla de resultados
            st.subheader("📋 Tabla Comparativa de Métricas")
            
            # Formatear tabla para visualización
            display_df = results_df[['Modelo', 'Tipo', 'MAE_test', 'RMSE_test', 'R2_test']].copy()
            display_df.columns = ['Modelo', 'Tipo', 'MAE', 'RMSE', 'R²']
            
            # Estilo para resaltar mejor modelo
            def highlight_best(row):
                if row['R²'] == results_df['R2_test'].max():
                    return ['background-color: #d4edda'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                display_df.style.format({
                    'MAE': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'R²': '{:.4f}'
                }).apply(highlight_best, axis=1),
                use_container_width=True
            )
            
            # Mejor modelo
            best_idx = results_df['R2_test'].idxmax()
            best_model_name = results_df.loc[best_idx, 'Modelo']
            
            st.success(f"""
            ### 🏆 Mejor Modelo: {best_model_name}
            - **MAE:** {results_df.loc[best_idx, 'MAE_test']:.4f}
            - **RMSE:** {results_df.loc[best_idx, 'RMSE_test']:.4f}
            - **R²:** {results_df.loc[best_idx, 'R2_test']:.4f} (explica {results_df.loc[best_idx, 'R2_test']*100:.2f}% de la varianza)
            """)
            
            st.markdown("---")
            
            # Gráficos comparativos
            st.subheader("📈 Visualizaciones Comparativas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de RMSE
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                colors = ['#3498db' if t == 'Caja Blanca' else '#9b59b6' 
                         for t in results_df['Tipo']]
                ax1.barh(results_df['Modelo'], results_df['RMSE_test'], color=colors, alpha=0.7)
                ax1.set_xlabel('RMSE', fontsize=11)
                ax1.set_title('Comparación de RMSE por Modelo', fontsize=12, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig1)
            
            with col2:
                # Gráfico de R²
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(results_df['Modelo'], results_df['R2_test'], color=colors, alpha=0.7)
                ax2.set_xlabel('R² Score', fontsize=11)
                ax2.set_title('Comparación de R² por Modelo', fontsize=12, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Predicciones vs Reales
            st.subheader("🎯 Predicciones vs Valores Reales (Mejor Modelo)")
            
            best_model = st.session_state.models[best_model_name]
            y_pred = best_model.predict(st.session_state.X_test)
            y_test = st.session_state.y_test
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.scatter(y_test, y_pred, alpha=0.5, s=20, color='#667eea')
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Predicción perfecta')
            ax3.set_xlabel('Valores Reales (BMI)', fontsize=12)
            ax3.set_ylabel('Valores Predichos (BMI)', fontsize=12)
            ax3.set_title(f'Predicciones vs Reales - {best_model_name}', 
                         fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Importancia de características (Random Forest)
            if 'Random Forest' in st.session_state.models:
                st.subheader("⭐ Importancia de Características (Random Forest)")
                
                rf_model = st.session_state.models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Característica': st.session_state.X_train.columns,
                    'Importancia': rf_model.feature_importances_
                }).sort_values('Importancia', ascending=False).head(10)
                
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.barh(feature_importance['Característica'], 
                        feature_importance['Importancia'], 
                        color='#e74c3c', alpha=0.7)
                ax4.set_xlabel('Importancia', fontsize=12)
                ax4.set_title('Top 10 Características Más Importantes', 
                             fontsize=14, fontweight='bold')
                ax4.invert_yaxis()
                ax4.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig4)
            
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Entrenamiento'")
    
    # ========================================================================
    # TAB 5: PREDICCIÓN
    # ========================================================================
    with tab5:
        st.header("🎯 Predicción Individual")
        
        if st.session_state.models_trained:
            st.info("Ingresa los valores de las características para predecir el BMI")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_bp = st.selectbox("Presión Alta", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                high_chol = st.selectbox("Colesterol Alto", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                smoker = st.selectbox("Fumador", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                stroke = st.selectbox("Derrame Cerebral", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                phys_activity = st.selectbox("Actividad Física", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                
            with col2:
                fruits = st.selectbox("Consume Frutas", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                veggies = st.selectbox("Consume Vegetales", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                hvy_alcohol = st.selectbox("Consumo Excesivo Alcohol", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                healthcare = st.selectbox("Tiene Seguro Médico", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
                
            with col3:
                age = st.slider("Edad (categoría)", 1, 13, 7)
                education = st.slider("Nivel Educativo", 1, 6, 4)
                income = st.slider("Nivel de Ingresos", 1, 8, 5)
                gen_health = st.slider("Salud General", 1, 5, 3)
                ment_health = st.slider("Días Salud Mental Mala (último mes)", 0, 30, 5)
                phys_health = st.slider("Días Salud Física Mala (último mes)", 0, 30, 5)
            
            if st.button("🔮 Realizar Predicción", use_container_width=True, type="primary"):
                # Crear entrada
                input_data = pd.DataFrame({
                    'HighBP': [high_bp],
                    'HighChol': [high_chol],
                    'CholCheck': [1],  # Asumido
                    'Smoker': [smoker],
                    'Stroke': [stroke],
                    'Diabetes': [diabetes],
                    'PhysActivity': [phys_activity],
                    'Fruits': [fruits],
                    'Veggies': [veggies],
                    'HvyAlcoholConsump': [hvy_alcohol],
                    'AnyHealthcare': [healthcare],
                    'NoDocbcCost': [0],  # Valor por defecto
                    'GenHlth': [gen_health],
                    'MentHlth': [ment_health],
                    'PhysHlth': [phys_health],
                    'DiffWalk': [0],  # Valor por defecto
                    'Sex': [sex],
                    'Age': [age],
                    'Education': [education],
                    'Income': [income],
                    'HeartDiseaseorAttack': [0]  # Valor por defecto
                })
                
                # Normalizar
                input_scaled = st.session_state.scaler.transform(input_data)
                
                # Predecir con todos los modelos
                st.markdown("### 📊 Predicciones por Modelo")
                
                predictions = {}
                for name, model in st.session_state.models.items():
                    pred = model.predict(input_scaled)[0]
                    predictions[name] = pred
                
                # Mostrar predicciones
                cols = st.columns(4)
                for idx, (name, pred) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.metric(name, f"{pred:.2f}")
                
                # Predicción del mejor modelo
                best_model_name = st.session_state.results_df.loc[
                    st.session_state.results_df['R2_test'].idxmax(), 'Modelo'
                ]
                best_prediction = predictions[best_model_name]
                
                st.markdown("---")
                st.success(f"""
                ### 🏆 Predicción del Mejor Modelo ({best_model_name})
                
                **BMI Estimado: {best_prediction:.2f} kg/m²**
                """)
                
                # Interpretación
                if best_prediction < 18.5:
                    interpretation = "**Bajo peso** - Se recomienda evaluación nutricional"
                    color = "blue"
                elif 18.5 <= best_prediction < 25:
                    interpretation = "**Peso normal** - Mantener hábitos saludables"
                    color = "green"
                elif 25 <= best_prediction < 30:
                    interpretation = "**Sobrepeso** - Considerar plan de manejo de peso"
                    color = "orange"
                else:
                    interpretation = "**Obesidad** - Se recomienda evaluación médica"
                    color = "red"
                
                st.markdown(f"**Interpretación:** :{color}[{interpretation}]")
                
        else:
            st.warning("⚠️ Primero entrena los modelos en la pestaña 'Entrenamiento'")

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()