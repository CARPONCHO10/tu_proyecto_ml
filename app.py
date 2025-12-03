import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="ML Models - Supervised vs Unsupervised", layout="wide")

# Título principal
st.title("Tarea Investigativa: Modelos Supervisados vs No Supervisados")
st.markdown("---")

# Carga de datos
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data  # Matriz de características
    y = iris.target  # Vector de etiquetas
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names, iris

X, y, feature_names, target_names, iris = load_data()

# Sidebar para navegación
st.sidebar.title("Navegación")
modo = st.sidebar.radio("Selecciona el modo:", ["Modo Supervisado", "Modo No Supervisado"])

if modo == "Modo Supervisado":
    st.header("🔍 Modelo Supervisado: Random Forest")
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entrenamiento del modelo
    with st.spinner("Entrenando modelo Random Forest..."):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
    
    # Métricas
    st.subheader("Métricas de Evaluación")
    col1, col2, col3, col4 = st.columns(4)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    # Prueba interactiva
    st.subheader("Prueba Interactiva")
    st.write("Ingresa los valores para las características de la flor Iris:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 
                                float(X[:, 0].min()), 
                                float(X[:, 0].max()), 
                                float(X[:, 0].mean()))
    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 
                               float(X[:, 1].min()), 
                               float(X[:, 1].max()), 
                               float(X[:, 1].mean()))
    with col3:
        petal_length = st.slider("Petal Length (cm)", 
                                float(X[:, 2].min()), 
                                float(X[:, 2].max()), 
                                float(X[:, 2].mean()))
    with col4:
        petal_width = st.slider("Petal Width (cm)", 
                               float(X[:, 3].min()), 
                               float(X[:, 3].max()), 
                                float(X[:, 3].mean()))
    
    # Predicción
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = rf_model.predict(input_data)[0]
    prediction_proba = rf_model.predict_proba(input_data)[0]
    
    st.success(f"**Clase predicha:** {target_names[prediction]} (Clase {prediction})")
    
    # Mostrar probabilidades
    st.write("**Probabilidades por clase:**")
    prob_cols = st.columns(3)
    for i, (col, prob) in enumerate(zip(prob_cols, prediction_proba)):
        with col:
            st.metric(f"Clase {target_names[i]}", f"{prob:.4f}")

else:
    st.header("Modelo No Supervisado: K-Means")
    
    # Entrenamiento del modelo
    with st.spinner("Aplicando K-Means..."):
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
    
    # Métricas de clustering
    st.subheader("Métricas de Calidad de Clustering")
    silhouette_avg = silhouette_score(X, cluster_labels)
    db_index = davies_bouldin_score(X, cluster_labels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Silhouette Score", f"{silhouette_avg:.4f}",
                 help="Valores cercanos a 1 indican clusters bien separados")
    with col2:
        st.metric("Davies-Bouldin Index", f"{db_index:.4f}",
                 help="Valores más bajos indican mejores clusters")
    
    # Visualización de clusters
    st.subheader("Visualización de Clusters")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Sepal features
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Sepal Length (cm)')
    ax1.set_ylabel('Sepal Width (cm)')
    ax1.set_title('Clusters - Sepal Features')
    plt.colorbar(scatter1, ax=ax1)
    
    # Gráfico 2: Petal features
    scatter2 = ax2.scatter(X[:, 2], X[:, 3], c=cluster_labels, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('Petal Length (cm)')
    ax2.set_ylabel('Petal Width (cm)')
    ax2.set_title('Clusters - Petal Features')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    st.pyplot(fig)

# Zona de Exportación
st.markdown("---")
st.header("Zona de Exportación (Dev Tools)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Exportar a JSON (para React)")
    
    if modo == "Modo Supervisado":
        # JSON para modelo supervisado
        json_data_supervised = {
            "model_type": "Supervised",
            "model_name": "Random Forest",
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            },
            "current_prediction": {
                "input": [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)],
                "output_class": int(prediction),
                "output_label": target_names[prediction]
            }
        }
        
        json_str = json.dumps(json_data_supervised, indent=2)
        st.download_button(
            label="Descargar JSON (Supervisado)",
            data=json_str,
            file_name="modelo_supervisado.json",
            mime="application/json"
        )
        st.code(json_str, language="json")
        
    else:
        # JSON para modelo no supervisado
        json_data_unsupervised = {
            "model_type": "Unsupervised",
            "algorithm": "K-Means",
            "parameters": {
                "n_clusters": 3,
                "random_state": 42
            },
            "metrics": {
                "silhouette_score": float(silhouette_avg),
                "davies_bouldin": float(db_index)
            },
            "cluster_labels": cluster_labels.tolist()
        }
        
        json_str = json.dumps(json_data_unsupervised, indent=2)
        st.download_button(
            label="Descargar JSON (No Supervisado)",
            data=json_str,
            file_name="modelo_no_supervisado.json",
            mime="application/json"
        )
        st.code(json_str, language="json")

with col2:
    st.subheader("Exportar Modelo Entrenado (.pkl)")
    
    if modo == "Modo Supervisado":
        # Serializar modelo supervisado
        model_bytes = pickle.dumps(rf_model)
        st.download_button(
            label="Descargar Modelo Random Forest (.pkl)",
            data=model_bytes,
            file_name="random_forest_model.pkl",
            mime="application/octet-stream"
        )
        st.info("El modelo .pkl puede ser cargado en cualquier aplicación Python para hacer predicciones.")
    else:
        # Serializar modelo no supervisado
        model_bytes = pickle.dumps(kmeans)
        st.download_button(
            label="Descargar Modelo K-Means (.pkl)",
            data=model_bytes,
            file_name="kmeans_model.pkl",
            mime="application/octet-stream"
        )
        st.info("El modelo .pkl puede ser cargado para asignar nuevos datos a clusters.")

# Información del dataset
st.sidebar.markdown("---")
st.sidebar.subheader("Información del Dataset")
st.sidebar.write(f"**Dataset:** Iris")
st.sidebar.write(f"**Muestras:** {len(X)}")
st.sidebar.write(f"**Características:** {len(feature_names)}")
st.sidebar.write(f"**Clases:** {list(target_names)}")
st.sidebar.write(f"**X shape:** {X.shape}")
st.sidebar.write(f"**y shape:** {y.shape}")