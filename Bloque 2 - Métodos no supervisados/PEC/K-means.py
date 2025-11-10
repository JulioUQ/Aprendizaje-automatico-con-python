import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import umap.umap_ as umap

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def evaluar_k_con_metricas(X, k_range, random_state=42):
    """
    Evalúa múltiples valores de K calculando inercia y silhouette con Jaccard
    Combina el método del codo y silhouette en una sola función
    """
    resultados = []
    
    for k in k_range:
        print(f"Evaluando K={k}...", end='\r')
        
        # Entrenamos K-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(X)
        
        # Calculamos inercia con distancia Jaccard
        centers = kmeans.cluster_centers_
        distances = cdist(X, centers, metric='jaccard')
        inertia_jaccard = np.sum(np.min(distances, axis=1))
        
        # Calculamos Silhouette con Jaccard
        silhouette = silhouette_score(X, labels, metric='jaccard')
        
        # Distribución de muestras por cluster
        cluster_sizes = np.bincount(labels)
        
        resultados.append({
            'k': k,
            'inertia_jaccard': inertia_jaccard,
            'silhouette': silhouette,
            'min_cluster_size': cluster_sizes.min(),
            'max_cluster_size': cluster_sizes.max(),
            'mean_cluster_size': cluster_sizes.mean()
        })
    
    print(" " * 50, end='\r')
    return pd.DataFrame(resultados)


def visualizar_metricas(df_metricas):
    """
    Visualiza las métricas de evaluación
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Método del codo con Jaccard
    ax1 = axes[0]
    ax1.plot(df_metricas['k'], df_metricas['inertia_jaccard'], 'bo-', 
             linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inercia (Distancia Jaccard)', fontsize=12)
    ax1.set_title('Método del Codo (Jaccard)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df_metricas['k'])
    
    # 2. Silhouette Score con Jaccard
    ax2 = axes[1]
    ax2.plot(df_metricas['k'], df_metricas['silhouette'], 'go-', 
             linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score (Jaccard)', fontsize=12)
    ax2.set_title('Silhouette Score (Mayor es mejor)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Umbral = 0')
    ax2.legend()
    ax2.set_xticks(df_metricas['k'])
    
    plt.tight_layout()
    plt.show()


def visualizar_clusters_pca(X, labels, n_clusters, max_samples=5000):
    """
    Visualiza los clusters usando PCA (con límite de muestras para velocidad)
    """
    # Si hay muchas muestras, tomar una muestra aleatoria
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    # Reducción a 2D con PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    # Visualización
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sample, 
                         cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)', fontsize=12)
    plt.title(f'Visualización de {n_clusters} Clusters (PCA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualizar_clusters_umap(X, labels, n_clusters=None, max_samples=5000):
    """
    Visualiza los clusters usando UMAP con colores bien diferenciados.
    Compatible con KMeans, HDBSCAN o cualquier algoritmo de clustering.

    Parámetros
    ----------
    X : np.array
        Matriz de datos (ej. one-hot o reducida).
    labels : np.array
        Etiquetas de cluster (pueden incluir -1 si hay ruido).
    n_clusters : int, opcional
        Número total de clusters (solo informativo en el título).
    max_samples : int
        Máximo de muestras a visualizar (para no saturar el gráfico).
    """
    # Muestreo aleatorio para velocidad
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    # Reducción UMAP
    reducer = umap.UMAP(
        n_components=2,
        metric='jaccard',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    X_umap = reducer.fit_transform(X_sample)

    # Identificar clusters únicos
    unique_labels = np.unique(labels_sample)
    n_labels = len(unique_labels)

    # Crear paleta de colores amplia y reproducible
    palette = sns.color_palette("tab20", n_labels)
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
    # Color gris para ruido (-1)
    color_map[-1] = (0.6, 0.6, 0.6)

    colors = [color_map[label] for label in labels_sample]

    # Visualización
    plt.figure(figsize=(12, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=30, alpha=0.7, edgecolor='none')

    # Crear leyenda ordenada
    handles = []
    for label in unique_labels:
        color = color_map[label]
        if label == -1:
            name = "Ruido / No asignado"
        else:
            name = f"Cluster {label}"
        handles.append(plt.Line2D([], [], marker='o', color=color, linestyle='', label=name))
    plt.legend(handles=handles, loc='best', frameon=True)

    # Título ejes
    title = f"Visualización de clusters con UMAP ({'K=' + str(n_clusters) if n_clusters else 'n_clusters desconocido'})"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analizar_clusters(df_original, labels, df_ingredientes, top_n=10):
    """
    Analiza las características de cada cluster
    """
    df_analisis = df_original.copy()
    df_analisis['cluster'] = labels
    
    print("="*80)
    print("ANÁLISIS DE CLUSTERS")
    print("="*80)
    
    for cluster_id in sorted(df_analisis['cluster'].unique()):
        recetas_cluster = df_analisis[df_analisis['cluster'] == cluster_id]
        n_recetas = len(recetas_cluster)
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id} - {n_recetas} recetas ({n_recetas/len(df_analisis)*100:.2f}%)")
        print(f"{'='*80}")
        
        # Ingredientes más frecuentes en este cluster
        ingredientes_cluster = df_ingredientes.loc[recetas_cluster.index]
        freq_ingredientes = ingredientes_cluster.sum().sort_values(ascending=False).head(top_n)
        
        print(f"\nTop {top_n} ingredientes más comunes:")
        for ing, freq in freq_ingredientes.items():
            print(f"  • {ing}: {freq} recetas ({freq/n_recetas*100:.1f}%)")
        
        # Muestra de recetas
        if 'name' in recetas_cluster.columns:
            print(f"\nMuestra de 3 recetas del cluster:")
            for idx, row in recetas_cluster.head(3).iterrows():
                print(f"  - {row['name']}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def pipeline_clustering_recetas(df_ingredientes_onehot, df_original, k_range=range(2, 11)):
    """
    Pipeline completo de clustering para recetas usando solo métrica Jaccard
    
    Parámetros:
    -----------
    df_ingredientes_onehot : DataFrame
        Matriz binaria de ingredientes (resultado de MultiLabelBinarizer)
    df_original : DataFrame
        DataFrame original con información de las recetas
    k_range : range
        Rango de valores K a evaluar
    """
    
    print("="*80)
    print("CLUSTERING K-MEANS PARA RECETAS (Métrica: Jaccard)")
    print("="*80)
    print(f"\nDimensiones del dataset: {df_ingredientes_onehot.shape}")
    print(f"Sparsity: {(df_ingredientes_onehot == 0).sum().sum() / df_ingredientes_onehot.values.size * 100:.2f}%")
    
    # Convertimos a array numpy
    X = df_ingredientes_onehot.values
    
    # EVALUACIÓN DE K CON INERCIA Y SILHOUETTE
    print("\n" + "="*80)
    print("EVALUACIÓN: Inercia y Silhouette con Jaccard")
    print("="*80)
    df_metricas = evaluar_k_con_metricas(X, k_range)
    
    print("\nRESULTADOS DE EVALUACIÓN:")
    print(df_metricas.to_string(index=False))
    
    # RECOMENDACIÓN DE K ÓPTIMO
    print("\n" + "="*80)
    print("RECOMENDACIÓN DE K ÓPTIMO")
    print("="*80)
    
    # Normalización de métricas
    df_norm = df_metricas.copy()
    
    # Inercia: menor es mejor → invertimos para normalización
    df_norm['inertia_norm'] = 1 - (df_norm['inertia_jaccard'] - df_norm['inertia_jaccard'].min()) / \
                               (df_norm['inertia_jaccard'].max() - df_norm['inertia_jaccard'].min())
    
    # Silhouette: mayor es mejor
    df_norm['silhouette_norm'] = (df_norm['silhouette'] - df_norm['silhouette'].min()) / \
                                 (df_norm['silhouette'].max() - df_norm['silhouette'].min())
    
    # Score combinado (50% inercia, 50% silhouette)
    df_norm['score_total'] = 0.5 * df_norm['inertia_norm'] + 0.5 * df_norm['silhouette_norm']
    
    k_optimo = df_norm.loc[df_norm['score_total'].idxmax(), 'k']
    
    print(f"\nK ÓPTIMO RECOMENDADO: {int(k_optimo)}")
    print("\nJustificación:")
    print(f"  • Inercia (Jaccard): {df_metricas.loc[df_metricas['k']==k_optimo, 'inertia_jaccard'].values[0]:.4f}")
    print(f"  • Silhouette (Jaccard): {df_metricas.loc[df_metricas['k']==k_optimo, 'silhouette'].values[0]:.4f}")
    print(f"  • Tamaño medio de cluster: {df_metricas.loc[df_metricas['k']==k_optimo, 'mean_cluster_size'].values[0]:.0f} recetas")
    
    # VISUALIZACIONES
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)
    visualizar_metricas(df_metricas)
    
    # MODELO FINAL
    print("\n" + "="*80)
    print(f"ENTRENANDO MODELO FINAL con K={int(k_optimo)}")
    print("="*80)
    
    kmeans_final = KMeans(n_clusters=int(k_optimo), random_state=42, n_init=20, max_iter=100)
    labels_final = kmeans_final.fit_predict(X)
    
    # Visualizar clusters
    print("\nGenerando visualización PCA...")
    visualizar_clusters_pca(X, labels_final, int(k_optimo))
    
    # Visualizar clusters
    print("\nGenerando visualización UMAP...")
    visualizar_clusters_umap(X, labels_final, int(k_optimo))

    # Análisis de clusters
    analizar_clusters(df_original, labels_final, df_ingredientes_onehot, top_n=15)
    
    return kmeans_final, labels_final, df_metricas


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    """
    # Suponiendo que ya tienes:
    # - df: DataFrame original con las recetas
    # - df_ingredientes_onehot: Matriz binaria de ingredientes
    
    # Ejecutar pipeline completo
    modelo, labels, metricas = pipeline_clustering_recetas(
        df_ingredientes_onehot=df_ingredientes_onehot,
        df_original=df,
        k_range=range(2, 8)  # Rango reducido para velocidad
    )
    
    # Agregar labels al DataFrame original
    df['cluster'] = labels
    
    # Guardar modelo (opcional)
    import joblib
    joblib.dump(modelo, 'kmeans_recetas.pkl')
    """
    
    print("Script listo para ejecutar.")
    print("\nPara usar, ejecuta:")
    print("modelo, labels, metricas = pipeline_clustering_recetas(df_ingredientes_onehot, df, k_range=range(2, 8))")