import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import umap
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIÓN PRINCIPAL SIMPLIFICADA
# ==============================================================================

def dbscan_rapido(df_ingredientes_onehot, df_original, eps, min_samples, 
                  metric='jaccard', mostrar_visualizaciones=True):
    """
    Ejecuta DBSCAN de forma rápida y muestra resultados inmediatos.
    
    Parámetros:
    -----------
    df_ingredientes_onehot : DataFrame
        Matriz binaria de ingredientes (recetas x ingredientes)
    df_original : DataFrame
        DataFrame original con información de las recetas
    eps : float
        Radio máximo de vecindad
    min_samples : int
        Número mínimo de puntos para formar un cluster
    metric : str
        Métrica de distancia ('jaccard', 'hamming', 'cosine', etc.)
    mostrar_visualizaciones : bool
        Si True, muestra gráficos PCA y UMAP
    
    Returns:
    --------
    labels : array
        Etiquetas de cluster para cada receta (-1 = ruido)
    """
    
    print("="*80)
    print(f"DBSCAN: eps={eps}, min_samples={min_samples}, metric={metric}")
    print("="*80)
    
    # Convertir a array
    X = df_ingredientes_onehot.values
    
    # Entrenar DBSCAN
    print("\n⏳ Entrenando DBSCAN...")
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = clusterer.fit_predict(X)
    
    # Métricas básicas
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_pct = (n_noise / len(labels)) * 100
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Clusters encontrados: {n_clusters}")
    print(f"   • Puntos de ruido: {n_noise} ({noise_pct:.1f}%)")
    print(f"   • Recetas en clusters: {len(labels) - n_noise}")
    
    # Silhouette score (solo si hay clusters válidos)
    if n_clusters >= 2 and n_noise < len(labels):
        labels_no_noise = labels[labels != -1]
        X_no_noise = X[labels != -1]
        if len(labels_no_noise) > 0 and len(set(labels_no_noise)) > 1:
            silhouette = silhouette_score(X_no_noise, labels_no_noise, metric=metric)
            print(f"   • Silhouette Score: {silhouette:.4f}")
    
    # Tamaños de clusters
    if n_clusters > 0:
        print(f"\n📏 TAMAÑO DE CLUSTERS:")
        cluster_sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            pct = (size / len(labels)) * 100
            print(f"   • Cluster {cluster_id}: {size} recetas ({pct:.1f}%)")
    
    # Visualizaciones
    if mostrar_visualizaciones and n_clusters > 0:
        visualizar_clusters(X, labels)
    
    # Análisis de clusters
    if n_clusters > 0:
        analizar_clusters(df_original, df_ingredientes_onehot, labels)
    else:
        print("\n⚠️  No se encontraron clusters. Prueba con:")
        print("   - Valores de eps más bajos")
        print("   - Valores de min_samples más bajos")
        print("   - Otra métrica de distancia")
    
    return labels


def visualizar_clusters(X, labels, max_samples=5000):
    """
    Visualiza clusters usando PCA y UMAP en paralelo.
    """
    print("\n📈 Generando visualizaciones...")
    
    # Muestreo si es necesario
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    # Separar ruido de clusters
    noise_mask = labels_sample == -1
    cluster_mask = ~noise_mask
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # === PCA ===
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    # Plot clusters
    if cluster_mask.any():
        unique_clusters = np.unique(labels_sample[cluster_mask])
        n_colors = len(unique_clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = labels_sample == cluster_id
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=f'Cluster {cluster_id}',
                          alpha=0.6, s=30)
    
    # Plot ruido
    if noise_mask.any():
        axes[0].scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                       c='lightgray', alpha=0.3, s=10, label='Ruido')
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    axes[0].set_title('Proyección PCA', fontsize=13, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # === UMAP ===
    print("   Calculando UMAP...")
    reducer = umap.UMAP(n_components=2, metric='jaccard', 
                       n_neighbors=15, min_dist=0.1, 
                       random_state=42, verbose=False)
    X_umap = reducer.fit_transform(X_sample)
    
    # Plot clusters
    if cluster_mask.any():
        for i, cluster_id in enumerate(unique_clusters):
            mask = labels_sample == cluster_id
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                          c=[colors[i]], label=f'Cluster {cluster_id}',
                          alpha=0.6, s=30)
    
    # Plot ruido
    if noise_mask.any():
        axes[1].scatter(X_umap[noise_mask, 0], X_umap[noise_mask, 1], 
                       c='lightgray', alpha=0.3, s=10, label='Ruido')
    
    axes[1].set_xlabel('UMAP 1', fontsize=11)
    axes[1].set_ylabel('UMAP 2', fontsize=11)
    axes[1].set_title('Proyección UMAP', fontsize=13, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("   ✓ Visualizaciones completadas")


def analizar_clusters(df_original, df_ingredientes_onehot, labels, top_clusters=10, top_ingredientes=5):
    """
    Muestra resumen de los top N clusters con sus ingredientes más frecuentes.
    """
    print("\n" + "="*80)
    print(f"TOP {top_clusters} CLUSTERS MÁS GRANDES")
    print("="*80)
    
    # Contar tamaños de clusters (sin ruido)
    cluster_sizes = pd.Series(labels[labels != -1]).value_counts()
    top_cluster_ids = cluster_sizes.head(top_clusters).index
    
    # Analizar cada cluster
    for cluster_id in top_cluster_ids:
        mask = labels == cluster_id
        n_recetas = mask.sum()
        pct = (n_recetas / len(labels)) * 100
        
        print(f"\n{'─'*80}")
        print(f"CLUSTER {cluster_id} → {n_recetas} recetas ({pct:.1f}%)")
        print(f"{'─'*80}")
        
        # Ingredientes más frecuentes
        ingredientes_cluster = df_ingredientes_onehot[mask]
        freq_ingredientes = ingredientes_cluster.sum().sort_values(ascending=False).head(top_ingredientes)
        
        print(f"\n🥘 Top {top_ingredientes} ingredientes:")
        for ing, freq in freq_ingredientes.items():
            freq_pct = (freq / n_recetas) * 100
            print(f"   • {ing}: {int(freq)} recetas ({freq_pct:.0f}%)")
        
        # Recetas de ejemplo
        if 'name' in df_original.columns:
            recetas_ejemplo = df_original[mask]['name'].head(5).tolist()
            print(f"\n📋 Ejemplos de recetas:")
            for i, receta in enumerate(recetas_ejemplo, 1):
                print(f"   {i}. {receta}")


# ==============================================================================
# FUNCIÓN DE EXPLORACIÓN RÁPIDA
# ==============================================================================

def explorar_parametros(df_ingredientes_onehot, df_original, 
                       eps_list, min_samples_list, metric='jaccard'):
    """
    Prueba rápidamente múltiples combinaciones y muestra resumen comparativo.
    """
    resultados = []
    
    print("="*80)
    print("EXPLORACIÓN RÁPIDA DE PARÁMETROS")
    print("="*80)
    
    X = df_ingredientes_onehot.values
    total = len(eps_list) * len(min_samples_list)
    counter = 0
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            counter += 1
            print(f"\n[{counter}/{total}] Probando eps={eps}, min_samples={min_samples}...")
            
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, 
                             metric=metric, n_jobs=-1)
            labels = clusterer.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_pct = (n_noise / len(labels)) * 100
            
            # Silhouette
            if n_clusters >= 2 and n_noise < len(labels):
                labels_no_noise = labels[labels != -1]
                X_no_noise = X[labels != -1]
                if len(labels_no_noise) > 0 and len(set(labels_no_noise)) > 1:
                    silhouette = silhouette_score(X_no_noise, labels_no_noise, metric=metric)
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            resultados.append({
                'eps': eps,
                'min_samples': min_samples,
                'clusters': n_clusters,
                'ruido': n_noise,
                'ruido_%': noise_pct,
                'silhouette': silhouette
            })
            
            print(f"   → {n_clusters} clusters, {n_noise} ruido ({noise_pct:.1f}%), silhouette={silhouette:.3f}")
    
    # Mostrar tabla resumen
    df_resultados = pd.DataFrame(resultados)
    
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)
    print(df_resultados.to_string(index=False))
    
    # Sugerencias
    print("\n" + "="*80)
    print("💡 SUGERENCIAS")
    print("="*80)
    
    # Mejor por silhouette
    df_validos = df_resultados[df_resultados['silhouette'] > 0]
    if len(df_validos) > 0:
        mejor = df_validos.loc[df_validos['silhouette'].idxmax()]
        print(f"\n✓ Mejor configuración (silhouette):")
        print(f"   eps={mejor['eps']}, min_samples={int(mejor['min_samples'])}")
        print(f"   → {int(mejor['clusters'])} clusters, silhouette={mejor['silhouette']:.3f}")
    
    # Más clusters
    mejor_clusters = df_resultados.loc[df_resultados['clusters'].idxmax()]
    if mejor_clusters['clusters'] > 0:
        print(f"\n✓ Más clusters encontrados:")
        print(f"   eps={mejor_clusters['eps']}, min_samples={int(mejor_clusters['min_samples'])}")
        print(f"   → {int(mejor_clusters['clusters'])} clusters")
    
    # Menos ruido
    if (df_resultados['clusters'] > 0).any():
        df_con_clusters = df_resultados[df_resultados['clusters'] > 0]
        menos_ruido = df_con_clusters.loc[df_con_clusters['ruido_%'].idxmin()]
        print(f"\n✓ Menos ruido (con clusters):")
        print(f"   eps={menos_ruido['eps']}, min_samples={int(menos_ruido['min_samples'])}")
        print(f"   → {menos_ruido['ruido_%']:.1f}% ruido")
    
    return df_resultados


# ==============================================================================
# EJEMPLOS DE USO
# ==============================================================================

"""
# ===== USO BÁSICO =====
# Probar una configuración específica
labels = dbscan_rapido(
    df_ingredientes_onehot=df_ingredientes_onehot,
    df_original=df,
    eps=0.5,
    min_samples=5,
    metric='jaccard'
)

# ===== EXPLORACIÓN RÁPIDA =====
# Probar múltiples configuraciones de golpe
resultados = explorar_parametros(
    df_ingredientes_onehot=df_ingredientes_onehot,
    df_original=df,
    eps_list=[0.3, 0.4, 0.5, 0.6],
    min_samples_list=[3, 5, 10],
    metric='jaccard'
)

# ===== PROBAR OTRAS MÉTRICAS =====
# Jaccard (recomendada para datos binarios)
labels_jaccard = dbscan_rapido(df_ingredientes_onehot, df, 
                                eps=0.5, min_samples=5, metric='jaccard')

# Hamming (otra opción para datos binarios)
labels_hamming = dbscan_rapido(df_ingredientes_onehot, df, 
                                eps=0.3, min_samples=5, metric='hamming')

# Cosine (para similitud angular)
labels_cosine = dbscan_rapido(df_ingredientes_onehot, df, 
                               eps=0.4, min_samples=5, metric='cosine')

# ===== GUARDAR RESULTADOS =====
# Agregar labels al DataFrame original
df['cluster_dbscan'] = labels
print(df['cluster_dbscan'].value_counts())
"""