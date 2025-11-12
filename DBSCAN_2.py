import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import umap
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def evaluar_dbscan_parametros(X, eps_list, min_samples_list, metric='jaccard'):
    """
    Evalúa diferentes combinaciones de hiperparámetros para DBSCAN
    
    Parámetros clave:
    - eps: Radio máximo de vecindad (epsilon)
    - min_samples: Número mínimo de puntos para formar un cluster
    """
    resultados = []
    total = len(eps_list) * len(min_samples_list)
    counter = 0
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            counter += 1
            print(f"Evaluando {counter}/{total}: eps={eps:.3f}, min_samples={min_samples}...", end='\r')
            
            # Entrenar DBSCAN
            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                n_jobs=-1
            )
            
            labels = clusterer.fit_predict(X)
            
            # Calcular métricas
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Silhouette solo si hay al menos 2 clusters y no todo es ruido
            if n_clusters >= 2 and n_noise < len(labels):
                # Calcular silhouette sin puntos de ruido
                labels_no_noise = labels[labels != -1]
                X_no_noise = X[labels != -1]
                
                if len(labels_no_noise) > 0 and len(set(labels_no_noise)) > 1:
                    silhouette = silhouette_score(X_no_noise, labels_no_noise, metric=metric)
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            # Cluster sizes (sin contar ruido)
            if n_clusters > 0:
                cluster_sizes = [sum(labels == i) for i in set(labels) if i != -1]
                min_size = min(cluster_sizes) if cluster_sizes else 0
                max_size = max(cluster_sizes) if cluster_sizes else 0
                mean_size = np.mean(cluster_sizes) if cluster_sizes else 0
            else:
                min_size = max_size = mean_size = 0
            
            resultados.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': (n_noise / len(labels)) * 100,
                'silhouette': silhouette,
                'min_cluster': min_size,
                'max_cluster': max_size,
                'mean_cluster': mean_size
            })
    
    print(" " * 100, end='\r')
    return pd.DataFrame(resultados)


def estimar_eps_optimo(X, k=5, metric='jaccard', sample_size=1000):
    """
    Estima un rango apropiado de eps usando k-nearest neighbors
    Calcula la distancia al k-ésimo vecino más cercano
    """
    print(f"\nEstimando rango de eps (muestreando {sample_size} puntos)...")
    
    # Muestreo para acelerar
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Calcular matriz de distancias
    print("Calculando distancias...")
    distances = pdist(X_sample, metric=metric)
    dist_matrix = squareform(distances)
    
    # Para cada punto, encontrar distancia al k-ésimo vecino
    k_distances = []
    for i in range(len(X_sample)):
        # Ordenar distancias (excluyendo la distancia a sí mismo)
        sorted_distances = np.sort(dist_matrix[i])
        if len(sorted_distances) > k:
            k_distances.append(sorted_distances[k])
    
    k_distances = np.array(k_distances)
    k_distances = np.sort(k_distances)
    
    # Visualizar k-distance plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances, linewidth=2)
    plt.xlabel('Puntos ordenados', fontsize=12)
    plt.ylabel(f'{k}-distancia (Jaccard)', fontsize=12)
    plt.title(f'K-Distance Plot para Estimación de Eps (k={k})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Sugerir rango
    percentiles = [50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(k_distances, p)
        plt.axhline(y=val, color='r', linestyle='--', alpha=0.5, 
                   label=f'Percentil {p}: {val:.3f}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Sugerencias
    eps_min = np.percentile(k_distances, 50)
    eps_max = np.percentile(k_distances, 95)
    
    print(f"\nRango sugerido de eps: [{eps_min:.3f}, {eps_max:.3f}]")
    print(f"Percentil 50: {eps_min:.3f}")
    print(f"Percentil 75: {np.percentile(k_distances, 75):.3f}")
    print(f"Percentil 90: {np.percentile(k_distances, 90):.3f}")
    print(f"Percentil 95: {eps_max:.3f}")
    
    return eps_min, eps_max


def visualizar_evaluacion_dbscan(df_resultados):
    """
    Visualiza los resultados de la evaluación de DBSCAN
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pivot tables para heatmaps
    pivot_clusters = df_resultados.pivot(
        index='eps', 
        columns='min_samples', 
        values='n_clusters'
    )
    
    pivot_noise = df_resultados.pivot(
        index='eps', 
        columns='min_samples', 
        values='noise_pct'
    )
    
    pivot_silhouette = df_resultados.pivot(
        index='eps', 
        columns='min_samples', 
        values='silhouette'
    )
    
    # 1. Número de clusters
    sns.heatmap(pivot_clusters, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[0, 0], cbar_kws={'label': 'N° Clusters'})
    axes[0, 0].set_title('Número de Clusters', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('min_samples', fontsize=11)
    axes[0, 0].set_ylabel('eps', fontsize=11)
    
    # 2. Porcentaje de ruido
    sns.heatmap(pivot_noise, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=axes[0, 1], cbar_kws={'label': '% Ruido'})
    axes[0, 1].set_title('Porcentaje de Ruido', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('min_samples', fontsize=11)
    axes[0, 1].set_ylabel('eps', fontsize=11)
    
    # 3. Silhouette Score
    sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=axes[1, 0], cbar_kws={'label': 'Silhouette'}, center=0)
    axes[1, 0].set_title('Silhouette Score (Jaccard)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('min_samples', fontsize=11)
    axes[1, 0].set_ylabel('eps', fontsize=11)
    
    # 4. Tabla resumen de mejores configuraciones
    axes[1, 1].axis('off')
    
    # Top 5 configuraciones por silhouette
    top_configs = df_resultados.nlargest(5, 'silhouette')[
        ['eps', 'min_samples', 'n_clusters', 'noise_pct', 'silhouette']
    ].round(3)
    
    table_data = [['eps', 'min_samples', 'clusters', 'noise%', 'silhouette']]
    for _, row in top_configs.iterrows():
        table_data.append([
            f"{row['eps']:.3f}",
            f"{int(row['min_samples'])}",
            f"{int(row['n_clusters'])}",
            f"{row['noise_pct']:.1f}",
            f"{row['silhouette']:.3f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.15, 0.18, 0.15, 0.15, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de header
    for i in range(5):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Top 5 Configuraciones (por Silhouette)', 
                        fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


def visualizar_clusters_pca_umap(X, labels, max_samples=5000):
    """
    Visualiza los clusters usando PCA y UMAP en paralelo
    """
    # Muestreo si es necesario
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Separar ruido de clusters
    noise_mask = labels_sample == -1
    cluster_mask = ~noise_mask
    n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
    n_noise = sum(noise_mask)
    
    # === SUBPLOT 1: PCA ===
    print("Calculando PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    # Plot clusters
    if cluster_mask.any():
        scatter1 = axes[0].scatter(
            X_pca[cluster_mask, 0], 
            X_pca[cluster_mask, 1], 
            c=labels_sample[cluster_mask], 
            cmap='tab10', 
            alpha=0.6, 
            s=30
        )
        plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
    
    # Plot ruido
    if noise_mask.any():
        axes[0].scatter(
            X_pca[noise_mask, 0], 
            X_pca[noise_mask, 1], 
            c='gray', 
            alpha=0.3, 
            s=10,
            label='Ruido'
        )
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)', fontsize=12)
    axes[0].set_title(f'PCA: {n_clusters} Clusters, {n_noise} ruido', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # === SUBPLOT 2: UMAP ===
    print("Calculando UMAP (esto puede tardar unos minutos)...")
    reducer = umap.UMAP(
        n_components=2,
        metric='jaccard',
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False
    )
    X_umap = reducer.fit_transform(X_sample)
    
    # Plot clusters
    if cluster_mask.any():
        scatter2 = axes[1].scatter(
            X_umap[cluster_mask, 0], 
            X_umap[cluster_mask, 1], 
            c=labels_sample[cluster_mask], 
            cmap='tab10', 
            alpha=0.6, 
            s=30
        )
        plt.colorbar(scatter2, ax=axes[1], label='Cluster ID')
    
    # Plot ruido
    if noise_mask.any():
        axes[1].scatter(
            X_umap[noise_mask, 0], 
            X_umap[noise_mask, 1], 
            c='gray', 
            alpha=0.3, 
            s=10,
            label='Ruido'
        )
    
    axes[1].set_xlabel('UMAP 1', fontsize=12)
    axes[1].set_ylabel('UMAP 2', fontsize=12)
    axes[1].set_title(f'UMAP: {n_clusters} Clusters, {n_noise} ruido', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Visualización completada.")


def analizar_clusters_dbscan(df_original, labels, df_ingredientes, top_n=10):
    """
    Analiza las características de cada cluster
    """
    df_analisis = df_original.copy()
    df_analisis['cluster'] = labels
    
    print("="*80)
    print("ANÁLISIS DE CLUSTERS DBSCAN")
    print("="*80)
    
    # Análisis de ruido
    n_noise = sum(labels == -1)
    if n_noise > 0:
        print(f"\n{'='*80}")
        print(f"RUIDO - {n_noise} recetas ({n_noise/len(df_analisis)*100:.2f}%)")
        print(f"{'='*80}")
        print("Estas recetas no se ajustan bien a ningún cluster.")
    
    # Análisis por cluster
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
            
        recetas_cluster = df_analisis[df_analisis['cluster'] == cluster_id]
        n_recetas = len(recetas_cluster)
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id} - {n_recetas} recetas ({n_recetas/len(df_analisis)*100:.2f}%)")
        print(f"{'='*80}")
        
        # Ingredientes más frecuentes
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

def pipeline_dbscan_recetas(df_ingredientes_onehot, df_original, 
                            eps_list=None, min_samples_list=None,
                            estimar_eps=True):
    """
    Pipeline completo de clustering DBSCAN para recetas
    
    Parámetros:
    -----------
    df_ingredientes_onehot : DataFrame
        Matriz binaria de ingredientes
    df_original : DataFrame
        DataFrame original con información de las recetas
    eps_list : list
        Lista de valores para eps a probar
    min_samples_list : list
        Lista de valores para min_samples a probar
    estimar_eps : bool
        Si True, estima el rango óptimo de eps antes de evaluar
    """
    
    print("="*80)
    print("CLUSTERING DBSCAN PARA RECETAS (Métrica: Jaccard)")
    print("="*80)
    print(f"\nDimensiones del dataset: {df_ingredientes_onehot.shape}")
    print(f"Sparsity: {(df_ingredientes_onehot == 0).sum().sum() / df_ingredientes_onehot.values.size * 100:.2f}%")
    
    # Convertir a array
    X = df_ingredientes_onehot.values
    
    # ESTIMACIÓN DE EPS
    if estimar_eps:
        print("\n" + "="*80)
        print("ESTIMACIÓN DE EPS ÓPTIMO")
        print("="*80)
        eps_min, eps_max = estimar_eps_optimo(X, k=5, sample_size=1000)
        
        if eps_list is None:
            # Generar lista de eps basada en la estimación
            eps_list = np.linspace(eps_min, eps_max, 5).round(3).tolist()
            print(f"\nEps generado automáticamente: {eps_list}")
    
    # Valores por defecto si no se proporcionaron
    if eps_list is None:
        eps_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    if min_samples_list is None:
        min_samples_list = [5, 10, 20, 30]
    
    print(f"\nParámetros a evaluar:")
    print(f"  • eps: {eps_list}")
    print(f"  • min_samples: {min_samples_list}")
    print(f"  • Total combinaciones: {len(eps_list) * len(min_samples_list)}")
    
    # EVALUACIÓN DE HIPERPARÁMETROS
    print("\n" + "="*80)
    print("EVALUACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    df_resultados = evaluar_dbscan_parametros(X, eps_list, min_samples_list)
    
    print("\nRESULTADOS COMPLETOS:")
    print(df_resultados.to_string(index=False))
    
    # RECOMENDACIÓN
    print("\n" + "="*80)
    print("RECOMENDACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    # Filtrar configuraciones válidas
    df_validos = df_resultados[
        (df_resultados['n_clusters'] >= 2) & 
        (df_resultados['silhouette'] > 0)
    ].copy()
    
    if len(df_validos) == 0:
        print("\n⚠️  ADVERTENCIA: No se encontraron configuraciones con clustering válido.")
        print("Considera ajustar los rangos de eps y min_samples.")
        print("\nIntentando con configuraciones alternativas (al menos 1 cluster)...")
        
        # Intentar con criterios más permisivos
        df_validos = df_resultados[
            (df_resultados['n_clusters'] >= 1)
        ].copy()
        
        if len(df_validos) == 0:
            print("\n❌ No se encontró ninguna configuración que genere clusters.")
            print("Sugerencias:")
            print("  - Aumenta el rango de eps (valores más grandes)")
            print("  - Reduce min_samples")
            return None, None, df_resultados
    
    # Normalizar métricas
    silhouette_range = df_validos['silhouette'].max() - df_validos['silhouette'].min()
    if silhouette_range > 0:
        df_validos['silhouette_norm'] = (df_validos['silhouette'] - df_validos['silhouette'].min()) / silhouette_range
    else:
        df_validos['silhouette_norm'] = 0.5  # Valor neutro si todos son iguales
    
    df_validos['noise_norm'] = 1 - (df_validos['noise_pct'] / 100)
    
    # Score combinado (70% silhouette, 30% bajo ruido)
    df_validos['score_total'] = 0.7 * df_validos['silhouette_norm'] + 0.3 * df_validos['noise_norm']
    
    # Verificar que hay scores válidos
    if df_validos['score_total'].isna().all():
        print("\n❌ No se pudo calcular score válido.")
        return None, None, df_resultados
    
    mejor_config = df_validos.loc[df_validos['score_total'].idxmax()]
    
    print(f"\nCONFIGURACIÓN ÓPTIMA:")
    print(f"  • eps: {mejor_config['eps']:.3f}")
    print(f"  • min_samples: {int(mejor_config['min_samples'])}")
    print(f"\nRESULTADOS:")
    print(f"  • Número de clusters: {int(mejor_config['n_clusters'])}")
    print(f"  • Puntos de ruido: {int(mejor_config['n_noise'])} ({mejor_config['noise_pct']:.2f}%)")
    print(f"  • Silhouette Score: {mejor_config['silhouette']:.4f}")
    print(f"  • Tamaño medio cluster: {mejor_config['mean_cluster']:.0f} recetas")
    
    # VISUALIZACIONES
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)
    visualizar_evaluacion_dbscan(df_resultados)
    
    # MODELO FINAL
    print("\n" + "="*80)
    print("ENTRENANDO MODELO FINAL")
    print("="*80)
    
    clusterer_final = DBSCAN(
        eps=mejor_config['eps'],
        min_samples=int(mejor_config['min_samples']),
        metric='jaccard',
        n_jobs=-1
    )
    
    labels_final = clusterer_final.fit_predict(X)
    
    # Visualizar
    print("\nGenerando visualización PCA + UMAP...")
    visualizar_clusters_pca_umap(X, labels_final)
    
    # Análisis
    analizar_clusters_dbscan(df_original, labels_final, df_ingredientes_onehot, top_n=15)
    
    return clusterer_final, labels_final, df_resultados

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
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    """
    # Ejecutar pipeline completo con estimación automática de eps
    modelo, labels, resultados = pipeline_dbscan_recetas(
        df_ingredientes_onehot=df_ingredientes_onehot,
        df_original=df,
        estimar_eps=True  # Estima el rango óptimo de eps
    )
    
    # O con parámetros personalizados
    modelo, labels, resultados = pipeline_dbscan_recetas(
        df_ingredientes_onehot=df_ingredientes_onehot,
        df_original=df,
        eps_list=[0.4, 0.5, 0.6, 0.7],
        min_samples_list=[5, 10, 20],
        estimar_eps=False
    )
    
    # O con unos parámetros rápidos
    labels = dbscan_rapido(
        df_ingredientes_onehot=df_ingredientes_onehot,
        df_original=df_sample,
        eps=0.3,
        min_samples=3,
        metric='jaccard'

    # O explorar múltiples parámetros rápidamente
    resultados = explorar_parametros(
        df_ingredientes_onehot, df_sample,
        eps_list=[0.2, 0.3, 0.4, 0.5, 0.6],
        min_samples_list=[3, 5, 10, 15]
        )
    )


    # Agregar labels al DataFrame original
    if labels is not None:
        df['cluster'] = labels
        print(df['cluster'].value_counts())
    """
    
    print("Script listo para ejecutar.")
    print("\nPara usar, ejecuta:")
    print("modelo, labels, resultados = pipeline_dbscan_recetas(df_ingredientes_onehot, df)")