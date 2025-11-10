import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def evaluar_hdbscan_parametros(X, min_cluster_sizes, min_samples_list, metric='jaccard'):
    """
    Evalúa diferentes combinaciones de hiperparámetros para HDBSCAN
    
    Parámetros clave:
    - min_cluster_size: Tamaño mínimo de cluster (muy importante)
    - min_samples: Controla qué tan conservador es el algoritmo
    """
    resultados = []
    total = len(min_cluster_sizes) * len(min_samples_list)
    counter = 0
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            counter += 1
            print(f"Evaluando {counter}/{total}: min_cluster_size={min_cluster_size}, min_samples={min_samples}...", end='\r')
            
            # Entrenar HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                core_dist_n_jobs=-1  # Usar todos los cores
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
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': (n_noise / len(labels)) * 100,
                'silhouette': silhouette,
                'min_cluster': min_size,
                'max_cluster': max_size,
                'mean_cluster': mean_size,
                'cluster_persistence': clusterer.cluster_persistence_ if hasattr(clusterer, 'cluster_persistence_') else None
            })
    
    print(" " * 100, end='\r')
    return pd.DataFrame(resultados)


def visualizar_evaluacion_hdbscan(df_resultados):
    """
    Visualiza los resultados de la evaluación de HDBSCAN
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pivot tables para heatmaps
    pivot_clusters = df_resultados.pivot(
        index='min_cluster_size', 
        columns='min_samples', 
        values='n_clusters'
    )
    
    pivot_noise = df_resultados.pivot(
        index='min_cluster_size', 
        columns='min_samples', 
        values='noise_pct'
    )
    
    pivot_silhouette = df_resultados.pivot(
        index='min_cluster_size', 
        columns='min_samples', 
        values='silhouette'
    )
    
    # 1. Número de clusters
    sns.heatmap(pivot_clusters, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[0, 0], cbar_kws={'label': 'N° Clusters'})
    axes[0, 0].set_title('Número de Clusters', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('min_samples', fontsize=11)
    axes[0, 0].set_ylabel('min_cluster_size', fontsize=11)
    
    # 2. Porcentaje de ruido
    sns.heatmap(pivot_noise, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=axes[0, 1], cbar_kws={'label': '% Ruido'})
    axes[0, 1].set_title('Porcentaje de Ruido', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('min_samples', fontsize=11)
    axes[0, 1].set_ylabel('min_cluster_size', fontsize=11)
    
    # 3. Silhouette Score
    sns.heatmap(pivot_silhouette, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=axes[1, 0], cbar_kws={'label': 'Silhouette'}, center=0)
    axes[1, 0].set_title('Silhouette Score (Jaccard)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('min_samples', fontsize=11)
    axes[1, 0].set_ylabel('min_cluster_size', fontsize=11)
    
    # 4. Tabla resumen de mejores configuraciones
    axes[1, 1].axis('off')
    
    # Top 5 configuraciones por silhouette
    top_configs = df_resultados.nlargest(5, 'silhouette')[
        ['min_cluster_size', 'min_samples', 'n_clusters', 'noise_pct', 'silhouette']
    ].round(3)
    
    table_data = [['min_cluster', 'min_samples', 'clusters', 'noise%', 'silhouette']]
    for _, row in top_configs.iterrows():
        table_data.append([
            f"{int(row['min_cluster_size'])}",
            f"{int(row['min_samples'])}",
            f"{int(row['n_clusters'])}",
            f"{row['noise_pct']:.1f}",
            f"{row['silhouette']:.3f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.18, 0.18, 0.15, 0.15, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Top 5 Configuraciones (por Silhouette)', 
                        fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


def visualizar_clusters_pca(X, labels, max_samples=5000):
    """
    Visualiza los clusters usando PCA
    """
    # Muestreo si es necesario
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    # Visualización
    plt.figure(figsize=(14, 8))
    
    # Separar ruido de clusters
    noise_mask = labels_sample == -1
    cluster_mask = ~noise_mask
    
    # Plot clusters
    if cluster_mask.any():
        scatter = plt.scatter(
            X_pca[cluster_mask, 0], 
            X_pca[cluster_mask, 1], 
            c=labels_sample[cluster_mask], 
            cmap='tab10', 
            alpha=0.6, 
            s=30,
            label='Clusters'
        )
        plt.colorbar(scatter, label='Cluster ID')
    
    # Plot ruido
    if noise_mask.any():
        plt.scatter(
            X_pca[noise_mask, 0], 
            X_pca[noise_mask, 1], 
            c='gray', 
            alpha=0.3, 
            s=10,
            label='Ruido'
        )
    
    n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
    n_noise = sum(noise_mask)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)', fontsize=12)
    plt.title(f'HDBSCAN: {n_clusters} Clusters, {n_noise} puntos de ruido', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analizar_clusters_hdbscan(df_original, labels, df_ingredientes, top_n=10):
    """
    Analiza las características de cada cluster
    """
    df_analisis = df_original.copy()
    df_analisis['cluster'] = labels
    
    print("="*80)
    print("ANÁLISIS DE CLUSTERS HDBSCAN")
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

def pipeline_hdbscan_recetas(df_ingredientes_onehot, df_original, 
                             min_cluster_sizes=None, min_samples_list=None):
    """
    Pipeline completo de clustering HDBSCAN para recetas
    
    Parámetros:
    -----------
    df_ingredientes_onehot : DataFrame
        Matriz binaria de ingredientes
    df_original : DataFrame
        DataFrame original con información de las recetas
    min_cluster_sizes : list
        Lista de valores para min_cluster_size a probar
    min_samples_list : list
        Lista de valores para min_samples a probar
    """
    
    print("="*80)
    print("CLUSTERING HDBSCAN PARA RECETAS (Métrica: Jaccard)")
    print("="*80)
    print(f"\nDimensiones del dataset: {df_ingredientes_onehot.shape}")
    print(f"Sparsity: {(df_ingredientes_onehot == 0).sum().sum() / df_ingredientes_onehot.values.size * 100:.2f}%")
    
    # Valores por defecto
    if min_cluster_sizes is None:
        min_cluster_sizes = [5, 10, 20, 30, 50, 100, 200, 300, 500]
    
    if min_samples_list is None:
        min_samples_list = [1, 3, 5, 10, 20, 30]
    
    print(f"\nParámetros a evaluar:")
    print(f"  • min_cluster_size: {min_cluster_sizes}")
    print(f"  • min_samples: {min_samples_list}")
    print(f"  • Total combinaciones: {len(min_cluster_sizes) * len(min_samples_list)}")
    
    # Convertir a array
    X = df_ingredientes_onehot.values
    
    # EVALUACIÓN DE HIPERPARÁMETROS
    print("\n" + "="*80)
    print("EVALUACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    df_resultados = evaluar_hdbscan_parametros(X, min_cluster_sizes, min_samples_list)
    
    print("\nRESULTADOS COMPLETOS:")
    print(df_resultados.to_string(index=False))
    
    # RECOMENDACIÓN
    print("\n" + "="*80)
    print("RECOMENDACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    # Filtrar configuraciones válidas (al menos 2 clusters y silhouette > 0)
    df_validos = df_resultados[
        (df_resultados['n_clusters'] >= 2) & 
        (df_resultados['silhouette'] > 0)
    ].copy()
    
    if len(df_validos) == 0:
        print("\n⚠️  ADVERTENCIA: No se encontraron configuraciones con clustering válido.")
        print("Considera ajustar los rangos de hiperparámetros.")
        return None, None, df_resultados
    
    # Normalizar métricas para selección
    df_validos['silhouette_norm'] = (df_validos['silhouette'] - df_validos['silhouette'].min()) / \
                                    (df_validos['silhouette'].max() - df_validos['silhouette'].min())
    
    df_validos['noise_norm'] = 1 - (df_validos['noise_pct'] / 100)  # Penalizar mucho ruido
    
    # Score combinado (70% silhouette, 30% bajo ruido)
    df_validos['score_total'] = 0.7 * df_validos['silhouette_norm'] + 0.3 * df_validos['noise_norm']
    
    mejor_config = df_validos.loc[df_validos['score_total'].idxmax()]
    
    print(f"\nCONFIGURACIÓN ÓPTIMA:")
    print(f"  • min_cluster_size: {int(mejor_config['min_cluster_size'])}")
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
    visualizar_evaluacion_hdbscan(df_resultados)
    
    # MODELO FINAL
    print("\n" + "="*80)
    print("ENTRENANDO MODELO FINAL")
    print("="*80)
    
    clusterer_final = hdbscan.HDBSCAN(
        min_cluster_size=int(mejor_config['min_cluster_size']),
        min_samples=int(mejor_config['min_samples']),
        metric='jaccard',
        core_dist_n_jobs=-1
    )
    
    labels_final = clusterer_final.fit_predict(X)
    
    # Visualizar
    print("\nGenerando visualización PCA...")
    visualizar_clusters_pca(X, labels_final)
    
    # Análisis
    analizar_clusters_hdbscan(df_original, labels_final, df_ingredientes_onehot, top_n=15)
    
    return clusterer_final, labels_final, df_resultados


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    """
    # Ejecutar pipeline completo
    modelo, labels, resultados = pipeline_hdbscan_recetas(
        df_ingredientes_onehot=df_ingredientes_onehot,
        df_original=df,
        min_cluster_sizes=[50, 100, 200, 300],
        min_samples_list=[5, 10, 20]
    )
    
    # Agregar labels al DataFrame original
    if labels is not None:
        df['cluster'] = labels
        
        # Ver distribución
        print(df['cluster'].value_counts())
    """
    
    print("Script listo para ejecutar.")
    print("\nPara usar, ejecuta:")
    print("modelo, labels, resultados = pipeline_hdbscan_recetas(df_ingredientes_onehot, df)")