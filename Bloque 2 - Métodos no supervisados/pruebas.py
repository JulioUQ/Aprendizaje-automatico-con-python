
# =============================================================
# Clustering con DBSCAN sobre la muestra
# =============================================================

# --- Inicialización del modelo DBSCAN ---
# eps controla el radio de vecindad (ajústalo con cuidado)
# min_samples define cuántos puntos forman una región densa
dbscan = DBSCAN(
    eps=0.25,           # radio; ajusta según la densidad observada en UMAP
    min_samples=10,     # número mínimo de puntos por cluster
    metric='cosine',    # métrica coherente con los embeddings semánticos
    n_jobs=-1           # usa todos los núcleos disponibles
)

# --- Entrenamiento sobre la muestra ---
labels_sample = dbscan.fit_predict(embeddings_sample)

# --- Estadísticas básicas ---
n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
n_noise = np.sum(labels_sample == -1)
print(f"Clusters encontrados: {n_clusters}")
print(f"Recetas clasificadas como ruido: {n_noise} ({n_noise/len(labels_sample):.2%})")

# =============================================================
# Visualización de los clusters (UMAP de la muestra)
# =============================================================

reducer_sample = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42
)

embeddings_2d_sample = reducer_sample.fit_transform(embeddings_sample)

# --- Colores de clusters ---
palette = sns.color_palette('tab20', n_colors=max(10, n_clusters))
colors = [
    (0.7, 0.7, 0.7, 0.5) if label == -1 else palette[int(label % len(palette))]
    for label in labels_sample
]

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d_sample[:, 0], embeddings_2d_sample[:, 1], s=6, c=colors, alpha=0.8)
plt.title("Clustering de recetas (muestra) con DBSCAN - métrica del coseno")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(r'../Visualizaciones/dbscan_clusters_umap_muestra.png', dpi=300)
plt.show()

# =============================================================
# Análisis de los clusters obtenidos (muestra)
# =============================================================

sns.countplot(
    x=[label for label in labels_sample if label != -1],
    order=pd.Series(labels_sample[labels_sample != -1]).value_counts().index,
    color='steelblue'
)
plt.title("Distribución de tamaños de clusters (DBSCAN - muestra)")
plt.xlabel("Cluster ID")
plt.ylabel("Número de recetas en la muestra")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(r'../Visualizaciones/dbscan_tamano_clusters_muestra.png', dpi=300)
plt.show()

# =============================================================
# Recetas representativas por cluster (en la muestra)
# =============================================================

# --- Extrae nombres de la muestra ---
names_sample = df.iloc[sample_idx]['name'].reset_index(drop=True)

def cluster_representatives_sample(names, embeddings, labels, top_k=3):
    reps = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue  # Ignorar ruido
        idx = np.where(labels == c)[0]
        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        dists = cosine_distances(cluster_emb, centroid).ravel()
        nearest = np.argsort(dists)[:top_k]
        reps[c] = names.iloc[idx[nearest]].tolist()
    return reps

reps_sample = cluster_representatives_sample(names_sample, embeddings_sample, labels_sample)

print("\n=== Recetas más representativas por cluster (DBSCAN - muestra) ===")
for c, names in reps_sample.items():
    print(f"\nCluster {c} ({np.sum(labels_sample==c)} recetas en la muestra):")
    for name in names:
        print(f"  - {name}")
