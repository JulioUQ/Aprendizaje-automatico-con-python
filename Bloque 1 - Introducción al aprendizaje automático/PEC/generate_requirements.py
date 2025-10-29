from pathlib import Path
import pkg_resources

# Lista de librerías utilizadas en tu notebook
packages = [
    "pandas",
    "matplotlib",
    "seaborn",
    "plotly",
    "folium",
    "numpy",
    "scikit-learn",
    "imbalanced-learn"
]

# Crear archivo requirements.txt con versiones instaladas si existen
requirements = []
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        requirements.append(f"{pkg}=={version}")
    except pkg_resources.DistributionNotFound:
        requirements.append(pkg)  # si no está instalada, dejamos sin versión

Path("requirements.txt").write_text("\n".join(requirements))
print("✅ Archivo 'requirements.txt' generado correctamente.")
