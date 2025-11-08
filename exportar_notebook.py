import sys, os
import nbformat
from nbconvert import HTMLExporter, MarkdownExporter, PDFExporter
from tkinter import Tk, filedialog

def export_notebook(ipynb, formato="html", ejecutar=True, salida=None):
    """
    Exporta un notebook de Jupyter (.ipynb) a HTML, PDF o Markdown sin incluir el código.

    Parámetros:
    -----------
    ipynb : str
        Ruta al archivo .ipynb.
    formato : str
        Formato de salida: 'html', 'pdf' o 'markdown'.
    ejecutar : bool
        Si True, ejecuta el notebook antes de exportar.
    salida : str o None
        Ruta completa del archivo de salida (sin extensión). Si es None, se pregunta al usuario.
    """
    # Leer el notebook
    nb = nbformat.read(ipynb, as_version=4)

    # Ejecutar si se desea
    if ejecutar:
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(ipynb) or '.'}})

    # Elegir exportador
    if formato == "html":
        exporter = HTMLExporter()
    elif formato in ("md", "markdown"):
        exporter = MarkdownExporter()
    elif formato == "pdf":
        exporter = PDFExporter()
    else:
        raise ValueError("❌ Formato no soportado: usa html, pdf o markdown")

    exporter.exclude_input = True
    body, _ = exporter.from_notebook_node(nb)

    # Si no se especificó salida, preguntar al usuario
    if salida is None:
        Tk().withdraw()  # Oculta la ventana principal de Tkinter
        salida = filedialog.asksaveasfilename(
            title="Guardar notebook exportado como...",
            defaultextension=f".{formato}",
            filetypes=[(f"{formato.upper()} files", f"*.{formato}"), ("Todos los archivos", "*.*")]
        )
        if not salida:
            print("❌ Exportación cancelada.")
            return

    # Guardar archivo
    with open(salida, "w", encoding="utf-8") as f:
        f.write(body)

    print(f"✔ Exportado correctamente a: {salida}")


if __name__ == "__main__":
    # Si no se pasa argumento, se abre un selector de archivo
    if len(sys.argv) < 2:
        print("📂 Selecciona el notebook que deseas exportar...")
        Tk().withdraw()
        ipynb = filedialog.askopenfilename(
            title="Selecciona un archivo Jupyter Notebook",
            filetypes=[("Jupyter Notebooks", "*.ipynb")]
        )
        if not ipynb:
            print("❌ No se seleccionó ningún archivo.")
            sys.exit(1)
    else:
        ipynb = sys.argv[1]

    # Si el formato no se pasa, se pregunta al usuario
    formato = sys.argv[2] if len(sys.argv) > 2 else None
    if not formato:
        formato = input("Introduce el formato de exportación (html/pdf/markdown): ").strip().lower()

    export_notebook(ipynb, formato)
