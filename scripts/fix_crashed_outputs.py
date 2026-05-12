"""
fix_crashed_outputs.py
======================

Limpia los outputs de "Kernel crashed" y "NameError" que quedaron 
visibles en notebooks específicos sin afectar nada más.

Notebooks afectados:
- 01_eda.ipynb
- 02_features_and_targets.ipynb
- 07_optimizacion_hiperparametros.ipynb
- 13_comparacion_final_y_conclusiones.ipynb

NO ejecuta nada — solo elimina outputs problemáticos de las últimas 
celdas afectadas. Los modelos, datos y métricas YA están guardados 
correctamente en disco.

USO:
    python scripts/fix_crashed_outputs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Patrones que identifican un output crasheado
CRASH_PATTERNS = [
    "Kernel crashed while executing",
    "NameError",
    "KeyboardInterrupt",
    "Please review the code in the cell",
    "vscodeJupyterKernelCrash",
]


def is_crashed_output(output):
    """Verifica si un output contiene patrones de crash."""
    if not isinstance(output, dict):
        return False
    
    # Buscar en text (output normal)
    text = output.get("text", [])
    if isinstance(text, list):
        text = "".join(text)
    elif not isinstance(text, str):
        text = ""
    
    # Buscar en data (html, plain, etc)
    data = output.get("data", {})
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                value = "".join(str(v) for v in value)
            elif not isinstance(value, str):
                value = str(value)
            text += "\n" + value
    
    # Buscar en evalue/ename (errors)
    evalue = output.get("evalue", "")
    ename = output.get("ename", "")
    if isinstance(evalue, list):
        evalue = "".join(evalue)
    text += "\n" + str(evalue) + "\n" + str(ename)
    
    # Buscar en traceback
    traceback = output.get("traceback", [])
    if isinstance(traceback, list):
        text += "\n" + "\n".join(str(t) for t in traceback)
    
    return any(pattern.lower() in text.lower() for pattern in CRASH_PATTERNS)


def clean_notebook(nb_path):
    """Limpia outputs crasheados de un notebook."""
    print(f"\nProcesando: {nb_path.name}")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    cells_cleaned = 0
    outputs_removed = 0
    
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        
        outputs = cell.get("outputs", [])
        if not outputs:
            continue
        
        # Filtrar outputs crasheados
        clean_outputs = []
        for output in outputs:
            if is_crashed_output(output):
                outputs_removed += 1
                print(f"  Celda {i}: removido output con crash")
            else:
                clean_outputs.append(output)
        
        if len(clean_outputs) != len(outputs):
            cell["outputs"] = clean_outputs
            cells_cleaned += 1
            
            # Si la celda quedó completamente vacía de outputs, 
            # también limpiar execution_count para que no muestre [error]
            if not clean_outputs:
                cell["execution_count"] = None
    
    if cells_cleaned > 0:
        # Guardar notebook limpio
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  ✅ Limpiados {outputs_removed} outputs en {cells_cleaned} celdas")
        print(f"  ✅ Notebook guardado: {nb_path}")
    else:
        print(f"  ℹ️  Sin outputs crasheados encontrados")
    
    return cells_cleaned


def main():
    print("=" * 70)
    print("LIMPIEZA DE OUTPUTS CRASHEADOS")
    print("=" * 70)
    
    # Notebooks específicos afectados (según el reporte de Mateo)
    notebooks = [
        "01_eda.ipynb",
        "02_features_and_targets.ipynb",
        "07_optimizacion_hiperparametros.ipynb",
        "13_comparacion_final_y_conclusiones.ipynb",
    ]
    
    nb_dir = ROOT / "notebooks"
    if not nb_dir.exists():
        print(f"❌ No existe {nb_dir}")
        sys.exit(1)
    
    total_cleaned = 0
    for nb_name in notebooks:
        nb_path = nb_dir / nb_name
        if not nb_path.exists():
            print(f"\n⚠️  No existe: {nb_path}")
            continue
        total_cleaned += clean_notebook(nb_path)
    
    print()
    print("=" * 70)
    print(f"✅ Total de celdas limpiadas: {total_cleaned}")
    print("=" * 70)
    print()
    print("Siguiente paso: re-build del Jupyter Book")
    print("  jupyter-book build .")
    print()


if __name__ == "__main__":
    main()
