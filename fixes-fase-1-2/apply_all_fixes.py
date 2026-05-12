"""
apply_all_fixes.py
==================

Script único que aplica TODAS las correcciones de la auditoría de Claude Code.

Ejecuta:
  1. Crea carpeta scripts/ si no existe
  2. Mueve fix_*.py de la raíz a scripts/ (si están en la raíz)
  3. Borra carpetas duplicadas notebook-14/ y fixes-opcion-a/
  4. Actualiza .gitignore para evitar que vuelvan a aparecer
  5. Aplica fix del GARCH (corrige el bug de RMSE 0.0145 → 0.0067)
  6. Imprime instrucciones manuales que faltan

Después de ejecutar este script, los siguientes pasos los haces tú a mano:
  - Editar README.md (rellenar 5 TBDs) — te paso el texto exacto.
  - Editar _config.yml (añadir exclude_patterns) — te paso el bloque.
  - Editar 3 notebooks (NB 04, 05, 06) añadiendo celdas markdown.
  - Re-ejecutar última celda del NB 07 (crash de kernel).
  - jupyter-book build .
  - git commit + push

Uso:
    python apply_all_fixes.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def step_1_setup_scripts_dir():
    """Crear scripts/ y mover archivos."""
    banner("Paso 1: setup de directorio scripts/")
    scripts_dir = ROOT / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    print(f"✅ Directorio scripts/ listo")
    
    # Mover archivos si están en la raíz
    for fname in ["fix_propagate_optimized.py",
                   "fix_tabla_estandar_vs_optimizado.py"]:
        src = ROOT / fname
        dst = scripts_dir / fname
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"✅ Movido: {fname} → scripts/")
        elif dst.exists():
            print(f"   Ya estaba en scripts/: {fname}")


def step_2_clean_duplicated_dirs():
    """Borrar notebook-14/ y fixes-opcion-a/."""
    banner("Paso 2: limpieza de directorios duplicados")
    
    for dirname in ["notebook-14", "fixes-opcion-a"]:
        dirpath = ROOT / dirname
        if dirpath.exists():
            shutil.rmtree(dirpath)
            print(f"✅ Borrado: {dirname}/")
        else:
            print(f"   No existía: {dirname}/")


def step_3_update_gitignore():
    """Actualizar .gitignore."""
    banner("Paso 3: actualizar .gitignore")
    
    gitignore_path = ROOT / ".gitignore"
    new_entries = [
        "# Carpetas temporales del proceso de desarrollo",
        "notebook-14/",
        "fixes-opcion-a/",
        "",
        "# Builds del Jupyter Book",
        "_build/",
        "",
        "# Caches",
        ".pytest_cache/",
        "__pycache__/",
        ".ipynb_checkpoints/",
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            existing = f.read()
    else:
        existing = ""
    
    # Añadir solo entradas que no estén ya
    to_add = []
    for entry in new_entries:
        if entry and entry not in existing:
            to_add.append(entry)
    
    if to_add:
        with open(gitignore_path, "a") as f:
            f.write("\n# Añadido por apply_all_fixes.py\n")
            f.write("\n".join(to_add))
            f.write("\n")
        print(f"✅ Añadidas {len(to_add)} entradas a .gitignore")
    else:
        print(f"   .gitignore ya tenía todo lo necesario")


def step_4_run_garch_fix():
    """Ejecutar el fix del GARCH."""
    banner("Paso 4: aplicar fix del GARCH (corrección de bug)")
    
    fix_script = ROOT / "scripts" / "fix_garch_benchmark.py"
    if not fix_script.exists():
        print(f"❌ No existe {fix_script}")
        print(f"   Asegúrate de tener el script en scripts/")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(fix_script)],
            cwd=str(ROOT),
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print(result.stdout)
            print("✅ Fix del GARCH aplicado exitosamente")
            return True
        else:
            print(f"❌ Fix del GARCH falló:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error ejecutando fix del GARCH: {e}")
        return False


def step_5_print_manual_instructions():
    """Instrucciones manuales que faltan."""
    banner("Paso 5: instrucciones manuales restantes")
    
    print("""
Los siguientes pasos son MANUALES (requieren tu juicio):

📝 5.1 Editar README.md líneas 92-96 (TBDs)
   Abre README.md y reemplaza las 5 líneas TBD con:
   
   - Mejor benchmark econométrico: GARCH(1,1) corregido (RMSE 0.00674, R² -0.08)
   - Mejor ML clásico: Ridge (RMSE 0.00658, R² test -0.030, R² CV +0.238)
   - Mejor modelo avanzado: LSTM (RMSE 0.00662) — empate técnico con CatBoost
   - HVRF modelo original: RMSE 0.00682, NO supera a Ridge (DM p=0.0122).
     Empate estadístico con ensamble naive 50/50 (DM p=0.0925).
   - Decisión final: Ridge como predictor de RMSE puntual; HVRF como 
     detector complementario de régimen.

📝 5.2 Editar _config.yml: añadir exclude_patterns
   En la sección sphinx: config:, añade:
   
       exclude_patterns:
         - '.conda/**'
         - '.pytest_cache/**'
         - 'notebook-14/**'
         - 'fixes-opcion-a/**'
         - 'notebooks/INSTRUCCIONES.md'
         - 'ARCHITECTURE.md'
         - 'README.md'

📝 5.3 Editar NB 04 (regresión): añadir celda markdown
   Justo después de la celda 1 (intro), añade una celda markdown nueva:
   
   ### Decisión metodológica: horizonte de evaluación
   
   Los targets `target_vol_7`, `target_vol_14` y `target_vol_21` se construyen
   en el notebook 02 (NB 02), pero la evaluación de modelos se concentra en
   `target_vol_7` por ser el horizonte primario (definido como `PRIMARY_HORIZON=7`
   en `src/config.py`). Esta elección permite comparar uniformemente todos los
   modelos sobre una métrica única.
   
   Replicar la metodología para los horizontes 14 y 21 es trivial: basta con
   cambiar `y_trainval = trainval[f"target_vol_{h}"]` y re-entrenar. La 
   infraestructura está lista, pero la decisión consciente fue concentrar el
   análisis estadístico en un solo horizonte para mantener el alcance del 
   proyecto manejable.

📝 5.4 Editar NB 05 (clasificación): añadir celda markdown
   Justo después de la celda intro, añade una celda markdown nueva:
   
   ### Justificación de la métrica principal
   
   Dado que `P(target_class=1) ≈ 0.486` (clases casi balanceadas) y los
   errores tipo I y II tienen costo simétrico en nuestro problema, **F1-score**
   se elige como métrica primaria por equilibrar precision y recall. **AUC**
   se reporta como métrica complementaria por su robustez al umbral de decisión
   y por permitir comparaciones formales mediante el test de DeLong.

📝 5.5 Editar NB 06 (balanceo): añadir celda markdown sobre ADASYN NaN
   Después de la celda donde se ejecuta ADASYN y aparece NaN, añade:
   
   ### Sobre el resultado NaN de ADASYN
   
   ADASYN falla con `ValueError: No samples will be generated` porque
   nuestro dataset tiene clases casi balanceadas (P(clase=1)=0.486).
   ADASYN está diseñado para datasets MUY desbalanceados (típicamente
   relación 1:10 o peor) y necesita identificar muestras minoritarias 
   "difíciles" rodeadas mayoritariamente por la clase contraria. En 
   nuestro caso, no hay tal asimetría, así que el algoritmo no encuentra
   candidatos válidos.
   
   **Esto NO es un bug**, es una propiedad esperada de ADASYN sobre datasets
   balanceados. Se reporta como NaN en la tabla y se compara contra SMOTE
   (que sí funciona aunque no aporta) y `class_weight='balanced'` (el más
   estable de los tres).

📝 5.6 Re-ejecutar última celda del NB 07
   Abre notebooks/07_optimizacion_hiperparametros.ipynb en VSCode.
   Ve a la ÚLTIMA celda de código.
   Click "Run cell" (Shift+Enter).
   Espera que termine limpiamente.
   Ctrl+S para guardar (esto borra el traceback rojo del crash anterior).

📝 5.7 Re-ejecutar NB 09 y NB 13
   Con el GARCH corregido, los notebooks 09 y 13 deben re-leer las
   predicciones. Run All:
   - notebooks/09_comparacion_estadistica.ipynb
   - notebooks/13_comparacion_final_y_conclusiones.ipynb

📝 5.8 Re-build del Jupyter Book
   jupyter-book build .

📝 5.9 Commit final y push
   git add .
   git commit -m "Aplicar correcciones de auditoría: fix GARCH, limpieza, README, docs"
   git push
""")


def main():
    banner("APLICANDO CORRECCIONES DE LA AUDITORÍA")
    print(f"Working dir: {ROOT}")
    
    step_1_setup_scripts_dir()
    step_2_clean_duplicated_dirs()
    step_3_update_gitignore()
    garch_ok = step_4_run_garch_fix()
    step_5_print_manual_instructions()
    
    banner("RESUMEN")
    print(f"✅ Pasos automáticos completados")
    if not garch_ok:
        print(f"⚠️  Fix del GARCH NO se aplicó automáticamente")
        print(f"   Ejecuta manualmente: python scripts/fix_garch_benchmark.py")
    print(f"📝 Pasos manuales pendientes: ver instrucciones arriba")
    print(f"\n⏱️  Tiempo estimado restante (manual): 15-20 min")


if __name__ == "__main__":
    main()
