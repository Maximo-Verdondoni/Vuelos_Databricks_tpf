"""Corre el pipeline completo en local, etapa por etapa, igual que los Jobs de Databricks.

Uso (desde cualquier carpeta):
    python run_pipeline_local.py                  # pipeline de datos (01a a 05)
    python run_pipeline_local.py --ml             # datos + modelo (06 y 07)
    python run_pipeline_local.py --desde 03       # retoma desde una etapa
    python run_pipeline_local.py --solo-ml        # solo 06 y 07 (Gold B ya generado)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Raíz del repo: la carpeta que tiene conf/parameters.yaml
REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / "conf" / "parameters.yaml").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

# Mismo orden que el DAG del Job de datos
PIPELINE_STEPS = [
    # 1. Bronze (ingesta)
    "src.bronze.01a_ingest_f1_kaggle",
    "src.bronze.01b_ingest_f2_weather",
    "src.bronze.01c_ingest_f3_holidays",
    # 2. Silver (limpieza y dimensiones)
    "src.silver.02a_clean_dimensions",
    "src.silver.02b_clean_weather",
    "src.silver.02c_clean_flights",
    # 3. Silver unificada
    "src.silver.03_build_silver_unified",
    # 4. Gold (modelo estrella y dataset de ML)
    "src.gold.04a_build_gold_star",
    "src.gold.04b_build_gold_ml",
    # 5. Gobernanza (en local se omite sola: no hay Unity Catalog)
    "src.gold.05_apply_uc_governance",
]

# Mismo orden que el Job de ML
ML_STEPS = [
    "src.ml.06_train_model",
    "src.ml.07_batch_scoring",
]


def _step_id(module_path: str) -> str:
    """'src.silver.03_build_silver_unified' -> '03'"""
    return module_path.rsplit(".", 1)[-1].split("_", 1)[0]


def run_step(module_path: str) -> float:
    print(f"\n--- Ejecutando: {module_path} ---", flush=True)
    start = time.time()
    # python -m: arranca desde la raíz del repo, así 'from src...' funciona
    result = subprocess.run([sys.executable, "-m", module_path], cwd=REPO_ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] El pipeline falló en la etapa: {module_path} ({elapsed:.0f} s)")
        print(f"Para retomar desde acá: python run_pipeline_local.py --desde {_step_id(module_path)}")
        sys.exit(result.returncode)
    print(f"--- OK: {module_path} ({elapsed:.0f} s) ---", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Medallion completo en local")
    parser.add_argument("--ml", action="store_true", help="Corre también el modelo (06 y 07)")
    parser.add_argument("--solo-ml", action="store_true", help="Corre solo el modelo (06 y 07)")
    parser.add_argument("--desde", help="Retoma desde una etapa, por ejemplo 02c o 03")
    args = parser.parse_args()

    if args.solo_ml:
        steps = list(ML_STEPS)
    else:
        steps = PIPELINE_STEPS + (ML_STEPS if args.ml else [])

    if args.desde:
        ids = [_step_id(s) for s in steps]
        if args.desde not in ids:
            sys.exit(f"Etapa '{args.desde}' no encontrada. Opciones: {', '.join(ids)}")
        steps = steps[ids.index(args.desde):]

    print(f"=== INICIANDO PIPELINE EN LOCAL ({len(steps)} etapas) ===")
    print(f"Raíz del repo: {REPO_ROOT}")

    total = 0.0
    for module_path in steps:
        total += run_step(module_path)

    print(f"\n=== ¡PIPELINE LOCAL COMPLETADO CON ÉXITO! ({total / 60:.1f} min) ===")


if __name__ == "__main__":
    main()