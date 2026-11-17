import subprocess
import sys

# Lista ordenada de todas las etapas del pipeline Medallion
pipeline_steps = [
    # 1. Bronze (Ingesta)
    ("src.bronze.01a_ingest_flights", "run_ingest_flights"),
    ("src.bronze.01b_ingest_weather", "run_ingest_weather"),
    ("src.bronze.01c_ingest_holidays", "run_ingest_holidays"),
    
    # 2. Silver (Limpieza y Dimensiones)
    ("src.silver.02a_clean_dimensions", "run_clean_dimensions"),
    ("src.silver.02b_clean_weather", "run_clean_weather"),
    ("src.silver.02c_clean_flights", "run_clean_flights"),
    
    # 3. Silver Unificada
    ("src.silver.03_build_silver_unified", "run_build_silver_unified"),
    
    # 4. Gold (Modelo Estrella y ML)
    ("src.gold.04a_build_gold_star", "run_build_gold_star"),
    ("src.gold.04b_build_gold_ml", "run_build_gold_ml"),
    
    # 5. Gobernanza UC (en local simula el registro o validación)
    ("src.gold.05_apply_uc_governance", "run_apply_uc_governance"),
]

def main():
    print("=== INICIANDO PIPELINE COMPLETO EN LOCAL ===")
    for module_path, func_name in pipeline_steps:
        print(f"\n--- Ejecutando: {module_path}.{func_name} ---")
        cmd = [sys.executable, "notebooks/_job_runner.py", module_path, func_name]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[ERROR] El pipeline falló en la etapa: {module_path}.{func_name}")
            sys.exit(result.returncode)
            
    print("\n=== ¡PIPELINE LOCAL COMPLETADO CON ÉXITO! ===")

if __name__ == "__main__":
    main()