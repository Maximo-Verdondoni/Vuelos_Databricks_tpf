from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str = "conf/parameters.yaml", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Carga la configuración desde YAML y aplica sobreescrituras de entorno/Job."""
    path = Path(config_path)
    if not path.exists():
        # Fallback por si se ejecuta desde notebooks/
        path = Path("..") / config_path

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                if "." in key:
                    section, subkey = key.split(".", 1)
                    config.setdefault(section, {})[subkey] = value
                else:
                    config[key] = value

    return config