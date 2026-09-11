import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Safely loads a YAML configuration file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(True, f"Configuration file not found at {path.absolute()}")
        
    with open(path, "r") as file:
        return yaml.safe_load(file)

def save_config(config: dict, config_path: str) -> None:
    """Safely writes a dictionary back to the YAML configuration file."""
    with open(config_path, "w") as stream:
        try:
            # default_flow_style=False keeps the clean, indented layout
            # sort_keys=False keeps your parameters in the original order
            yaml.safe_dump(config, stream, default_flow_style=False, sort_keys=False)
        except yaml.YAMLError as exc:
            print(f"Error writing YAML file: {exc}")
            raise