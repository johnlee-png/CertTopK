"""Configuration loading utilities."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent
