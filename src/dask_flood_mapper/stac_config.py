"""Configuration management for Dask Flood Mapper."""

import shutil
from importlib.resources import files
from pathlib import Path

import yaml
from appdirs import user_config_dir

CONFIG_FILE: str = "config.yaml"
CONFIG_PATH = files("dask_flood_mapper").joinpath(CONFIG_FILE)
USER_CONFIG_DIR = Path(user_config_dir("dask_flood_mapper"))


class ConfigError(Exception):
    """Custom exception for configuration errors."""


def make_user_config_path(user_config_dir: Path) -> Path:
    """Create the path for the user configuration file."""
    return user_config_dir / CONFIG_FILE


def get_user_config(user_config_dir: Path = USER_CONFIG_DIR) -> Path:
    """Get the user configuration file path."""
    user_config_path: Path = make_user_config_path(user_config_dir)
    if not user_config_path.exists():
        msg: str = "User configuration does not exist yet."
        msg += " Use set_user_config to set configuration."
        raise ConfigError(msg)
    return user_config_path


def set_user_config(user_config_dir: Path = USER_CONFIG_DIR) -> Path:
    """Set the user configuration file, creating it if it does not exist."""
    if not user_config_dir.exists():
        user_config_dir.mkdir(parents=True)
    user_config_path: Path = make_user_config_path(user_config_dir)
    if not user_config_path.exists():
        shutil.copy(CONFIG_PATH, user_config_path)  # type: ignore
    return user_config_path


def load_config(user_config_dir: Path = USER_CONFIG_DIR) -> dict:
    """Load the configuration from the user config file."""
    yaml_file = set_user_config(user_config_dir)
    with yaml_file.open() as file:
        return yaml.safe_load(file)
