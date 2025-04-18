import shutil
from importlib.resources import files
from pathlib import Path

import yaml
from appdirs import user_config_dir

CONFIG_FILE = "config.yaml"
CONFIG_PATH = files("dask_flood_mapper").joinpath(CONFIG_FILE)
USER_CONFIG_DIR = Path(user_config_dir("dask_flood_mapper"))


def make_user_config_path(user_config_dir):
    return user_config_dir / CONFIG_FILE


def get_user_config(user_config_dir=USER_CONFIG_DIR):
    user_config_path = make_user_config_path(user_config_dir)
    if not user_config_path.exists():
        raise Exception(
            "User configuration does not exist yet."
            + " Use set_user_config to set configuration."
        )
    return user_config_path


def set_user_config(user_config_dir=USER_CONFIG_DIR):
    if not user_config_dir.exists():
        user_config_dir.mkdir(parents=True)
    user_config_path = make_user_config_path(user_config_dir)
    if not user_config_path.exists():
        shutil.copy(CONFIG_PATH, user_config_path)
    return user_config_path


def load_config(user_config_dir=USER_CONFIG_DIR):
    yaml_file = set_user_config(user_config_dir)
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)
