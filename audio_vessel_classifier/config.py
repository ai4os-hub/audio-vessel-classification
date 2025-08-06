"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the API interface.
Do not misuse this module to define variables that are not CONSTANTS.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""

import builtins
import logging
import os
from importlib import metadata
from pathlib import Path

import yaml

homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf_path = os.path.join(homedir, "etc", "config.yaml")
with open(conf_path, "r") as f:
    CONF = yaml.safe_load(f)


def check_conf(conf=CONF):
    """
    Checks for configuration parameters
    """
    for group, val in sorted(conf.items()):
        for g_key, g_val in sorted(val.items()):
            gg_keys = g_val.keys()

            if g_val["value"] is None:
                continue

            if "type" in gg_keys:
                var_type = getattr(builtins, g_val["type"])
                if not isinstance(g_val["value"], var_type):
                    raise TypeError(
                        "The selected value for {} must be a {}.".format(
                            g_key, g_val["type"]
                        )
                    )

            if ("choices" in gg_keys) and (
                g_val["value"] not in g_val["choices"]
            ):
                raise ValueError(
                    "The selected value for {} is not an available choice."
                    .format(g_key)
                )

            if "range" in gg_keys:
                if (
                    g_val["range"][0] is not None
                    and g_val["range"][0] > g_val["value"]
                ):
                    raise ValueError(
                        "The selected value for {} is lower than the "
                        "minimal possible value.".format(g_key)
                    )

                if (
                    g_val["range"][1] != "None"
                    and g_val["range"][1] < g_val["value"]
                ):
                    raise ValueError(
                        "The selected value for {} is higher than the "
                        "maximal possible value.".format(g_key)
                    )


check_conf()


def get_conf_dict(conf=CONF):
    """
    Return configuration as dict
    """
    conf_d = {}
    for group, val in conf.items():
        conf_d[group] = {}
        for g_key, g_val in val.items():
            conf_d[group][g_key] = g_val["value"]

    return conf_d


# Constants
API_NAME = "audio_vessel_classifier"
DEFAULT_METADATA_FILENAME = "ai4-metadata.yml"

# Get BASE_DIR of the application and possible metadata env variable
BASE_DIR = Path(__file__).resolve().parents[1]
AI4_METADATA_DIR = os.getenv(f"{API_NAME.upper()}_AI4_METADATA_DIR")


# Fallback logic to locate ai4-metadata.yml
def find_metadata_path():
    # 1. Check env var
    if AI4_METADATA_DIR and os.path.isfile(
        os.path.join(AI4_METADATA_DIR, DEFAULT_METADATA_FILENAME)
    ):
        return AI4_METADATA_DIR

    # 2. Check BASE_DIR directory
    possible_path = os.path.join(BASE_DIR, DEFAULT_METADATA_FILENAME)
    if os.path.isfile(possible_path):
        return BASE_DIR

    # 3. Check subfolder with API_NAME
    subdir = os.path.join(BASE_DIR, API_NAME)
    if os.path.isfile(os.path.join(subdir, DEFAULT_METADATA_FILENAME)):
        return subdir

    # 4. Check parent folder
    parent_dir = os.path.abspath(os.path.join(BASE_DIR, "..", API_NAME))
    if os.path.isfile(os.path.join(parent_dir, DEFAULT_METADATA_FILENAME)):
        return parent_dir

    raise FileNotFoundError(
        f"Could not find {DEFAULT_METADATA_FILENAME} in expected locations."
    )


# Try to load YAML metadata
try:
    AI4_METADATA_DIR = find_metadata_path()
    metadata_file_path = os.path.join(
        AI4_METADATA_DIR, DEFAULT_METADATA_FILENAME
    )
    with open(metadata_file_path, "r", encoding="utf-8") as stream:
        AI4_METADATA = yaml.safe_load(stream)
except Exception as e:
    raise RuntimeError(f"Error loading AI4 metadata: {e}")

# Try to load package metadata from pyproject.toml
try:
    PACKAGE_METADATA = metadata.metadata(API_NAME)
except metadata.PackageNotFoundError:
    raise RuntimeError(
        f"Package metadata for '{API_NAME}' not found. Is it installed?"
    )

# Build PROJECT_METADATA dict
try:
    PROJECT_METADATA = {
        "name": PACKAGE_METADATA["Name"],
        "description": AI4_METADATA.get(
            "description", "No description provided."
        ),
        "license": PACKAGE_METADATA["License"],
        "version": PACKAGE_METADATA["Version"],
        "url": PACKAGE_METADATA.get("Project-URL", "N/A"),
    }

    # Parse authors and emails
    _emails_list = PACKAGE_METADATA.get("Author-email", "").split(", ")
    _emails = (
        dict(map(lambda s: s[:-1].split(" <"), _emails_list))
        if _emails_list[0]
        else {}
    )
    PROJECT_METADATA["author-email"] = _emails
    PROJECT_METADATA["author"] = (
        ", ".join(_emails.keys()) if _emails else "Unknown"
    )
except Exception as e:
    raise RuntimeError(f"Error building PROJECT_METADATA: {e}")

# Logging configuration
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper(), logging.INFO)
