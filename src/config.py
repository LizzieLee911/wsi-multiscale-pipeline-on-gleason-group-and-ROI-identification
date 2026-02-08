"""Configuration loading, environment selection, and path resolution."""

import os
from pathlib import Path

import yaml


_DEFAULT_ENV = "nscc"


def get_env(cli_env=None):
    """Return the active environment name.

    Priority: *cli_env* argument > ``BASELINE_ENV`` env‑var > ``"nscc"``.
    """
    if cli_env:
        return cli_env
    return os.environ.get("BASELINE_ENV", _DEFAULT_ENV)


def _resolve_paths(node, work_root):
    """Recursively join every leaf string value to *work_root*."""
    if isinstance(node, dict):
        return {k: _resolve_paths(v, work_root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v, work_root) for v in node]
    if isinstance(node, str):
        return Path(work_root) / node
    return node


def _find_config_dir(config_dir=None):
    """Return an absolute ``Path`` to the config directory.

    If *config_dir* is ``None``, defaults to ``<project_root>/config``.
    """
    if config_dir is not None:
        return Path(config_dir)
    return Path(__file__).resolve().parent.parent / "config"


def load_paths(env=None, config_dir=None):
    """Load ``paths.yaml`` for *env* and resolve all relative paths.

    Returns a nested dict whose leaf values are absolute ``Path`` objects.
    """
    config_dir = _find_config_dir(config_dir)
    env = env or get_env()

    with open(config_dir / "paths.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if env not in raw:
        raise KeyError(
            f"Environment '{env}' not found in paths.yaml. "
            f"Available: {list(raw.keys())}"
        )

    env_cfg = raw[env]
    work_root = env_cfg.pop("work_root")

    resolved = _resolve_paths(env_cfg, work_root)
    resolved["work_root"] = Path(work_root)
    return resolved


def load_params(config_dir=None):
    """Load ``params.yaml`` and return the raw dict."""
    config_dir = _find_config_dir(config_dir)

    with open(config_dir / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
