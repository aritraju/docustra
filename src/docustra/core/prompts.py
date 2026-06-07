"""
Prompt Registry — versioned prompt management
══════════════════════════════════════════════
Prompts are stored in YAML files under ``prompts/<version>/``.
The active version is set via ``Settings.prompt_version`` (default: "v1").

Usage
-----
    from docustra.core.prompts import get_prompt

    # Returns a LangChain ChatPromptTemplate ready for use
    template = get_prompt("shared", "citation_rag")
    chain = template | llm

Versioning
----------
To create a new version:
  1. Copy ``prompts/v1/`` to ``prompts/v2/``
  2. Edit the YAML keys you want to change
  3. Set ``PROMPT_VERSION=v2`` in your .env
  4. Old responses in evals remain reproducible because the version is logged
     in ``RAGResponse.metadata["prompt_version"]``
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import ChatPromptTemplate

from docustra.core.config import get_settings

# Repository root → prompts/ directory
# File lives at: <repo>/src/docustra/core/prompts.py  (3 parents up = <repo>/src, 4 = <repo>)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_PROMPTS_DIR = _REPO_ROOT / "prompts"


@functools.lru_cache(maxsize=128)
def _load_yaml(version: str, module: str) -> dict[str, Any]:
    """Load and cache a prompt YAML file."""
    path = _PROMPTS_DIR / version / f"{module}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Available versions: {[d.name for d in _PROMPTS_DIR.iterdir() if d.is_dir()]}"
        )
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt(module: str, key: str, version: str | None = None) -> ChatPromptTemplate:
    """
    Load a versioned prompt template.

    Parameters
    ----------
    module : str
        YAML file name without extension (e.g. "shared", "adaptive", "corrective")
    key : str
        Key within the YAML file (e.g. "citation_rag", "router")
    version : str | None
        Prompt version string (e.g. "v1"). Defaults to ``Settings.prompt_version``.

    Returns
    -------
    ChatPromptTemplate
        Ready for use in a LangChain chain.
    """
    version = version or get_settings().prompt_version
    data = _load_yaml(version, module)

    if key not in data:
        available = [k for k in data if not k.startswith("_") and k != "metadata"]
        raise KeyError(
            f"Prompt key '{key}' not found in {module}.yaml (v{version}). "
            f"Available keys: {available}"
        )

    prompt_def = data[key]
    if not isinstance(prompt_def, dict):
        raise ValueError(f"Prompt '{key}' in {module}.yaml must be a dict with 'system'/'human' keys")

    messages: list[tuple[str, str]] = []
    if "system" in prompt_def:
        messages.append(("system", prompt_def["system"].rstrip()))
    if "human" in prompt_def:
        messages.append(("human", prompt_def["human"].rstrip()))

    if not messages:
        raise ValueError(f"Prompt '{key}' in {module}.yaml has no 'system' or 'human' keys")

    return ChatPromptTemplate.from_messages(messages)


def get_prompt_version() -> str:
    """Return the active prompt version from settings."""
    return get_settings().prompt_version


def invalidate_cache() -> None:
    """Clear the prompt cache — useful in tests when swapping versions."""
    _load_yaml.cache_clear()
