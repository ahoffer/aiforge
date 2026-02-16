"""Model adapter registry.

Maps model name prefixes to adapter classes. select_adapter() picks
the right one by longest-prefix match and caches the result so each
model name resolves at most once.
"""

import logging

from adapters.base import ModelAdapter
from adapters.qwen25coder import Qwen25CoderAdapter
from adapters.qwen3 import Qwen3Adapter


log = logging.getLogger(__name__)

_REGISTRY: list[tuple[str, type[ModelAdapter]]] = [
    ("qwen2.5-coder", Qwen25CoderAdapter),
    ("qwen3", Qwen3Adapter),
]

_cache: dict[str, ModelAdapter] = {}


def select_adapter(model_name: str) -> ModelAdapter:
    """Return a cached adapter instance for the given model name.

    Matches the longest registered prefix. Unknown models get the
    passthrough base adapter.
    """
    if model_name in _cache:
        return _cache[model_name]

    best_cls = ModelAdapter
    best_len = 0
    for prefix, cls in _REGISTRY:
        if model_name.startswith(prefix) and len(prefix) > best_len:
            best_cls = cls
            best_len = len(prefix)

    adapter = best_cls()
    _cache[model_name] = adapter
    log.info("selected adapter %s for model %s", best_cls.__name__, model_name)
    return adapter
