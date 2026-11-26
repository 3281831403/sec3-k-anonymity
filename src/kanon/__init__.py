"""k-anonymity tooling for publishing microdata."""

from .generalize import (
    anon_batch,
    check_k,
    classes,
    generalize,
    generalize_value,
    group_key,
    parse_qi,
    validate_config,
)

__all__ = [
    "parse_qi",
    "group_key",
    "classes",
    "generalize_value",
    "generalize",
    "check_k",
    "validate_config",
    "anon_batch",
]

__version__ = "0.1.0"