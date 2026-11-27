"""k-anonymity helpers for publishing tabular microdata."""

from __future__ import annotations

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

# Deepest generalization step.  At this level every non-empty value is
# replaced by a single ``*`` (the top of the masking hierarchy).
LEVELS = 3


def parse_qi(columns):
    """Return the quasi-identifier columns declared by ``columns``.

    ``columns`` may be a comma separated string, a sequence of names or
    ``None``.  Names are stripped of surrounding whitespace, empty /
    whitespace-only names are dropped and the first-seen order is kept
    while duplicates are removed.
    """
    if columns is None:
        return []
    if isinstance(columns, str):
        parts = columns.split(",")
    elif isinstance(columns, (set, frozenset)):
        parts = sorted(columns)
    else:
        parts = list(columns)
    names = []
    for part in parts:
        name = str(part).strip()
        if name and name not in names:
            names.append(name)
    return names


def _cell_text(value):
    """Normalize one cell to a hashable, comparable string."""
    if value is None:
        return ""
    return str(value)


def group_key(row, qi):
    """Return the equivalence class key of one row.

    Columns are canonicalized by name (sorted) so that the key is
    independent of the order in which quasi-identifiers are declared.
    Values stay bound to their column name because the sorted position
    always corresponds to the same sorted column.
    """
    columns = sorted(parse_qi(qi))
    return tuple(_cell_text(row.get(column, "")) for column in columns)


def _record_signature(row):
    """Build an order-independent, hashable signature of a whole record."""
    return tuple(sorted((str(key), repr(value)) for key, value in row.items()))


def _dedupe_rows(rows):
    """Drop repeated identical records, keeping the first occurrence."""
    seen = set()
    unique = []
    for row in rows:
        signature = _record_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique


def classes(rows, qi, dedupe=True):
    """Group rows into equivalence classes keyed by :func:`group_key`.

    By default verbatim-duplicate records count only once, matching the
    publishing rule "one output row per distinct input record".  Pass
    ``dedupe=False`` to count every supplied row, which is the right
    choice for already-generalized output where identical projections
    represent distinct underlying records.
    """
    columns = parse_qi(qi)
    source = _dedupe_rows(rows) if dedupe else rows
    grouped = {}
    for row in source:
        grouped.setdefault(group_key(row, columns), []).append(row)
    return grouped


def generalize_value(value, level):
    """Return ``value`` reduced to the given generalization level.

    Each step masks one more trailing character with ``*``.  At
    :data:`LEVELS` (and beyond) the whole value collapses to a single
    ``*``, which is also the top marker for an empty / missing value so
    that the highest level merges every value into one class.  Level
    ``0`` and unreadable levels return the original text.
    """
    text = "" if value is None else str(value)
    try:
        step = int(level)
    except (TypeError, ValueError):
        step = 0
    if step <= 0:
        return text
    if not text or step >= LEVELS:
        return "*"
    keep = max(len(text) - step, 0)
    return text[:keep] + "*" * (len(text) - keep)


def _valid_k(k):
    """Return True for an allowed k value (an integer of at least 2)."""
    return isinstance(k, int) and not isinstance(k, bool) and k >= 2


def _project(rows, columns, level):
    """Project rows onto the QI columns generalized at one uniform level."""
    return [
        {
            column: generalize_value(row.get(column, ""), level)
            for column in columns
        }
        for row in rows
    ]


def _minimum_class_size(rows, columns):
    """Return the smallest equivalence class size among projected rows."""
    grouped = classes(rows, columns, dedupe=False)
    if not grouped:
        return 0
    return min(len(members) for members in grouped.values())


def generalize(rows, qi, k):
    """Generalize rows until every equivalence class holds at least ``k``.

    Exact duplicate records are collapsed first (a repeated record counts
    once).  Generalization then proceeds in synchronous rounds: every
    quasi-identifier column is raised by one masking step each round, so
    all quasi-identifiers are covered and iteration continues until every
    class reaches size ``k`` or the hierarchy tops out.

    The output keeps only the generalized quasi-identifier columns and
    has one row per distinct input record.  The result is a pure function
    of the inputs, so repeating a call yields an identical result.

    ``k`` must be an integer of at least 2; invalid arguments raise
    :class:`ValueError`.  When a valid ``k`` cannot be reached (for
    example fewer than ``k`` distinct records exist), the maximally
    generalized rows are returned with ``satisfied`` set to ``False``.
    """
    columns = parse_qi(qi)
    if not columns:
        raise ValueError("at least one quasi-identifier column is required")
    if not _valid_k(k):
        raise ValueError("k must be an integer of at least 2")

    records = _dedupe_rows(rows)
    if not records:
        return {
            "rows": [],
            "levels": {column: 0 for column in columns},
            "k": k,
            "satisfied": False,
        }

    output = []
    level = 0
    for level in range(0, LEVELS + 1):
        output = _project(records, columns, level)
        if _minimum_class_size(output, columns) >= k:
            return {
                "rows": output,
                "levels": {column: level for column in columns},
                "k": k,
                "satisfied": True,
            }

    # Reached the top of the hierarchy without meeting k: report honestly.
    return {
        "rows": output,
        "levels": {column: LEVELS for column in columns},
        "k": k,
        "satisfied": False,
    }


def check_k(rows, qi, k):
    """Return True when every equivalence class holds at least ``k`` rows.

    Rows are counted exactly as supplied: on generalized output, repeated
    and identical projections stand for distinct underlying records and
    each counts toward its class.
    """
    columns = parse_qi(qi)
    if not columns or not _valid_k(k):
        return False
    grouped = classes(rows, columns, dedupe=False)
    if not grouped:
        return False
    return all(len(members) >= k for members in grouped.values())


def validate_config(config):
    """Return a list of problems found in the configuration."""
    problems = []
    if not isinstance(config, dict):
        return ["configuration must be a mapping"]

    if not parse_qi(config.get("qi")):
        problems.append("at least one quasi-identifier column is required")

    k = config.get("k")
    if not _valid_k(k):
        problems.append("k must be an integer of at least 2")

    unique = parse_qi(config.get("unique"))
    retained = config.get("columns")
    if retained is not None:
        names = parse_qi(retained)
        for name in unique:
            if name in names:
                problems.append("unique identifier column retained: " + name)
    return problems


def anon_batch(datasets, config):
    """Generalize every dataset of a batch independently.

    One shared configuration is validated once, but each dataset is
    generalized from its own records with its own equivalence classes and
    levels; no plan or state is shared between datasets.
    """
    problems = validate_config(config)
    if problems:
        raise ValueError("; ".join(problems))
    columns = parse_qi(config["qi"])
    k = config["k"]
    return {
        name: generalize(rows, columns, k)
        for name, rows in datasets.items()
    }
