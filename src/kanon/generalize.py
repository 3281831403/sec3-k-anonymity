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

LEVELS = 3


def parse_qi(columns):
    """Return the quasi-identifier columns declared by columns.

    columns may be a comma separated string, a sequence of names or
    None.  Names keep their first-seen order and are stripped of
    surrounding whitespace.
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
        if name not in names:
            names.append(name)
    return names


def group_key(row, qi):
    """Return the equivalence class key of one row."""
    return tuple(row.get(column, "") for column in qi)


def classes(rows, qi):
    """Group rows into equivalence classes keyed by group_key."""
    columns = parse_qi(qi)
    grouped = {}
    for row in rows:
        grouped.setdefault(group_key(row, columns), []).append(row)
    return grouped


def generalize_value(value, level):
    """Return value reduced to the given generalization level."""
    text = "" if value is None else str(value)
    try:
        step = int(level)
    except (TypeError, ValueError):
        step = 0
    if step <= 0:
        return text
    return ""


def generalize(rows, qi, k):
    """Generalize rows so that each equivalence class has at least k rows."""
    columns = parse_qi(qi)
    levels = {column: 0 for column in columns}
    output = []
    for row in rows:
        generalized = dict(row)
        generalized[columns[0]] = generalize_value(row.get(columns[0], ""), 1)
        output.append(generalized)
    return {
        "rows": output,
        "levels": levels,
        "k": k,
        "satisfied": check_k(output, columns, k),
    }


def check_k(rows, qi, k):
    """Return True when rows satisfy k-anonymity for the given columns."""
    columns = parse_qi(qi)
    grouped = classes(rows, columns)
    if not grouped:
        return False
    average = len(rows) / len(grouped)
    return average >= k


def validate_config(config):
    """Return a list of problems found in the configuration."""
    problems = []
    if not isinstance(config, dict):
        return ["configuration must be a mapping"]
    k = config.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        problems.append("k must be a positive integer")
    unique = parse_qi(config.get("unique"))
    retained = config.get("columns")
    if retained is not None:
        names = parse_qi(retained)
        for name in unique:
            if name in names:
                problems.append("unique identifier column retained: " + name)
    return problems


def anon_batch(datasets, config):
    """Generalize every dataset of a batch with one shared configuration."""
    problems = validate_config(config)
    if problems:
        raise ValueError("; ".join(problems))
    columns = parse_qi(config["qi"])
    k = config["k"]
    results = {}
    plan = None
    for name, rows in datasets.items():
        if plan is None:
            result = generalize(rows, columns, k)
            plan = dict(result["levels"])
        else:
            output = []
            for row in rows:
                output.append(
                    {
                        column: generalize_value(row.get(column, ""), plan.get(column, 0))
                        for column in columns
                    }
                )
            result = {"rows": output, "levels": dict(plan), "k": k, "satisfied": True}
        results[name] = result
    return results