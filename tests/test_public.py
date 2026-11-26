"""Public acceptance tests for the k-anonymity helpers."""

from collections import Counter

from kanon import (
    anon_batch,
    check_k,
    classes,
    generalize,
    parse_qi,
    validate_config,
)


def test_parse_qi_normalizes_and_rejects_empty():
    assert parse_qi(" zip , age , zip ") == ["zip", "age"]
    assert parse_qi(["age", " ", "zip", "age"]) == ["age", "zip"]
    assert parse_qi(None) == []
    assert parse_qi("   ") == []
    assert validate_config({"qi": "   ", "k": 3, "columns": []}) != []


def test_classes_group_unique_records():
    rows = [
        {"person_id": "p1", "zip": "100", "age": "20"},
        {"person_id": "p2", "zip": "100", "age": "20"},
        {"person_id": "p1", "zip": "100", "age": "20"},
        {"person_id": "p3", "zip": "200", "age": "30"},
    ]
    grouped = classes(rows, ["zip", "age"])
    assert len(grouped) == 2
    assert sorted(len(group) for group in grouped.values()) == [1, 2]
    assert sum(len(group) for group in grouped.values()) == 3
    assert set(grouped) == set(classes(rows, ["age", "zip"]))


def test_generalize_masks_all_quasi_identifiers():
    rows = [
        {"person_id": "p1", "zip": "11001", "age": "21"},
        {"person_id": "p2", "zip": "11002", "age": "22"},
        {"person_id": "p3", "zip": "11003", "age": "23"},
        {"person_id": "p4", "zip": "11004", "age": "24"},
    ]
    result = generalize(rows, ["zip", "age"], 2)
    assert result["satisfied"] is True
    assert result["k"] == 2
    output = result["rows"]
    assert len(output) == len(rows)
    assert all(set(row) == {"zip", "age"} for row in output)
    assert all("*" in value for row in output for value in row.values())
    counts = Counter(tuple(row[column] for column in ["zip", "age"]) for row in output)
    assert min(counts.values()) >= 2


def test_check_k_rejects_one_small_class():
    lopsided = [
        {"person_id": "a1", "zip": "100"},
        {"person_id": "a2", "zip": "100"},
        {"person_id": "a3", "zip": "100"},
        {"person_id": "a4", "zip": "100"},
        {"person_id": "a5", "zip": "100"},
        {"person_id": "b1", "zip": "200"},
    ]
    assert check_k(lopsided, ["zip"], 2) is False
    balanced = [
        {"person_id": "a1", "zip": "100"},
        {"person_id": "a2", "zip": "100"},
        {"person_id": "b1", "zip": "200"},
        {"person_id": "b2", "zip": "200"},
    ]
    assert check_k(balanced, ["zip"], 2) is True


def test_anon_batch_keeps_datasets_independent():
    datasets = {
        "first": [
            {"person_id": "a1", "zip": "11001", "age": "21"},
            {"person_id": "a2", "zip": "11002", "age": "22"},
            {"person_id": "a3", "zip": "11003", "age": "23"},
            {"person_id": "a4", "zip": "11004", "age": "24"},
        ],
        "second": [
            {"person_id": "b1", "zip": "51001", "age": "61"},
            {"person_id": "b2", "zip": "51002", "age": "62"},
            {"person_id": "b3", "zip": "52001", "age": "63"},
            {"person_id": "b4", "zip": "52002", "age": "64"},
        ],
    }
    config = {
        "qi": ["zip", "age"],
        "k": 2,
        "unique": ["person_id"],
        "columns": ["zip", "age"],
    }
    result = anon_batch(datasets, config)
    assert set(result) == {"first", "second"}
    for name, rows in datasets.items():
        counts = Counter(tuple(row[column] for column in ["zip", "age"]) for row in result[name]["rows"])
        assert min(counts.values()) >= 2
    assert result["second"]["rows"] == generalize(datasets["second"], ["zip", "age"], 2)["rows"]