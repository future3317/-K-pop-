from kpop_scope.taxonomy import all_labels, labels_by_group, load_taxonomy


def test_taxonomy_loads_required_groups():
    tax = load_taxonomy()
    assert {"Style", "Mood", "Arrangement", "Structure"}.issubset(tax)
    assert "dance-pop" in labels_by_group()["Style"]
    assert "dance break" in all_labels()
    for items in tax.values():
        for item in items:
            assert item["name"]
            assert item["description_zh"]
            assert isinstance(item["audio_evidence"], list)
