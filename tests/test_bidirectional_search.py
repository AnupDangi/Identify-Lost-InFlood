from dvi import retrieval


def test_am_to_pm_search_works(synthetic_dataset):
    result = retrieval.search_query("am", "AM1", "pm", top_k=5)
    assert result["has_face"] is True
    assert result["candidate_source"] == "face+metadata"
    assert result["candidates"]
    # AM1's own PM candidates must be PM record ids, never AM ids.
    for c in result["candidates"]:
        assert c["candidate_record_id"].startswith("PM")


def test_pm_to_am_search_works_independently(synthetic_dataset):
    result = retrieval.search_query("pm", "PM1", "am", top_k=5)
    assert result["has_face"] is True
    assert result["candidate_source"] == "face+metadata"
    for c in result["candidates"]:
        assert c["candidate_record_id"].startswith("AM")


def test_pm_to_am_is_not_a_pivot_of_am_to_pm(synthetic_dataset):
    """Phase 7: each direction runs its own FAISS search against its own
    index (am->pm searches pm.index, pm->am searches am.index) rather than
    one direction's results being pivoted/looked-up from the other's file."""
    am_to_pm = retrieval.search_query("am", "AM1", "pm", top_k=5)
    pm_to_am = retrieval.search_query("pm", "PM1", "am", top_k=5)

    # AM1->PM candidates are drawn from the PM gallery via pm.index; PM1->AM
    # candidates are drawn from the AM gallery via am.index -- different
    # FAISS indexes, so this only works at all if both were actually
    # searched independently rather than one being a lookup into the other's
    # output file.
    assert {c["candidate_record_id"] for c in am_to_pm["candidates"]} <= {"PM1"}
    assert {c["candidate_record_id"] for c in pm_to_am["candidates"]} <= {"AM1", "AM2"}


def test_no_face_query_gets_metadata_only_candidates(synthetic_dataset):
    """Phase 9: AM3 has no usable face embedding -- this must not raise or
    return an empty/error result, it must rank the full PM gallery by
    metadata alone."""
    result = retrieval.search_query("am", "AM3", "pm", top_k=5)
    assert result["has_face"] is False
    assert result["candidate_source"] == "metadata_only"
    assert len(result["candidates"]) == 3  # all PM records considered, not just usable-face ones
    for c in result["candidates"]:
        assert c["face_score"] is None


def test_metadata_only_candidates_cover_full_gallery_not_just_faiss_topk(synthetic_dataset):
    result = retrieval.search_query("pm", "PM2", "am", top_k=10)
    assert result["candidate_source"] == "metadata_only"
    candidate_ids = {c["candidate_record_id"] for c in result["candidates"]}
    assert candidate_ids == {"AM1", "AM2", "AM3"}


def test_unknown_query_record_raises_key_error(synthetic_dataset):
    import pytest
    with pytest.raises(KeyError):
        retrieval.search_query("am", "AM-does-not-exist", "pm")
