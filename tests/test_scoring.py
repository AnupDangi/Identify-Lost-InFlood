from dvi import scoring


def test_unknown_metadata_returns_neutral_score():
    am, pm = {}, {}
    assert scoring.sex_score(am, pm) == 0.5
    assert scoring.age_score(am, pm) == 0.5
    assert scoring.height_score(am, pm) == 0.5
    assert scoring.date_score(am, pm) == 0.5
    assert scoring.location_score(am, pm) == 0.5


def test_ages_close_together_rank_higher_than_wildly_different():
    am = {"age_min": 30, "age_max": 30}
    close_pm = {"age_min": 31, "age_max": 31}
    far_pm = {"age_min": 70, "age_max": 70}
    assert scoring.age_score(am, close_pm) > scoring.age_score(am, far_pm)


def test_pm_found_before_am_missing_is_penalized_not_crashed():
    am = {"event_date_normalized": "2026-06-01", "calendar_type": "AD"}
    pm = {"event_date_normalized": "2026-01-01", "calendar_type": "AD"}  # before AM's date
    score = scoring.date_score(am, pm)
    assert 0.0 <= score < 0.5  # implausible but not a hard filter


def test_date_score_unparseable_is_neutral_not_penalized():
    am = {"event_date": "garbage"}
    pm = {"event_date": "also garbage"}
    assert scoring.date_score(am, pm) == 0.5


def test_sex_score_blends_metadata_and_vision_signals():
    # Both signals agree -> full score.
    am = {"sex": "male", "detected_sex": "male"}
    pm = {"sex": "male", "detected_sex": "male"}
    assert scoring.sex_score(am, pm) == 1.0
    # Metadata agrees but vision disagrees -> soft penalty, not a hard filter.
    pm_vision_mismatch = {"sex": "male", "detected_sex": "female"}
    mixed = scoring.sex_score(am, pm_vision_mismatch)
    assert 0.0 < mixed < 1.0


def test_metadata_score_weights_sum_to_one():
    assert abs(sum(scoring.WEIGHTS.values()) - 1.0) < 1e-9


def test_fusion_score_metadata_only_when_no_face():
    result = scoring.fusion_score(None, 0.7)
    assert result == 0.7


def test_fusion_score_blends_face_and_metadata():
    result = scoring.fusion_score(1.0, 0.0, face_weight=0.6)
    assert abs(result - 0.6) < 1e-9


# --- Phase 6: four explicit conflict signals, unknown never counts as conflict ---

def test_conflict_signals_unknown_never_flagged():
    am = {"sex": "male"}  # no detected_sex
    pm = {"sex": "female"}  # no detected_sex
    conflicts = scoring.compute_conflicts(am, pm)
    assert conflicts["am_metadata_vision_conflict"] is False
    assert conflicts["pm_metadata_vision_conflict"] is False
    assert conflicts["pair_vision_conflict"] is False
    assert conflicts["pair_metadata_conflict"] is True  # both known, both recorded, differ


def test_conflict_signals_are_independent():
    am = {"sex": "male", "detected_sex": "female"}  # am metadata/vision conflict
    pm = {"sex": "male", "detected_sex": "male"}     # pm metadata/vision agree
    conflicts = scoring.compute_conflicts(am, pm)
    assert conflicts["am_metadata_vision_conflict"] is True
    assert conflicts["pm_metadata_vision_conflict"] is False
    assert conflicts["pair_metadata_conflict"] is False  # am.sex == pm.sex (male == male)
    assert conflicts["pair_vision_conflict"] is True      # am.detected_sex (female) != pm.detected_sex (male)
