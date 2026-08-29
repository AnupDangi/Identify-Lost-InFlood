from dvi.quality import (
    QUALITY_GOOD,
    QUALITY_MODERATE,
    QUALITY_POOR,
    QUALITY_UNUSABLE,
    assess_quality,
)


def test_no_face_hard_requirements_stay_unusable():
    result = assess_quality(detector_score=None, face_w=None, face_h=None)
    assert result.usable is False
    assert result.quality_band == QUALITY_UNUSABLE


def test_face_too_small_is_unusable():
    result = assess_quality(detector_score=0.9, face_w=10, face_h=10, min_face_size=40)
    assert result.usable is False
    assert "face_too_small" in result.quality_reasons


def test_low_detector_confidence_is_unusable():
    result = assess_quality(detector_score=0.1, face_w=100, face_h=100, min_det_score=0.5)
    assert result.usable is False
    assert "low_detector_confidence" in result.quality_reasons


def test_strong_face_is_good_band():
    result = assess_quality(detector_score=0.95, face_w=200, face_h=200, blur_score=500,
                             landmarks_available=True, min_face_size=40, min_det_score=0.5,
                             min_blur_score=100)
    assert result.usable is True
    assert result.quality_band == QUALITY_GOOD
    assert result.quality_reasons == []


def test_borderline_face_is_moderate_or_poor_but_still_usable():
    result = assess_quality(detector_score=0.55, face_w=45, face_h=45, blur_score=10,
                             min_face_size=40, min_det_score=0.5, min_blur_score=100)
    assert result.usable is True
    assert result.quality_band in (QUALITY_MODERATE, QUALITY_POOR)
    assert result.quality_reasons  # informational, not empty


def test_blur_threshold_disabled_by_default_never_rejects():
    # min_blur_score defaults to None -- a very blurry-but-otherwise-fine face
    # must stay usable unless the caller explicitly opts into a blur floor.
    result = assess_quality(detector_score=0.9, face_w=200, face_h=200, blur_score=0.01)
    assert result.usable is True
