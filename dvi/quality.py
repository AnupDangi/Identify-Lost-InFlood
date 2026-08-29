"""Face image quality assessment, per docs/P0_5_IMPLEMENTATION.md Phase 4.

Wraps the existing hard requirements from scripts/build_embeddings.py (no face /
face too small / low detector confidence -> unusable, unchanged) and adds an
informational quality_band + quality_reasons for records that DO have a usable
face, so a "usable but blurry/borderline" embedding is visible to scoring and to
reviewers instead of being silently treated the same as a crisp frontal shot.

IMPORTANT: the thresholds below (DEFAULT_MIN_FACE_SIZE, DEFAULT_MIN_DET_SCORE,
DEFAULT_MIN_BLUR_SCORE) are provisional engineering defaults, not scientifically
validated forensic cutoffs. blur is OFF by default (min_blur_score=None) so this
module does not silently start rejecting anything build_embeddings.py previously
accepted -- it only adds banding/reasons on top of the existing hard gate.
scripts/evaluate_retrieval.py's --quality-threshold-sweep is where these should
eventually be tuned against measured Recall@K, not asserted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

QUALITY_GOOD = "good"
QUALITY_MODERATE = "moderate"
QUALITY_POOR = "poor"
QUALITY_UNUSABLE = "unusable"

# Provisional engineering defaults -- see module docstring.
DEFAULT_MIN_FACE_SIZE = 40      # px, either dimension below this -> unusable
DEFAULT_MIN_DET_SCORE = 0.5     # InsightFace det_score below this -> unusable
DEFAULT_MIN_BLUR_SCORE: float | None = None  # Laplacian variance; None = not enforced


@dataclass
class QualityAssessment:
    usable: bool
    quality_band: str
    quality_reasons: list[str] = field(default_factory=list)


def assess_quality(
    *,
    detector_score: float | None,
    face_w: float | None,
    face_h: float | None,
    blur_score: float | None = None,
    landmarks_available: bool | None = None,
    min_face_size: float = DEFAULT_MIN_FACE_SIZE,
    min_det_score: float = DEFAULT_MIN_DET_SCORE,
    min_blur_score: float | None = DEFAULT_MIN_BLUR_SCORE,
) -> QualityAssessment:
    reasons: list[str] = []

    # Hard requirements -- unchanged from the pre-Phase-4 build_embeddings.py gate.
    if face_w is None or face_h is None or face_w < min_face_size or face_h < min_face_size:
        return QualityAssessment(False, QUALITY_UNUSABLE, ["face_too_small"])
    if detector_score is None or detector_score < min_det_score:
        return QualityAssessment(False, QUALITY_UNUSABLE, ["low_detector_confidence"])

    # Soft signals -- informational only, never flip usable=False.
    is_blurry = min_blur_score is not None and blur_score is not None and blur_score < min_blur_score
    if is_blurry:
        reasons.append("blurry")

    is_borderline_det = detector_score < (min_det_score + 0.15)
    if is_borderline_det:
        reasons.append("borderline_detector_confidence")

    is_small_margin = face_w < min_face_size * 1.5 or face_h < min_face_size * 1.5
    if is_small_margin:
        reasons.append("small_face_margin")

    if landmarks_available is False:
        reasons.append("no_landmarks")

    n_flags = sum([is_blurry, is_borderline_det, is_small_margin, landmarks_available is False])
    if n_flags == 0:
        band = QUALITY_GOOD
    elif n_flags == 1:
        band = QUALITY_MODERATE
    else:
        band = QUALITY_POOR

    return QualityAssessment(True, band, reasons)
