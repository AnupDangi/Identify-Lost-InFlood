"""Model/version identifiers shared across scripts and the API.

Bump FACE_MODEL_VERSION whenever the face pipeline (detector, embedding model,
or preprocessing) changes -- scripts/build_embeddings.py's cache invalidates on
this string (Phase 3). Bump RANKING_MODEL_VERSION whenever dvi/scoring.py's
weights or bands change -- it's recorded on each review (Phase 18) so a past
review can be traced back to the heuristic that produced its shortlist.
"""
FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_VERSION = "insightface-buffalo_l-arcface"
RANKING_MODEL_VERSION = "dvi-scoring-heuristic-v1"
