import numpy as np

from dvi.retrieval import FAISS_FULL_GALLERY_THRESHOLD, resolve_faiss_k


def test_known_vector_returns_itself_at_rank_one():
    import faiss

    rng = np.random.default_rng(42)
    dim = 16
    n = 25
    matrix = rng.normal(size=(n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    query_idx = 7
    query = matrix[query_idx:query_idx + 1]
    sims, idxs = index.search(query, 5)

    assert idxs[0][0] == query_idx
    assert sims[0][0] > 0.999  # cosine similarity of a vector with itself


def test_resolve_faiss_k_defaults_to_full_gallery():
    assert resolve_faiss_k("all", 6000) == 6000
    assert resolve_faiss_k(None, 250) == 250


def test_resolve_faiss_k_caps_at_gallery_size():
    assert resolve_faiss_k(1000, 50) == 50


def test_resolve_faiss_k_explicit_int_used_when_under_gallery_size():
    assert resolve_faiss_k(10, 500) == 10


def test_resolve_faiss_k_full_gallery_capped_at_threshold_for_huge_galleries():
    huge = FAISS_FULL_GALLERY_THRESHOLD + 50_000
    assert resolve_faiss_k("all", huge) == FAISS_FULL_GALLERY_THRESHOLD
