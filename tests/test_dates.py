from dvi.dates import get_normalized, normalize_date


def test_known_bs_to_ad_conversion():
    # Same example used in docs/P0_5_IMPLEMENTATION.md and the original spec.
    result = normalize_date("2083-05-12")
    assert result["calendar_type"] == "BS"
    assert result["event_date_normalized"] == "2026-08-28"


def test_known_ad_passthrough():
    result = normalize_date("2026-08-28")
    assert result["calendar_type"] == "AD"
    assert result["event_date_normalized"] == "2026-08-28"


def test_devanagari_digits_handled():
    result = normalize_date("२०८३-०५-१२")
    assert result["calendar_type"] == "BS"
    assert result["event_date_normalized"] == "2026-08-28"


def test_explicit_calendar_marker_overrides_heuristic():
    # A year that the plain year-range heuristic would read as BS, but that's
    # explicitly marked A.D., should be trusted over the heuristic.
    result = normalize_date("2083-05-12 A.D")
    assert result["calendar_type"] == "AD"
    assert result["event_date_normalized"] == "2083-05-12"


def test_empty_and_garbage_return_unknown_not_a_crash():
    for raw in ("", None, "not a date", "unknown"):
        result = normalize_date(raw)
        assert result["calendar_type"] == "unknown"
        assert result["event_date_normalized"] == ""


def test_get_normalized_prefers_precomputed_columns():
    record = {"event_date": "garbage", "event_date_normalized": "2026-01-01", "calendar_type": "AD"}
    result = get_normalized(record)
    assert result["event_date_normalized"] == "2026-01-01"
    assert result["calendar_type"] == "AD"


def test_get_normalized_falls_back_to_raw_event_date():
    record = {"event_date": "2083-05-12"}
    result = get_normalized(record)
    assert result["event_date_normalized"] == "2026-08-28"
