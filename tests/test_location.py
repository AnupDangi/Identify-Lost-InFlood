from dvi.location import canonical_province, location_score, normalize_text


def test_normalize_text_cleans_punctuation_and_whitespace():
    assert normalize_text("Kathmandu,  Nepal.") == "kathmandu nepal"
    assert normalize_text("  ") == ""
    assert normalize_text(None) == ""


def test_unknown_location_is_neutral():
    assert location_score({}, {}) == 0.5
    assert location_score({"location": "somewhere"}, {}) == 0.5


def test_same_municipality_scores_strongest():
    am = {"location": "Bagmati Kathmandu Kathmandu Metro", "municipality": "Kathmandu Metro"}
    pm = {"location": "found near Kathmandu Metro city hall"}
    same_muni = location_score(am, pm)

    am_district_only = {"location": "Bagmati Kathmandu", "district": "Kathmandu"}
    pm_district_only = {"location": "somewhere in Kathmandu district"}
    same_district = location_score(am_district_only, pm_district_only)

    assert same_muni > same_district


def test_district_beats_province_beats_unrelated():
    am_district = {"district": "Kathmandu", "location": "Kathmandu"}
    pm_same_district = {"location": "Kathmandu area"}
    district_score = location_score(am_district, pm_same_district)

    am_province = {"province": "Bagmati", "location": "Bagmati"}
    pm_same_province = {"location": "Bagmati region"}
    province_score = location_score(am_province, pm_same_province)

    assert district_score > province_score


def test_token_overlap_fallback_when_no_structured_fields():
    am = {"location": "Rasuwa flood affected area riverside"}
    pm = {"location": "Rasuwa riverside disaster site"}
    strong_overlap = location_score(am, pm)

    am2 = {"location": "Rasuwa flood affected area"}
    pm2 = {"location": "Kailali unrelated place entirely"}
    no_overlap = location_score(am2, pm2)

    assert strong_overlap > no_overlap


def test_canonical_province_best_effort_alias_match():
    assert canonical_province("कोशी प्रदेश") == "koshi"
    assert canonical_province("बागमती प्रदेश") == "bagmati"
    assert canonical_province("not a real province") is None
