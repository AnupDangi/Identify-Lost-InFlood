import evaluate_retrieval as ev


def test_no_ground_truth_is_reported_clearly(tmp_path):
    rows = ev.load_ground_truth(tmp_path / "does_not_exist.csv")
    assert rows == []
    report = {"ground_truth_path": str(tmp_path / "does_not_exist.csv"), "ground_truth_available": False}
    md = ev.render_markdown(report)
    assert "Ground-truth evaluation has not been performed." in md


def test_core_evaluation_runs_over_synthetic_dataset(synthetic_dataset):
    gt_rows = [
        {"query_record_id": "AM1", "query_type": "am", "correct_candidate_record_id": "PM1",
         "candidate_type": "pm", "condition": "clean"},
        {"query_record_id": "PM1", "query_type": "pm", "correct_candidate_record_id": "AM1",
         "candidate_type": "am", "condition": "clean"},
        {"query_record_id": "AM3", "query_type": "am", "correct_candidate_record_id": "PM2",
         "candidate_type": "pm", "condition": "blur"},
    ]
    report = ev.run_core_evaluation(gt_rows, top_k_cap=20)

    assert report["overall"]["queries"] == 3
    for k in ev.RECALL_KS:
        assert f"recall_at_{k}" in report["overall"]

    per_query = {q["query_record_id"]: q for q in report["per_query"]}
    # AM1 has only one PM record with a usable face (PM1) in its gallery ->
    # PM1 must be found (some rank, not necessarily 1 given random vectors,
    # but present).
    assert per_query["AM1"]["source"] == "face+metadata"
    # AM3 has no usable face -> must fall back to metadata_only and still
    # find the correct candidate somewhere in the full 3-record PM gallery.
    assert per_query["AM3"]["source"] == "metadata_only"
    assert per_query["AM3"]["rank"] is not None

    assert "blur" in report["by_condition"]
    assert "clean" in report["by_condition"]


def test_ablation_runs_and_reports_three_conditions(synthetic_dataset):
    gt_rows = [
        {"query_record_id": "AM1", "query_type": "am", "correct_candidate_record_id": "PM1",
         "candidate_type": "pm", "condition": "clean"},
        {"query_record_id": "AM3", "query_type": "am", "correct_candidate_record_id": "PM2",
         "candidate_type": "pm", "condition": "blur"},
    ]
    result = ev.run_ablation(gt_rows, top_k_cap=20)
    assert set(result.keys()) == {"face_only", "metadata_only", "face_plus_metadata"}
    for summary in result.values():
        assert summary["queries"] == 2


def test_quality_threshold_sweep_reports_enrollment_rates(synthetic_dataset):
    gt_rows = [
        {"query_record_id": "AM1", "query_type": "am", "correct_candidate_record_id": "PM1",
         "candidate_type": "pm", "condition": "clean"},
    ]
    results = ev.run_quality_threshold_sweep(gt_rows, top_k_cap=20)
    assert len(results) == len(ev.SWEEP_MIN_DET_SCORE) * len(ev.SWEEP_MIN_BLUR_SCORE)
    for r in results:
        assert "am_enrollment_rate" in r and "pm_enrollment_rate" in r
