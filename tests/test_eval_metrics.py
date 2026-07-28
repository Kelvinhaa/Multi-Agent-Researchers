from eval.metrics import hit_rate_at_k, mrr, recall_at_k


def test_hit_rate_is_one_when_any_expected_source_retrieved():
    assert hit_rate_at_k(["travel_policy", "corporate_card"], ["travel_policy"]) == 1.0


def test_hit_rate_is_zero_on_disjoint_sets():
    assert hit_rate_at_k(["leave_policy"], ["travel_policy"]) == 0.0


def test_hit_rate_is_zero_when_nothing_retrieved():
    assert hit_rate_at_k([], ["travel_policy"]) == 0.0


def test_hit_rate_counts_duplicate_sources_once():
    assert hit_rate_at_k(["travel_policy", "travel_policy"], ["travel_policy"]) == 1.0


def test_recall_is_fractional_when_only_some_expected_sources_found():
    assert recall_at_k(["travel_policy"], ["travel_policy", "corporate_card"]) == 0.5


def test_recall_is_one_when_all_expected_sources_found():
    assert (
        recall_at_k(
            ["corporate_card", "travel_policy"], ["travel_policy", "corporate_card"]
        )
        == 1.0
    )


def test_recall_ignores_duplicate_retrievals():
    assert recall_at_k(["travel_policy", "travel_policy"], ["travel_policy"]) == 1.0


def test_recall_is_zero_when_expected_sources_empty():
    assert recall_at_k(["travel_policy"], []) == 0.0


def test_mrr_is_one_when_first_result_matches():
    assert mrr(["travel_policy", "leave_policy"], ["travel_policy"]) == 1.0


def test_mrr_uses_reciprocal_of_first_matching_rank():
    assert mrr(["a", "b", "travel_policy"], ["travel_policy"]) == 1 / 3


def test_mrr_is_zero_when_no_result_matches():
    assert mrr(["a", "b"], ["travel_policy"]) == 0.0
