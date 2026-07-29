import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from eval.judges import average_precision, score_context_precision, score_faithfulness
from eval.metrics import hit_rate_at_k, mrr, recall_at_k
from eval.run import aggregate, token_cost


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


def test_average_precision_is_one_when_all_contexts_relevant():
    assert average_precision([True, True, True]) == 1.0


def test_average_precision_is_zero_when_none_relevant():
    assert average_precision([False, False]) == 0.0


def test_average_precision_rewards_relevant_contexts_ranked_higher():
    early = average_precision([True, False, False, False])
    late = average_precision([False, False, False, True])

    assert early > late


def test_average_precision_matches_hand_computed_value():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert average_precision([True, False, True]) == (1.0 + 2 / 3) / 2


def test_average_precision_handles_empty_input():
    assert average_precision([]) == 0.0


def test_faithfulness_is_fraction_of_supported_claims():
    verdict = MagicMock(
        claims=[
            MagicMock(supported=True),
            MagicMock(supported=True),
            MagicMock(supported=False),
            MagicMock(supported=False),
        ]
    )
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=verdict)

    with patch("eval.judges._faithfulness_llm", fake_llm):
        score = asyncio.run(score_faithfulness("answer", ["context"]))

    assert score == 0.5


def test_faithfulness_returns_none_when_no_claims_extracted():
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(claims=[]))

    with patch("eval.judges._faithfulness_llm", fake_llm):
        assert asyncio.run(score_faithfulness("answer", ["context"])) is None


def test_faithfulness_returns_none_without_contexts():
    assert asyncio.run(score_faithfulness("answer", [])) is None


def test_context_precision_uses_judged_relevance_in_rank_order():
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=MagicMock(relevance=[True, False, True]))

    with patch("eval.judges._relevance_llm", fake_llm):
        score = asyncio.run(score_context_precision("q", "ref", ["c1", "c2", "c3"]))

    assert score == (1.0 + 2 / 3) / 2


def test_context_precision_is_zero_without_contexts():
    assert asyncio.run(score_context_precision("q", "ref", [])) == 0.0


def test_token_cost_prices_mini_by_longest_prefix_not_gpt_4o():
    # "gpt-4o-mini-2024-07-18" startswith BOTH "gpt-4o" and "gpt-4o-mini".
    # Matching the shorter prefix would bill mini traffic at the gpt-4o rate.
    usage = {"gpt-4o-mini-2024-07-18": {"input_tokens": 1_000_000, "output_tokens": 0}}

    assert token_cost(usage) == 0.15


def test_token_cost_sums_input_and_output_at_their_own_rates():
    usage = {
        "gpt-4o-2024-11-20": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    }

    assert token_cost(usage) == 12.50


def test_token_cost_sums_across_multiple_models():
    usage = {
        "gpt-4o-mini-2024-07-18": {"input_tokens": 1_000_000, "output_tokens": 0},
        "gpt-4o-2024-11-20": {"input_tokens": 1_000_000, "output_tokens": 0},
    }

    assert token_cost(usage) == 0.15 + 2.50


def test_token_cost_ignores_models_with_no_known_price():
    assert token_cost({"mystery-model-v9": {"input_tokens": 1_000_000}}) == 0.0


def test_token_cost_is_zero_for_empty_usage():
    assert token_cost({}) == 0.0


def test_aggregate_reports_operational_means_and_total_cost():
    records = [
        {
            "cluster": "finance",
            "steps": 2,
            "latency_s": 4.0,
            "cost_usd": 0.01,
            "tokens": 100,
            "error": None,
        },
        {
            "cluster": "finance",
            "steps": 4,
            "latency_s": 6.0,
            "cost_usd": 0.03,
            "tokens": 300,
            "error": None,
        },
    ]

    result = aggregate(records)

    assert result["operational"]["steps"] == 3.0
    assert result["operational"]["latency_s"] == 5.0
    assert round(result["total_cost_usd"], 6) == 0.04


def test_aggregate_averages_each_metric_over_scored_items():
    records = [
        {
            "cluster": "finance",
            "hit_rate": 1.0,
            "recall": 1.0,
            "mrr": 1.0,
            "faithfulness": 0.8,
            "error": None,
        },
        {
            "cluster": "finance",
            "hit_rate": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "faithfulness": 0.4,
            "error": None,
        },
    ]

    result = aggregate(records)

    assert result["overall"]["hit_rate"] == 0.5
    assert round(result["overall"]["faithfulness"], 6) == 0.6


def test_aggregate_excludes_none_scores_rather_than_counting_them_as_zero():
    records = [
        {"cluster": "people", "faithfulness": 1.0, "error": None},
        {"cluster": "people", "faithfulness": None, "error": None},
    ]

    assert aggregate(records)["overall"]["faithfulness"] == 1.0


def test_aggregate_excludes_errored_items():
    records = [
        {"cluster": "people", "hit_rate": 1.0, "error": None},
        {"cluster": "people", "hit_rate": 0.0, "error": "boom"},
    ]

    result = aggregate(records)

    assert result["overall"]["hit_rate"] == 1.0
    assert result["failed"] == 1


def test_aggregate_breaks_metrics_down_by_cluster():
    records = [
        {"cluster": "finance", "hit_rate": 1.0, "error": None},
        {"cluster": "engineering", "hit_rate": 0.0, "error": None},
    ]

    result = aggregate(records)

    assert result["by_cluster"]["finance"]["hit_rate"] == 1.0
    assert result["by_cluster"]["engineering"]["hit_rate"] == 0.0


def test_aggregate_handles_no_scorable_records():
    assert (
        aggregate([{"cluster": "x", "hit_rate": 1.0, "error": "boom"}])["overall"] == {}
    )
