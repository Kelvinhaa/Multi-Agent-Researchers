
def hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1.0 if any retrieved source is expected, else 0.0.

    Binary hit rate — deliberately not fractional. Use recall_at_k when the
    question genuinely requires more than one source.
    """
    return 1.0 if set(retrieved_sources) & set(expected_sources) else 0.0


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of expected sources that were retrieved.

    Empty expected_sources -> 0.0 (unanswerable items are scored on abstention
    instead, and never reach this function).
    """
    expected = set(expected_sources)
    if not expected:
        return 0.0
    return len(expected & set(retrieved_sources)) / len(expected)


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal of the 1-indexed rank of the first expected source. 0.0 if none."""
    expected = set(expected_sources)
    for i, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return 1.0 / i
    return 0.0
