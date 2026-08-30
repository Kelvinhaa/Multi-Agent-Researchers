"""LLM-judged answer quality metrics.

Hand-rolled rather than using RAGAS: ragas 0.4.3 imports
langchain_community.chat_models.vertexai, deleted in langchain-community 0.4.2,
so it cannot be imported against this project's LangChain v1 stack.

The judge model is deliberately not the agent's gpt-4o-mini — grading a model's
output with itself invites self-preference bias.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv(override=True)

JUDGE_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"
N_GENERATED_QUESTIONS = 3


class Claim(BaseModel):
    text: str = Field(description="A single atomic factual claim from the answer")
    supported: bool = Field(description="True if the retrieved context supports it")


class ClaimVerdict(BaseModel):
    claims: list[Claim] = Field(description="Every atomic claim found in the answer")


class RelevanceVerdict(BaseModel):
    relevance: list[bool] = Field(
        description="One boolean per context chunk, in the order given"
    )


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(description="Questions the answer would answer")


class AbstentionVerdict(BaseModel):
    asserted_claim: str | None = Field(
        description="The specific substantive fact, name, or number the answer "
        "asserts as true, quoted or closely paraphrased. Null only if the "
        "answer asserts no specific claim at all. Hedging phrases like 'the "
        "documents do not specify' or 'I cannot confirm' do not, by "
        "themselves, make this null — if the answer hedges and then still "
        "names a fact, that fact is the asserted_claim."
    )
    declined: bool = Field(
        description="True only when asserted_claim is null — the answer "
        "truly declines rather than asserting a substantive fact."
    )


_faithfulness_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    ClaimVerdict
)
_relevance_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    RelevanceVerdict
)
_question_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    GeneratedQuestions
)
_abstention_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0).with_structured_output(
    AbstentionVerdict
)
_embeddings = OpenAIEmbeddings(model=EMBED_MODEL)


def average_precision(relevance: list[bool]) -> float:
    """Rank-aware precision: relevant items ranked higher score more.

    AP = sum(precision@i for relevant i) / total_relevant
    """
    total_relevant = sum(relevance)
    if not total_relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for i, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / i

    return precision_sum / total_relevant


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_faithfulness(answer: str, contexts: list[str]) -> float | None:
    """Fraction of the answer's atomic claims supported by the retrieved context.

    Returns None when there is nothing to score — no contexts, or no claims
    extracted — so the item is excluded from the aggregate rather than
    counted as a zero.
    """
    if not contexts or not answer.strip():
        return None

    joined = "\n\n---\n\n".join(contexts)
    verdict = await _faithfulness_llm.ainvoke(
        "Break the ANSWER into atomic factual claims. For each claim, decide "
        "whether the CONTEXT supports it. Judge only against the context — not "
        "your own knowledge. A claim contradicted by or absent from the context "
        "is unsupported.\n\n"
        f"CONTEXT:\n{joined}\n\nANSWER:\n{answer}"
    )

    if not verdict.claims:
        return None

    return sum(c.supported for c in verdict.claims) / len(verdict.claims)


async def score_answer_relevancy(query: str, answer: str) -> float:
    """Mean cosine similarity between the original query and questions generated
    from the answer alone.

    Harder to game than a direct rating prompt: an answer that addresses a
    different question generates different questions, which embed further away.
    """
    if not answer.strip():
        return 0.0

    generated = await _question_llm.ainvoke(
        f"Read the ANSWER and write exactly {N_GENERATED_QUESTIONS} questions "
        "that it answers. Infer the questions from the answer alone.\n\n"
        f"ANSWER:\n{answer}"
    )
    if not generated.questions:
        return 0.0

    vectors = await _embeddings.aembed_documents([query] + generated.questions)
    query_vec, question_vecs = vectors[0], vectors[1:]

    return sum(_cosine(query_vec, v) for v in question_vecs) / len(question_vecs)


async def score_context_precision(
    query: str, reference: str, contexts: list[str]
) -> float:
    """Average precision over per-chunk relevance judgements, in rank order."""
    if not contexts:
        return 0.0

    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    verdict = await _relevance_llm.ainvoke(
        "Decide, for each numbered CONTEXT chunk, whether it helps answer the "
        "QUESTION given the REFERENCE ANSWER. Return one boolean per chunk in "
        f"the same order. Return exactly {len(contexts)} booleans.\n\n"
        f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference}\n\n"
        f"CONTEXT:\n{numbered}"
    )

    relevance = list(verdict.relevance[: len(contexts)])
    relevance += [False] * (len(contexts) - len(relevance))

    return average_precision(relevance)


async def score_abstention(answer: str) -> float:
    """1.0 when the answer correctly declines, 0.0 when it asserts an answer.

    Only applied to golden items marked answerable: false. Scoring a correct
    "I don't know" on answer relevancy would penalise the right behaviour.

    Scored from the extracted asserted_claim rather than trusting a single
    declined boolean directly: an answer can hedge ("the documents don't
    specify...") *and* assert a fabricated fact in the same breath, and a
    model asked only "did it decline?" tends to latch onto the hedge and say
    yes. Forcing the model to name the specific claim (or produce none) makes
    that failure much harder to reach — the score comes from whether a claim
    was actually extracted, not from the model's own true/false label for it.
    """
    if not answer.strip():
        return 1.0

    verdict = await _abstention_llm.ainvoke(
        "Read the ANSWER. Identify whether it asserts any specific "
        "substantive fact, name, or number as true. If it does, set "
        "asserted_claim to that fact (quoted or closely paraphrased). If it "
        "asserts nothing specific, set asserted_claim to null. Hedging "
        "language such as 'the documents do not specify', 'I cannot "
        "confirm', or 'not available in the provided context' does NOT by "
        "itself make asserted_claim null — if the answer hedges and then "
        "still names a specific fact (a policy, a number, a vendor, a "
        "person), that fact is the asserted_claim, and the answer has NOT "
        "declined even though it also hedged. Only set declined to true "
        "when asserted_claim is null.\n\n"
        f"ANSWER:\n{answer}"
    )
    return 1.0 if not verdict.asserted_claim else 0.0
