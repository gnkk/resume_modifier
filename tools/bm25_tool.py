"""
BM25 lexical ranking.

The coarse pre-filter between scraping and screening. It answers one
narrow question cheaply across the whole scrape: is this posting even in
the candidate's field? Everything subtler is the screener's job.

Why BM25 and not embeddings: the question here is lexical, and BM25
answers it with arithmetic over token frequencies — no model, no
download, no torch, deterministic, and fast enough over a few hundred
postings to be invisible. Embedding similarity would answer the same
question slightly better while adding gigabytes of dependency, and it
would be equally blind to the things that actually disqualify a posting
(seniority band, work authorization, contract-vs-permanent), because
those are short common phrases that carry almost no weight in ANY
bag-of-words or whole-document similarity measure.

That blindness is the reason this stage only ever pre-filters. A staff-
level posting reads as a near-perfect lexical match to a mid-level
candidate — same stack, same methods, more of them — and BM25 will
happily rank it first. The screener is what reads "Staff" and "10+
years" and rates it a 2.

BM25 over TF-IDF specifically: it saturates term frequency (the twelfth
mention of "Python" adds almost nothing, so a posting can't win on
repetition) and normalizes for document length (so longer senior-level
postings don't accumulate matches by sheer volume). Both failure modes
are live in this corpus.

Scores are NOT comparable across runs or corpora — BM25's IDF term is
computed from the pool it is given. Always rank within a run and keep a
proportion; never threshold on an absolute score.
"""

import re

from logger_setup import get_logger

log = get_logger(__name__)

# Keep +, # and . inside tokens so "c++", "c#", "node.js" and "3.11"
# survive tokenization. Leading character must be alphanumeric.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Deliberately small. BM25's IDF term already discounts words appearing in
# most documents, so an aggressive stopword list is redundant and risks
# removing something load-bearing. This covers only the highest-frequency
# English function words plus the job-posting boilerplate that appears in
# essentially every record.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for
with from by as is are was were be been being will would can could should
we you they our your their it its has have had do does did not no so such
job role position work working experience years team teams company
opportunity candidate candidates applicants apply please
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word characters, drop stopwords and 1-char tokens."""
    if not text:
        return []
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _document_text(record: dict) -> str:
    """
    The text BM25 scores for one posting.

    Title is included because it carries the highest-signal terms in the
    whole record. The full description is used when present, falling back
    to the snippet — a posting whose description failed to fetch still
    gets ranked on what there is rather than being silently sunk to the
    bottom.
    """
    return " ".join(
        str(part)
        for part in (
            record.get("title"),
            record.get("company"),
            record.get("location"),
            record.get("description_full") or record.get("description_snippet"),
        )
        if part
    )


def rank_pool(pool: list[dict], query_text: str, keep: int) -> list[dict]:
    """
    Rank postings by lexical similarity to the candidate, keep the top
    `keep`, and annotate each survivor with `bm25_rank` and `bm25_score`.

    `bm25_rank` is 1-based and survives into the final pool, where it
    serves as the tie-break for postings the screener rated equally —
    which matters, because an integer rating produces heavy ties.

    Degrades to an unranked truncation on any failure: losing the ranking
    costs pool quality, while failing the run costs the run. A pool
    smaller than `keep` is returned ranked but uncut.
    """
    if not pool:
        return pool

    if not query_text or not query_text.strip():
        log.error("bm25_tool: empty query text; skipping the ranking stage.")
        return pool[:keep]

    try:
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(_document_text(record)) for record in pool]
        if not any(corpus):
            log.error("bm25_tool: every posting tokenized to nothing; skipping the ranking stage.")
            return pool[:keep]

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(query_text))
    except Exception as exc:  # noqa: BLE001 — ranking is an optimization, never load-bearing
        log.error("bm25_tool: ranking failed (%s); falling back to scrape order.", exc, exc_info=True)
        log.info("  Warning: BM25 ranking failed — using scrape order instead.")
        return pool[:keep]

    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    for rank, index in enumerate(order, start=1):
        pool[index]["bm25_rank"] = rank
        pool[index]["bm25_score"] = round(float(scores[index]), 3)

    ranked = [pool[i] for i in order[:keep]]

    if len(pool) > keep:
        log.info(
            "  BM25 ranked %d posting(s) against the resume, kept top %d "
            "(score %.2f down to %.2f).",
            len(pool), len(ranked),
            scores[order[0]], scores[order[len(ranked) - 1]],
        )
    else:
        log.info("  BM25 ranked all %d posting(s) (none cut).", len(pool))

    return ranked
