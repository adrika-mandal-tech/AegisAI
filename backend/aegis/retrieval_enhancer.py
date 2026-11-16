# backend/aegis/retrieval_enhancer.py
"""Retrieval quality & hybrid ranking enhancers (Section 6).

Provides pre-retrieval and post-retrieval enhancements without any graph DB:
- Query rewriting / expansion (LLM + fast deterministic keyword expansion)
- HyDE (Hypothetical Document Embedding)
- Hybrid Reciprocal Rank Fusion (Dense vector similarity + BM25 lexical scoring)
- Content-hash & Jaccard near-duplicate deduplication
- Score-based filtering
- Contextual header injection (document filename, page, section heading)
- Cross-chunk context assembly & compression
- LLM & lexical relevance reranking
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

from aegis.core.logging import get_logger

log = get_logger(__name__)

_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "will", "with", "how", "why", "can", "could", "should", "would", "about",
    "into", "than", "then", "them", "they", "these", "those",
}
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.\-%$]+")


def _tokenize_lexical(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in _WORD_TOKEN_RE.findall(text or "")
        if len(tok) > 1 and tok.lower() not in _STOPWORDS
    ]


class QueryEnhancer:
    """Enhances queries before vector & hybrid retrieval."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def expand_keywords(self, query: str) -> str:
        """Fast deterministic keyword extraction/expansion without LLM latency."""
        tokens = _tokenize_lexical(query)
        if not tokens:
            return query
        # Keep unique salient terms in original order
        seen: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.append(t)
        return " ".join(seen)

    def rewrite(self, query: str, *, max_rewrites: int = 2) -> list[str]:
        """Return query variants for multi-query retrieval."""
        if not self._llm:
            kw = self.expand_keywords(query)
            return [query, kw] if kw and kw.lower() != query.lower() else [query]

        prompt = (
            "Rewrite the following user question to improve vector retrieval.\n"
            "Return up to 2 alternative phrasings, one per line.\n"
            "Do NOT add explanations.\n\n"
            f"Question: {query}\n\n"
            "Alternative phrasings:"
        )
        try:
            text = self._llm.generate(prompt, temperature=0.3, max_output_tokens=200).strip()
            lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
            variants = [query] + [v for v in lines if v and v.lower() != query.lower()][:max_rewrites]
            log.info("QueryEnhancer.rewrite", original=query[:80], variants=len(variants) - 1)
            return variants
        except Exception as exc:
            log.warning("Query rewrite failed", error=str(exc)[:200])
            return [query]

    def hyde(self, query: str, *, max_sentences: int = 3) -> str:
        """Generate a hypothetical answer passage for HyDE retrieval."""
        if not self._llm:
            return query

        prompt = (
            "Write a short, factual passage that would answer the question below.\n"
            f"Question: {query}\n\n"
            f"Write at most {max_sentences} sentences. Do NOT say 'I don't know'.\n"
            "Answer:"
        )
        try:
            text = self._llm.generate(prompt, temperature=0.2, max_output_tokens=300).strip()
            if not text:
                return query
            return text
        except Exception as exc:
            log.warning("HyDE generation failed", error=str(exc)[:200])
            return query

    def reformulate(self, query: str, *, reason: str = "") -> str:
        """Reformulate a query when initial retrieval yields weak scores."""
        if not self._llm:
            return self.expand_keywords(query)

        prompt = (
            "The following question did not return good search results.\n"
            f"Question: {query}\n"
        )
        if reason:
            prompt += f"Problem: {reason}\n"
        prompt += (
            "Rewrite the question to be more specific, using keywords likely to appear "
            "in technical or business documents. Return ONLY the rewritten question.\n\n"
            "Rewritten question:"
        )
        try:
            text = self._llm.generate(prompt, temperature=0.3, max_output_tokens=200).strip()
            return text if text else query
        except Exception as exc:
            log.warning("Query reformulation failed", error=str(exc)[:200])
            return query


def compute_bm25_scores(query: str, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Compute BM25 lexical relevance scores across candidate chunk texts."""
    if not texts:
        return []
    q_terms = _tokenize_lexical(query)
    if not q_terms:
        return [0.0 for _ in texts]

    doc_tokens = [_tokenize_lexical(t) for t in texts]
    doc_lens = [len(toks) for toks in doc_tokens]
    avg_dl = max(1.0, sum(doc_lens) / len(doc_lens))
    n_docs = len(texts)

    df: Counter[str] = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1

    scores: list[float] = []
    for toks, dl in zip(doc_tokens, doc_lens):
        tf = Counter(toks)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            n_q = df.get(term, 0)
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = f + k1 * (1.0 - b + b * (dl / avg_dl))
            s += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
        scores.append(s)
    return scores


def hybrid_rrf_rerank(
    query: str,
    chunks: list[Any],
    *,
    top_k: int = 10,
    rrf_k: int = 60,
    dense_weight: float = 0.65,
    lexical_weight: float = 0.35,
) -> list[Any]:
    """Fuse dense vector similarity rank and BM25 lexical rank via Reciprocal Rank Fusion."""
    if len(chunks) <= 1:
        return chunks[:top_k]

    # Dense ranking (descending by chunk.score)
    dense_order = sorted(
        range(len(chunks)),
        key=lambda idx: float(getattr(chunks[idx], "score", 0.0) or 0.0),
        reverse=True,
    )
    dense_rank = {idx: rank for rank, idx in enumerate(dense_order)}

    # BM25 lexical ranking
    bm25 = compute_bm25_scores(query, [getattr(c, "text", "") or "" for c in chunks])
    max_bm25 = max(bm25) if bm25 else 0.0
    if max_bm25 <= 0.0:
        return [chunks[i] for i in dense_order[:top_k]]

    lex_order = sorted(range(len(chunks)), key=lambda idx: bm25[idx], reverse=True)
    lex_rank = {idx: rank for rank, idx in enumerate(lex_order)}

    fused: list[tuple[float, int]] = []
    for idx in range(len(chunks)):
        rrf_score = (
            dense_weight / (rrf_k + dense_rank[idx] + 1)
            + lexical_weight / (rrf_k + lex_rank[idx] + 1)
        )
        # Blend normalized BM25 bonus gently into chunk.score for downstream visibility
        base_dense = float(getattr(chunks[idx], "score", 0.0) or 0.0)
        norm_lex = bm25[idx] / max_bm25
        blended = round(min(1.0, base_dense * 0.8 + norm_lex * 0.2), 4)
        try:
            chunks[idx].score = max(base_dense, blended)
        except Exception:
            pass
        fused.append((rrf_score, idx))

    fused.sort(key=lambda pair: pair[0], reverse=True)
    return [chunks[idx] for _, idx in fused[:top_k]]


def deduplicate_chunks(chunks: list[Any], *, max_results: int = 10) -> list[Any]:
    """Remove exact and near-duplicate chunks while preserving highest-scored occurrence."""
    seen_hashes: set[str] = set()
    seen_token_sets: list[set[str]] = []
    unique: list[Any] = []

    for chunk in chunks:
        txt = (getattr(chunk, "text", "") or "").strip()
        key = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]
        if key in seen_hashes:
            continue

        toks = set(_tokenize_lexical(txt))
        is_near_dup = False
        if len(toks) >= 25:
            for prev_toks in seen_token_sets:
                union_len = len(toks | prev_toks)
                if union_len > 0 and (len(toks & prev_toks) / union_len) >= 0.90:
                    is_near_dup = True
                    break
        if is_near_dup:
            continue

        seen_hashes.add(key)
        if toks:
            seen_token_sets.append(toks)
        unique.append(chunk)
        if len(unique) >= max_results:
            break

    log.info("deduplicate_chunks", input=len(chunks), output=len(unique))
    return unique


def filter_by_score(chunks: list[Any], *, min_score: float = 0.3) -> list[Any]:
    """Filter chunks below a similarity threshold."""
    return [c for c in chunks if (getattr(c, "score", 0.0) or 0.0) >= min_score]


def inject_contextual_headers(chunks: list[Any], *, max_header_chars: int = 220) -> list[Any]:
    """Prepend provenance header (filename, page, section_heading) into chunk text."""
    if not chunks:
        return chunks

    section_keywords = ("section_heading", "section_path", "title", "heading", "section")
    enriched = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        meta = getattr(chunk, "metadata", {}) or {}
        parts: list[str] = []
        src = getattr(chunk, "source", "")
        pg = getattr(chunk, "page", None)
        if src and src != "unknown":
            parts.append(f"Source: {src}")
        if pg is not None:
            parts.append(f"Page: {pg}")
        for key in section_keywords:
            value = meta.get(key)
            if value and str(value).strip():
                parts.append(f"Section: {str(value).strip()}")
                break
        if parts:
            header = " | ".join(parts)[:max_header_chars]
            if not text.startswith(f"[{header}]"):
                chunk.text = f"[{header}]\n{text}"
        enriched.append(chunk)
    return enriched


def assemble_compressed_context(chunks: list[Any], *, max_words: int = 3600) -> str:
    """Assemble retrieved chunks into a deduplicated, citation-tagged context string."""
    if not chunks:
        return ""

    blocks: list[str] = []
    used_words = 0
    for idx, chunk in enumerate(chunks, start=1):
        txt = (getattr(chunk, "text", "") or "").strip()
        if not txt:
            continue
        words = txt.split()
        if used_words + len(words) > max_words:
            remaining = max_words - used_words
            if remaining < 40:
                break
            txt = " ".join(words[:remaining]) + " ..."
            words = words[:remaining]

        src = getattr(chunk, "source", "document")
        pg = getattr(chunk, "page", None)
        meta = getattr(chunk, "metadata", {}) or {}
        sec = meta.get("section_heading") or ""
        score = float(getattr(chunk, "score", 0.0) or 0.0)

        tag_parts = [f"Doc #{idx}: {src}"]
        if pg is not None:
            tag_parts.append(f"p.{pg}")
        if sec:
            tag_parts.append(f"§{sec}")
        tag_parts.append(f"relevance={score:.2f}")
        blocks.append(f"[{' | '.join(tag_parts)}]\n{txt}")
        used_words += len(words)

    return "\n\n---\n\n".join(blocks)


def rerank_chunks(query: str, chunks: list[Any], *, llm: Any | None = None, top_k: int = 10) -> list[Any]:
    """Rerank retrieved chunks using hybrid BM25+dense fusion (fast) + optional LLM judge."""
    if not chunks:
        return chunks

    # Always apply fast deterministic hybrid RRF first
    candidates = hybrid_rrf_rerank(query, chunks, top_k=min(len(chunks), top_k * 2))
    if not llm or len(candidates) <= 1:
        return candidates[:top_k]

    scored: list[tuple[float, Any]] = []
    for chunk in candidates[:top_k]:
        prompt = (
            f"Question: {query}\n\n"
            f"Passage:\n{chunk.text[:1200]}\n\n"
            "Rate the relevance of the passage to the question on a scale from 0.0 to 1.0.\n"
            "Return ONLY a number between 0.0 and 1.0.\n"
            "Score:"
        )
        try:
            text = llm.generate(prompt, temperature=0.0, max_output_tokens=20).strip()
            score = max(0.0, min(1.0, float(text)))
        except Exception:
            score = float(getattr(chunk, "score", 0.0) or 0.0)
        scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:top_k]]
