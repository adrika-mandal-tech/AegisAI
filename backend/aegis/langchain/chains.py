"""LangChain RAG chains for AegisAI."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from aegis.langchain.embeddings import AegisEmbeddings
from aegis.langchain.retriever import AegisRetriever


# Default RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain enough information to answer, say so. Cite relevant sources using [chunk_id] format."""

QA_PROMPT_TEMPLATE = """You are a precise question-answering system. Use the following context to answer the question.

Context:
{context}

Question: {question}

Provide a clear, concise answer with citations to the relevant context chunks using [chunk_id] format. If the answer cannot be determined from the context, state that clearly."""

SUMMARIZATION_PROMPT_TEMPLATE = """Summarize the following document content. Focus on key points, findings, and conclusions.

Content:
{context}

Summary:"""


def format_docs(docs: list[Any]) -> str:
    """Format retrieved documents for prompt context."""
    formatted = []
    for i, doc in enumerate(docs):
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Chunk: {chunk_id} | Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(
    llm: BaseLanguageModel,
    retriever: AegisRetriever | None = None,
    prompt_template: str | None = None,
    top_k: int = 5,
) -> Any:
    """Create a LangChain RAG chain.

    Args:
        llm: LangChain-compatible language model
        retriever: AegisRetriever instance (creates default if None)
        prompt_template: Custom prompt template (uses default if None)
        top_k: Number of documents to retrieve

    Returns:
        Runnable chain: question -> answer with sources
    """
    if retriever is None:
        retriever = AegisRetriever(top_k=top_k)

    prompt = ChatPromptTemplate.from_template(prompt_template or RAG_PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def create_qa_chain(
    llm: BaseLanguageModel,
    retriever: AegisRetriever | None = None,
    prompt_template: str | None = None,
    top_k: int = 5,
    return_sources: bool = True,
) -> Any:
    """Create a LangChain QA chain that returns both answer and sources.

    Args:
        llm: LangChain-compatible language model
        retriever: AegisRetriever instance
        prompt_template: Custom prompt template
        top_k: Number of documents to retrieve
        return_sources: Whether to return source documents

    Returns:
        Runnable chain that returns dict with 'answer' and optionally 'sources'
    """
    if retriever is None:
        retriever = AegisRetriever(top_k=top_k)

    prompt = ChatPromptTemplate.from_template(prompt_template or QA_PROMPT_TEMPLATE)

    def retrieve_and_format(query: str) -> dict[str, Any]:
        docs = retriever.invoke(query)
        context = format_docs(docs)
        return {
            "context": context,
            "question": query,
            "source_documents": docs if return_sources else None,
        }

    chain = (
        RunnablePassthrough.assign(**{"retrieved": lambda x: retrieve_and_format(x["question"])})
        | (lambda x: {"context": x["retrieved"]["context"], "question": x["question"], "source_documents": x["retrieved"]["source_documents"]})
        | {
            "answer": prompt | llm | StrOutputParser(),
            "sources": lambda x: x["source_documents"],
        }
    )

    return chain


def create_summarization_chain(
    llm: BaseLanguageModel,
    prompt_template: str | None = None,
    chunk_size: int = 10000,
) -> Any:
    """Create a document summarization chain.

    Args:
        llm: LangChain-compatible language model
        prompt_template: Custom prompt template
        chunk_size: Maximum characters per chunk for long documents

    Returns:
        Runnable chain: document_text -> summary
    """
    prompt = ChatPromptTemplate.from_template(prompt_template or SUMMARIZATION_PROMPT_TEMPLATE)

    def chunk_and_summarize(text: str) -> str:
        if len(text) <= chunk_size:
            return (prompt | llm | StrOutputParser()).invoke({"context": text})

        # Chunk long documents and summarize each chunk
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in chunks:
            summary = (prompt | llm | StrOutputParser()).invoke({"context": chunk})
            summaries.append(summary)

        # Combine summaries
        combined = "\n\n".join(summaries)
        if len(combined) > chunk_size:
            # Recursive summarization
            return chunk_and_summarize(combined)
        return combined

    return chunk_and_summarize


def create_multi_query_rag_chain(
    llm: BaseLanguageModel,
    retriever: AegisRetriever | None = None,
    prompt_template: str | None = None,
    num_queries: int = 3,
) -> Any:
    """Create a multi-query RAG chain that generates multiple queries for better retrieval.

    Args:
        llm: LangChain-compatible language model
        retriever: AegisRetriever instance (should have enable_multi_query=True)
        prompt_template: Custom prompt template
        num_queries: Number of alternative queries to generate

    Returns:
        Runnable chain with multi-query retrieval
    """
    if retriever is None:
        retriever = AegisRetriever(top_k=5, enable_multi_query=True)

    # Query generation prompt
    query_gen_prompt = ChatPromptTemplate.from_template(
        "Generate {num_queries} alternative search queries for: {question}\n"
        "Return one query per line, no explanations."
    )

    def multi_query_retrieve(question: str) -> list[str]:
        # Generate alternative queries
        queries_text = (query_gen_prompt | llm | StrOutputParser()).invoke(
            {"question": question, "num_queries": num_queries}
        )
        queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
        queries = [question] + queries[:num_queries]

        # Retrieve for each query
        all_docs = []
        for q in queries:
            docs = retriever.invoke(q)
            all_docs.extend(docs)

        # Deduplicate by chunk_id
        seen = set()
        unique_docs = []
        for doc in all_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)

        return format_docs(unique_docs[: retriever.top_k])

    prompt = ChatPromptTemplate.from_template(prompt_template or RAG_PROMPT_TEMPLATE)

    chain = (
        {
            "context": RunnablePassthrough.assign(context=multi_query_retrieve) | (lambda x: x["context"]),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain