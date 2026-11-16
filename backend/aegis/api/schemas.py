# backend/aegis/api/schemas.py
"""FastAPI request/response schemas for the AegisAI AgentSwarm REST & SSE endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from aegis.schemas.agent import (
    AgentWorkIllustration,
    ArtifactMetadata,
    ChartData,
    ResponseEvaluationMetrics,
    TableData,
)


class AgentRequest(BaseModel):
    """Request body for single-agent endpoints."""

    input: str = Field(..., min_length=1, description="The input text for the agent.")


class ToolRequest(BaseModel):
    """Request body for the Tool Agent endpoint."""

    tool: str = Field(..., min_length=1, description="Tool name to execute.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Tool-specific arguments.")


class AgentResponse(BaseModel):
    """Standard response wrapper for single-agent endpoints."""

    success: bool = True
    data: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response for the ``/api/health`` endpoint."""

    status: str = "ok"
    version: str = ""
    environment: str = ""
    upload_limits: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_files_per_batch": 100,
            "max_file_size_mb": 500,
            "max_total_size_mb": 1024,
            "max_concurrent_uploads": 10,
            "supported_formats": [
                ".pdf",
                ".docx",
                ".doc",
                ".pptx",
                ".xls",
                ".xlsx",
                ".csv",
                ".json",
                ".html",
                ".htm",
                ".txt",
                ".md",
                ".zip",
                ".png",
                ".jpg",
                ".jpeg",
                ".tiff",
                ".bmp",
                ".webp",
            ],
            "supports_urls": True,
            "supports_repos": True,
        }
    )
    pipeline_agents: list[str] = Field(
        default_factory=lambda: [
            "supervisor",
            "planner",
            "retriever",
            "researcher",
            "validator",
            "reasoner",
            "critic",
            "report_generator",
        ]
    )
    models: dict[str, Any] = Field(
        default_factory=lambda: {
            "primary_llm": "gemini-2.5-flash",
            "embedding_model": "BAAI/bge-m3",
            "secondary_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    )
    openlm: dict[str, Any] = Field(default_factory=dict)
    qdrant: dict[str, Any] = Field(default_factory=dict)
    chunking_defaults: dict[str, Any] = Field(
        default_factory=lambda: {
            "chunk_size_words": 900,
            "chunk_overlap_pct": 0.10,
            "chunk_overlap_words": 90,
        }
    )


# ── Embedding Test ──────────────────────────────────────────────────────────


class EmbeddingTestRequest(BaseModel):
    """Request for embedding test endpoint."""

    text: str = Field(default="test embedding")
    provider: str | None = Field(default=None)
    model_name: str | None = Field(default=None)


class EmbeddingTestResponse(BaseModel):
    """Response for embedding test endpoint."""

    success: bool = True
    provider: str
    model: str
    dimensions: int
    vector_preview: list[float]
    error: str | None = None


# ── Pipeline ────────────────────────────────────────────────────────────────


class PipelineRequest(BaseModel):
    """Request body for the full 8-agent LangGraph pipeline endpoint."""

    task: str = Field(..., min_length=1, description="The task to run through the full agent pipeline.")
    doc_id: str | None = Field(default=None, description="Optional single document filter.")
    document_ids: list[str] = Field(default_factory=list, description="Optional multi-document filter.")
    session_id: str | None = Field(default=None, description="Optional session ID for multi-turn memory.")
    requested_formats: list[str] = Field(
        default_factory=list,
        description="Requested deliverable formats (pdf, docx, pptx, xlsx, csv, html, md).",
    )


class PipelineStepResult(BaseModel):
    """A single step's result in the pipeline trace."""

    agent: str = ""
    status: str = ""


class PipelineResponse(BaseModel):
    """Response from the full 8-agent LangGraph pipeline."""

    success: bool = True
    execution_id: str = ""
    session_id: str = ""
    task_type: str = "general"
    task_complexity: str = "medium"
    report_title: str = ""
    report_markdown: str = ""
    action_items: list[str] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)
    chart_data: list[dict[str, Any]] = Field(default_factory=list)
    table_data: list[dict[str, Any]] = Field(default_factory=list)
    execution_plan: dict[str, Any] | list[dict[str, Any]] = Field(default_factory=list)
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_trace: list[PipelineStepResult] = Field(default_factory=list)
    agent_work_illustrations: list[dict[str, Any]] = Field(default_factory=list)
    response_metrics: dict[str, Any] = Field(default_factory=dict)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    revision_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


# ── Background Long-Running Task Orchestration ──────────────────────────────


class TaskSubmitRequest(BaseModel):
    """Request to launch an asynchronous background execution task."""

    task: str = Field(..., min_length=1)
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    requested_formats: list[str] = Field(default_factory=list)


class TaskStatusResponse(BaseModel):
    """Full status payload for a background execution task."""

    success: bool = True
    execution_id: str = ""
    session_id: str = ""
    user_query: str = ""
    state: str = "PENDING"
    progress_pct: int = 0
    current_agent: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    requested_formats: list[str] = Field(default_factory=list)
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    agent_work_illustrations: list[dict[str, Any]] = Field(default_factory=list)
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    response_metrics: dict[str, Any] = Field(default_factory=dict)
    report_title: str = ""
    report_markdown: str = ""
    chart_data: list[dict[str, Any]] = Field(default_factory=list)
    table_data: list[dict[str, Any]] = Field(default_factory=list)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    events_count: int = 0
    errors: list[str] = Field(default_factory=list)
    error: str | None = None


class TaskListResponse(BaseModel):
    """List of recent background tasks."""

    success: bool = True
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0


# ── Streaming / SSE ─────────────────────────────────────────────────────────


class SSEEvent(BaseModel):
    """A single Server-Sent Event payload sent to the frontend."""

    agent: str = ""
    type: str = ""
    output: str = ""
    status: str = ""


# ── PDF & Multi-Format Deliverables ─────────────────────────────────────────


class PDFReportRequest(BaseModel):
    """Request body for PDF report generation."""

    markdown: str = Field(..., description="Markdown report content")
    title: str = Field(default="AegisAI Report", description="Report title")


class ArtifactGenerateRequest(BaseModel):
    """Request body for generating multi-format deliverables on demand."""

    execution_id: str = Field(default="", description="Optional execution ID to attach artifacts to.")
    title: str = Field(default="AegisAI Intelligence Report")
    markdown: str = Field(..., min_length=1)
    action_items: list[str] = Field(default_factory=list)
    formats: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "pptx", "xlsx", "csv", "html", "md"]
    )


class ArtifactListResponse(BaseModel):
    """List of downloadable deliverable artifacts."""

    success: bool = True
    execution_id: str = ""
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


# ── Dataset Analytics ───────────────────────────────────────────────────────


class DatasetProfileResponse(BaseModel):
    """Statistical profile and generated visualizations for a structured dataset."""

    success: bool = True
    document_id: str = ""
    filename: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


# ── Documents ───────────────────────────────────────────────────────────────


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document."""

    success: bool = True
    filename: str = ""
    chunks: int = 0
    error: str | None = None
    source_type: str = "file"
    document_id: str = ""
    status: str = "ready"
    parser_used: str = ""


class DocumentListItem(BaseModel):
    """A single document in the list."""

    id: str = ""
    filename: str = ""
    uploaded_at: str = ""
    status: str = "unknown"
    chunks: int = 0
    error: str = ""
    source_type: str = "file"
    file_size_bytes: int = 0
    parser_used: str = ""
    is_tabular: bool = False


class DocumentListResponse(BaseModel):
    """Response listing all uploaded documents."""

    success: bool = True
    documents: list[DocumentListItem] = Field(default_factory=list)
    error: str | None = None


# ── Batch Upload ────────────────────────────────────────────────────────────


class BatchUploadItem(BaseModel):
    """Result for a single file in a batch upload."""

    filename: str = ""
    success: bool = False
    chunks: int = 0
    error: str = ""
    document_id: str = ""
    status: str = "ready"


class BatchUploadResponse(BaseModel):
    """Response for a batch document upload."""

    success: bool = True
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    results: list[BatchUploadItem] = Field(default_factory=list)
    error: str | None = None


# ── Chunk Metadata ──────────────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Structured metadata for a document chunk stored in Qdrant."""

    filename: str = ""
    type: str = "document_chunk"
    source_type: str = "file"
    chunk_index: int = 0
    doc_id: str = ""
    content_hash: str = ""
    token_count: int = 0
    word_count: int = 0
    char_offset: int = 0
    page_number: int | None = None
    section_path: str = ""
    parser_used: str = ""
    chunk_strategy: str = ""
    text: str = ""


class ChunkMetadataResponse(BaseModel):
    """Response for chunk metadata queries."""

    success: bool = True
    chunks: list[ChunkMetadata] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


class IngestStatusResponse(BaseModel):
    """Response for ingestion status queries."""

    success: bool = True
    doc_id: str = ""
    filename: str = ""
    status: str = ""
    stage: str = ""
    progress: int = 0
    chunks_total: int = 0
    chunks_stored: int = 0
    chunks_deduped: int = 0
    embed_latency_ms: int = 0
    storage_latency_ms: int = 0
    errors: list[str] = Field(default_factory=list)
    error: str | None = None


# ── Chunk Retrieval ─────────────────────────────────────────────────────────


class ChunkRetrievalRequest(BaseModel):
    """Request to retrieve chunks for a document with pagination."""

    doc_id: str = Field(..., min_length=1)
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=200)


class ChunkRetrievalResponse(BaseModel):
    """Response with paginated chunks for a document."""

    success: bool = True
    doc_id: str = ""
    chunks: list[ChunkMetadata] = Field(default_factory=list)
    total_chunks: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0
    error: str | None = None


# ── Retrieval Test ──────────────────────────────────────────────────────────


class RetrievalTestRequest(BaseModel):
    """Request to test retrieval against a specific document."""

    doc_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)


class RetrievalTestResult(BaseModel):
    """Single retrieval test result."""

    chunk_index: int = 0
    score: float = 0.0
    text_preview: str = ""
    content_hash: str = ""
    token_count: str | int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestResponse(BaseModel):
    """Response for retrieval test."""

    success: bool = True
    doc_id: str = ""
    query: str = ""
    results: list[RetrievalTestResult] = Field(default_factory=list)
    total_matches: int = 0
    search_time_ms: int = 0
    error: str | None = None


# ── Qdrant Health ───────────────────────────────────────────────────────────


class QdrantHealthResponse(BaseModel):
    """Response for Qdrant health/status check."""

    success: bool = True
    collection_name: str = ""
    vector_size: int = 0
    vectors_count: int = 0
    indexed_vectors_count: int = 0
    points_count: int = 0
    segments_count: int = 0
    disk_usage_bytes: int = 0
    status: str = ""
    error: str | None = None


# ── URL / Repo ingestion ────────────────────────────────────────────────────


class SourceIngestRequest(BaseModel):
    """Request to ingest content from a URL or git repo."""

    source: str = Field(..., min_length=1, description="URL or git repo URL")
    source_type: str = Field("url", pattern="^(url|repo)$")


class SourceIngestResponse(BaseModel):
    """Response after ingesting a URL or repo."""

    success: bool = True
    source: str = ""
    source_type: str = ""
    chunks: int = 0
    document_id: str = ""
    error: str | None = None


# ── Pipeline History ────────────────────────────────────────────────────────


class HistoryItem(BaseModel):
    """A single pipeline run in history."""

    execution_id: str = ""
    task: str = ""
    report_title: str = ""
    report_markdown: str = ""
    timestamp: str = ""
    agent_count: int = 0
    response_metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Response listing past pipeline runs."""

    success: bool = True
    runs: list[HistoryItem] = Field(default_factory=list)
    error: str | None = None


# ── Chat ────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = ""
    timestamp: str = ""


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(..., min_length=1, description="The user message.")
    history: list[ChatMessage] = Field(default_factory=list)
    mode: str = Field("auto", pattern="^(auto|plan|research|analysis)$")
    context_documents: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    session_id: str | None = Field(default=None)
    requested_formats: list[str] = Field(default_factory=list)
    tool_settings: dict[str, Any] = Field(default_factory=dict)


class EvidenceSchema(BaseModel):
    """Evidence item returned in chat responses."""

    id: str = ""
    claim: str = ""
    source: str = ""
    supporting_passage: str = ""
    confidence: float = 0.0
    relevance: float = 0.0


class CitationSchema(BaseModel):
    """Citation reference returned in chat responses."""

    id: str = ""
    claim: str = ""
    source: str = ""
    passage_excerpt: str = ""


class ChatResponse(BaseModel):
    """Enhanced response from the chat endpoint with metrics, illustrations & deliverables."""

    success: bool = True
    execution_id: str = ""
    session_id: str = ""
    response: str = ""
    plan: dict[str, Any] | None = None
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: list[dict[str, str]] = Field(default_factory=list)
    agent_work_illustrations: list[AgentWorkIllustration | dict[str, Any]] = Field(
        default_factory=list
    )
    response_metrics: ResponseEvaluationMetrics | dict[str, Any] = Field(
        default_factory=dict
    )
    artifacts: list[ArtifactMetadata | dict[str, Any]] = Field(default_factory=list)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    evidence: list[EvidenceSchema] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    charts: list[ChartData] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    document_metrics: dict[str, Any] | None = None
    evaluation_results: dict[str, Any] | None = None
    report_markdown: str | None = None


# ── Document Analysis ───────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for document analysis."""

    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    analysis_type: str = Field(
        "general", pattern="^(general|summary|risks|recommendations|contradictions)$"
    )


class AnalyzeResponse(BaseModel):
    """Response from document analysis."""

    success: bool = True
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None


# ── Minimal RAG API (/upload, /query, /health) ──────────────────────────────


class UploadResponse(BaseModel):
    """Response after ``POST /upload`` (thin ingestion entry point)."""

    success: bool = True
    document_id: str = ""
    filename: str = ""
    chunks: int = 0
    deduped: int = 0
    status: str = "completed"
    error: str | None = None


class QueryRequest(BaseModel):
    """Request body for ``POST /query``."""

    query: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
    doc_id: str | None = Field(
        default=None,
        description="Optionally restrict retrieval to a single document ID.",
    )


class QuerySource(BaseModel):
    """A retrieved chunk cited by the answer."""

    chunk_id: str = ""
    document_id: str = ""
    source: str = ""
    score: float = 0.0
    snippet: str = ""
    page: int | None = None
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    """Response from ``POST /query``: grounded LLM answer + sources."""

    success: bool = True
    answer: str = ""
    sources: list[QuerySource] = Field(default_factory=list)
    chunks_retrieved: int = 0
    error: str | None = None
