# backend/aegis/schemas/agent.py
"""Shared Pydantic models for every agent's input and output contracts.

All specialized agents (Supervisor, Planner, Researcher, Retriever, Validator,
Reasoner/Analyst, Critic, ReportGenerator, MemoryManager) produce and consume
these shapes. Keeping them in one place avoids circular imports between
``agents/``, ``api/``, ``services/``, and ``memory/``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Foundation types ───────────────────────────────────────────────────────


class Evidence(BaseModel):
    """Structured evidence item with claim-source relationship."""

    model_config = {"extra": "ignore"}

    id: str = Field(default="", description="Unique evidence identifier")
    claim: str = Field(..., description="The claim being supported")
    source: str = Field(..., description="Source document or URL")
    supporting_passage: str = Field(..., description="Exact passage from source supporting the claim")
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the source (page, section, URL, etc.)",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in this evidence")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to the task")
    timestamp: str = Field(default_factory=_utc_iso)


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    model_config = {"extra": "ignore"}

    chunk_id: str = Field(default="", description="Stable unique identifier for this chunk")
    document_id: str = Field(default="", description="ID of the source document")
    text: str = Field(default="", description="Chunk text content")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity score from vector search")
    source: str = Field(default="", description="Filename or source identifier")
    page: int | None = Field(default=None, description="Page number if available")
    section_heading: str = Field(default="", description="Section heading if available")
    chunk_index: int = Field(default=0, description="Index within the document")
    extraction_method: str = Field(default="", description="Parser/extraction method used")
    embedding_model: str = Field(default="", description="Embedding model used for this chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentInput(BaseModel):
    """Generic input envelope for every agent."""

    task: str = Field(..., description="The user's original task prompt.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context passed between agents.",
    )


class AgentOutput(BaseModel):
    """Standard output envelope for every agent."""

    model_config = {"extra": "ignore"}

    success: bool = True
    summary: str = ""
    details: str = ""
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Supervisor ────────────────────────────────────────────────────────────


class SupervisorDecision(AgentOutput):
    """Structured routing and classification output from the Supervisor agent."""

    model_config = {"extra": "ignore"}

    intent: str = Field(default="research_and_analysis", description="Classified user intent")
    task_type: str = Field(
        default="hybrid",
        description="Task category: simple_qa, document_intelligence, external_research, data_analysis, deliverable_generation, hybrid",
    )
    task_complexity: str = Field(default="medium", description="low, medium, or high")
    selected_agents: list[str] = Field(
        default_factory=lambda: [
            "supervisor",
            "planner",
            "retriever",
            "researcher",
            "validator",
            "reasoner",
            "critic",
            "report_generator",
        ],
        description="Ordered list of agents assigned for this execution",
    )
    requires_retrieval: bool = True
    requires_research: bool = True
    requires_data_analysis: bool = False
    requires_visualizations: bool = False
    requested_deliverables: list[str] = Field(default_factory=list)


# ── Planner ───────────────────────────────────────────────────────────────


class SubtaskItem(BaseModel):
    """Structured subtask specification per Section 7.2."""

    model_config = {"extra": "ignore"}

    task_id: str = Field(default="", description="Unique subtask ID (e.g., T1)")
    description: str = Field(default="", description="Actionable subtask description")
    assigned_agent: str = Field(default="retriever", description="Responsible specialized agent")
    dependencies: list[str] = Field(default_factory=list, description="Dependent task_ids")
    priority: str = Field(default="medium", description="high, medium, or low")
    expected_output: str = Field(default="", description="Expected deliverable from this subtask")
    status: str = Field(default="pending", description="pending, running, completed, failed, skipped")
    validation_criteria: str = Field(default="", description="Criteria to verify subtask completion")


class PlanStep(BaseModel):
    """A single step in a plan (backward-compatible with SubtaskItem enrichment)."""

    model_config = {"extra": "ignore"}

    step: int = Field(..., ge=1)
    agent: str = Field(..., description="Agent name responsible.")
    description: str = ""
    dependencies: list[int] = Field(default_factory=list)
    task_id: str = ""
    assigned_agent: str = ""
    priority: str = "medium"
    expected_output: str = ""
    status: str = "pending"
    validation_criteria: str = ""
    tools: list[str] = Field(default_factory=list)

    def to_subtask(self) -> SubtaskItem:
        return SubtaskItem(
            task_id=self.task_id or f"T{self.step}",
            description=self.description,
            assigned_agent=self.assigned_agent or self.agent,
            dependencies=[f"T{d}" for d in self.dependencies],
            priority=self.priority or "medium",
            expected_output=self.expected_output or f"Structured output from {self.agent}",
            status=self.status or "pending",
            validation_criteria=self.validation_criteria or "Grounded in verified evidence without unsupported claims",
        )


class AgentPlan(BaseModel):
    """Complete plan returned by the Planner agent."""

    model_config = {"extra": "ignore"}

    objective: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    subtasks: list[SubtaskItem] = Field(default_factory=list)
    estimated_complexity: str = "medium"


class AgentStep(BaseModel):
    """A step executed by one agent during workflow (for traceability)."""

    agent: str
    input: str = ""
    output: str = ""
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=_utc_iso)


# ── Researcher / Retriever ─────────────────────────────────────────────────


_EVIDENCE_CLAIM_KEYS = ("claim", "finding", "statement", "assertion", "title")
_EVIDENCE_PASSAGE_KEYS = (
    "supporting_passage",
    "passage",
    "excerpt",
    "quote",
    "text",
    "content",
)
_EVIDENCE_SOURCE_KEYS = ("source", "url", "document", "filename", "origin")


def coerce_evidence(item: Any) -> Evidence | None:
    """Build an ``Evidence`` from a loosely-shaped dict (e.g. LLM output)."""
    if isinstance(item, Evidence):
        return item
    if not isinstance(item, dict):
        return None

    try:
        return Evidence(**item)
    except Exception:
        pass

    claim = next((str(item[k]) for k in _EVIDENCE_CLAIM_KEYS if item.get(k)), "")
    passage = next(
        (str(item[k]) for k in _EVIDENCE_PASSAGE_KEYS if item.get(k)), ""
    )
    source = next((str(item[k]) for k in _EVIDENCE_SOURCE_KEYS if item.get(k)), "")
    if not claim and not passage:
        return None

    def _clamp(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    try:
        return Evidence(
            id=str(item.get("id", "")),
            claim=claim or passage,
            source=source or "unknown",
            supporting_passage=passage or claim,
            source_metadata=item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {},
            confidence=_clamp(item.get("confidence"), 0.5),
            relevance=_clamp(item.get("relevance"), 0.5),
        )
    except Exception:
        return None


class ResearchResult(BaseModel):
    """Result produced by the Researcher or Retriever agent."""

    model_config = {"extra": "ignore"}

    query: str = ""
    content: str = ""
    source: str = ""
    relevance_score: float = 0.0
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# ── Validator ─────────────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Output of the Validator agent."""

    model_config = {"extra": "ignore"}

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_hallucination: bool = False
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verified_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    numerical_checks: list[dict[str, Any]] = Field(default_factory=list)


# ── Critic ─────────────────────────────────────────────────────────────────


class CritiqueResult(BaseModel):
    """Output of the Critic agent, supporting bounded revision routing."""

    model_config = {"extra": "ignore"}

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
    approved: bool = True
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    hallucination_flags: list[str] = Field(default_factory=list)
    revision_instructions: str = ""
    target_revision_agent: str = "reasoner"


# ── Charts, Tables & Deliverables ──────────────────────────────────────────


class ChartData(BaseModel):
    """Chart configuration and rendered base64 PNG for visualization."""

    model_config = {"extra": "ignore"}

    chart_id: str = Field(default="", description="Unique chart identifier")
    chart_type: str = Field(
        default="bar",
        description="bar, line, pie, histogram, scatter, boxplot, heatmap, correlation_matrix, time_series, comparative, distribution, data_quality",
    )
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    image_data: str = Field(default="", description="Base64-encoded PNG image data")
    file_path: str = Field(default="", description="Saved PNG path if persisted")

    @property
    def png_base64(self) -> str:
        """Alias for ``image_data`` base64 PNG bytes."""
        return self.image_data


class TableData(BaseModel):
    """Structured table data for reports and analytics."""

    model_config = {"extra": "ignore"}

    title: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactItem(BaseModel):
    """Metadata for a generated downloadable deliverable linked to execution_id."""

    model_config = {"extra": "ignore"}

    artifact_id: str = ""
    execution_id: str = ""
    filename: str = ""
    format: str = ""  # pdf, docx, pptx, xlsx, csv, html, md
    file_path: str = ""
    size_bytes: int = 0
    mime_type: str = ""
    download_url: str = ""
    verified: bool = True
    created_at: str = Field(default_factory=_utc_iso)


class ReportOutput(BaseModel):
    """Output of the Report Generator & Deliverable Engine."""

    model_config = {"extra": "ignore"}

    title: str = "AegisAI Report"
    executive_summary: str = ""
    sections: list[dict[str, str]] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    markdown: str = ""
    conversational_response: str = ""
    charts: list[ChartData] = Field(default_factory=list)
    artifacts: list[ArtifactItem] = Field(default_factory=list)


# ── Memory ─────────────────────────────────────────────────────────────────


class MemoryEntry(BaseModel):
    """A single entry stored in or retrieved from Qdrant."""

    id: str = ""
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


# ── Metrics, Sidebar Live Evaluation & Agent Work Illustrations ────────────


class DocumentMetrics(BaseModel):
    """Computed metrics for ingested documents."""

    model_config = {"extra": "ignore"}

    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_density: float = Field(default=0.0, ge=0.0)
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity: int = Field(default=0)
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_count: int = Field(default=0)
    word_count: int = Field(default=0)
    table_count: int = Field(default=0)
    chart_data_points: int = Field(default=0)


class EvaluationResult(BaseModel):
    """Evaluation results from the evaluator agents."""

    model_config = {"extra": "ignore"}

    hallucination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    factual_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ResponseMetrics(BaseModel):
    """Per-response live evaluation metrics surfaced in the Workspace Sidebar."""

    model_config = {"extra": "ignore"}

    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    context_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    task_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_call_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentWorkIllustration(BaseModel):
    """Structured summary/illustration of work executed by an individual agent."""

    model_config = {"extra": "ignore"}

    agent: str
    label: str = ""
    role: str = ""
    status: str = "completed"  # completed, skipped, revised, error
    summary: str = ""
    key_outputs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=_utc_iso)


# Backward-compatible aliases
ArtifactMetadata = ArtifactItem
ResponseEvaluationMetrics = ResponseMetrics
