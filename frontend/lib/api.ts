// Frontend API layer for AegisAI AgentSwarm.
// Connects to FastAPI (/api/*) for 8-agent orchestration, SSE streaming,
// background tasks, multi-format ingestion, dataset analytics, and artifact downloads.

const BASE = process.env.NEXT_PUBLIC_API_URL || "/api";
const API_BASE = BASE.replace(/\/+$/, "");

export function apiUrl(path: string): string {
  const endpoint = `${API_BASE}${path}`;
  if (/^https?:\/\//i.test(endpoint)) {
    return endpoint;
  }
  if (typeof window !== "undefined") {
    return new URL(endpoint, window.location.origin).toString();
  }
  const backend =
    process.env.AEGIS_BACKEND_URL || process.env.API_HOST || "http://localhost:8000";
  return new URL(endpoint, backend).toString();
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    let errorBody: unknown;
    try {
      errorBody = await res.json();
    } catch {
      errorBody = await res.text();
    }
    throw new Error(`API error ${res.status}: ${JSON.stringify(errorBody)}`);
  }
  return res.json();
}

// ── Shared Types ──────────────────────────────────────────────────────

export interface ResponseMetrics {
  hallucination_rate?: number;
  accuracy?: number;
  correctness?: number;
  context_precision?: number;
  context_recall?: number;
  task_success_rate?: number;
  tool_call_accuracy?: number;
  evidence_count?: number;
  agents_completed?: number;
}

export interface AgentWorkIllustration {
  agent: string;
  label?: string;
  agent_label?: string;
  role: string;
  status: string;
  summary: string;
  key_outputs?: string[];
  metrics?: Record<string, any>;
  duration_ms?: number;
  timestamp?: string;
}

export interface ArtifactItem {
  artifact_id: string;
  execution_id: string;
  filename: string;
  format: string;
  file_path?: string;
  size_bytes: number;
  mime_type: string;
  download_url: string;
  verified?: boolean;
  created_at?: string;
}

export interface SubtaskItem {
  id: string;
  title?: string;
  description: string;
  assigned_agent: string;
  dependencies?: string[];
  status?: string;
  expected_output?: string;
}

export interface HealthInfo {
  status: string;
  version: string;
  environment: string;
  upload_limits?: {
    max_files_per_batch: number;
    max_file_size_mb: number;
    max_total_size_mb: number;
    max_concurrent_uploads: number;
    supported_formats: string[];
    supports_urls: boolean;
    supports_repos: boolean;
  };
  pipeline_agents?: string[];
  models?: Record<string, any>;
  openlm?: Record<string, any>;
  qdrant?: Record<string, any>;
  chunking_defaults?: {
    chunk_size_words: number;
    chunk_overlap_pct: number;
    chunk_overlap_words: number;
  };
}

export interface PipelineStep {
  agent: string;
  status: string;
}

export interface ChartData {
  chart_type: string;
  title: string;
  labels: string[];
  values: number[];
  datasets?: Record<string, any>[];
  metadata?: Record<string, any>;
  image_data?: string;
  png_base64?: string;
}

export interface TableData {
  title: string;
  headers: string[];
  rows: string[][];
  metadata?: Record<string, any>;
}

// ── Pipeline API ──────────────────────────────────────────────────────

export interface PipelineResult {
  success: boolean;
  execution_id?: string;
  session_id?: string;
  task_type?: string;
  task_complexity?: string;
  report_title: string;
  report_markdown: string;
  action_items: string[];
  pipeline_trace: PipelineStep[];
  charts?: string[];
  chart_data?: ChartData[];
  table_data?: TableData[];
  execution_plan?: any;
  subtasks?: SubtaskItem[];
  agent_work_illustrations?: AgentWorkIllustration[];
  response_metrics?: ResponseMetrics;
  analysis_results?: Record<string, any>;
  artifacts?: ArtifactItem[];
  evaluation?: Record<string, any>;
  confidence_score?: number;
  revision_count?: number;
  warnings?: string[];
  error?: string;
}

export function runPipeline(
  task: string,
  options?: {
    doc_id?: string;
    document_ids?: string[];
    session_id?: string;
    requested_formats?: string[];
  }
): Promise<PipelineResult> {
  return request("/pipeline/run", {
    method: "POST",
    body: JSON.stringify({
      task,
      doc_id: options?.doc_id,
      document_ids: options?.document_ids || [],
      session_id: options?.session_id,
      requested_formats: options?.requested_formats || [],
    }),
  });
}

export function downloadReportPdf(markdown: string, title: string): void {
  fetch(apiUrl("/pipeline/report/pdf"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ markdown, title }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.blob();
    })
    .then((blob) => {
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${(title || "AegisAI_Report").replace(/[^A-Za-z0-9_-]+/g, "_")}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    })
    .catch((error) => {
      console.error("PDF download failed:", error);
    });
}

// ── Deliverable Artifacts API ─────────────────────────────────────────

export function getArtifactDownloadUrl(artifact: ArtifactItem): string {
  if (artifact.download_url) {
    if (artifact.download_url.startsWith("http")) return artifact.download_url;
    if (artifact.download_url.startsWith("/api/")) {
      return typeof window !== "undefined"
        ? new URL(artifact.download_url, window.location.origin).toString()
        : artifact.download_url;
    }
    return apiUrl(artifact.download_url);
  }
  return apiUrl(
    `/artifacts/${encodeURIComponent(artifact.execution_id)}/${encodeURIComponent(artifact.filename)}`
  );
}

export function listAllArtifacts(): Promise<{
  success: boolean;
  artifacts: ArtifactItem[];
  total: number;
}> {
  return request("/artifacts");
}

export function listExecutionArtifacts(executionId: string): Promise<{
  success: boolean;
  execution_id: string;
  artifacts: ArtifactItem[];
  total: number;
}> {
  return request(`/artifacts/${encodeURIComponent(executionId)}`);
}

export function generateArtifactsOnDemand(payload: {
  execution_id?: string;
  title: string;
  markdown: string;
  action_items?: string[];
  formats?: string[];
}): Promise<{
  success: boolean;
  execution_id: string;
  artifacts: ArtifactItem[];
  total: number;
  error?: string;
}> {
  return request("/artifacts/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ── Background Long-Running Tasks API ─────────────────────────────────

export interface TaskRecord {
  execution_id: string;
  session_id: string;
  user_query: string;
  state: string;
  progress_pct: number;
  current_agent: string;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  agent_steps?: PipelineStep[];
  agent_work_illustrations?: AgentWorkIllustration[];
  subtasks?: SubtaskItem[];
  artifacts?: ArtifactItem[];
  response_metrics?: ResponseMetrics;
  report_title?: string;
  report_markdown?: string;
  chart_data?: ChartData[];
  table_data?: TableData[];
  analysis_results?: Record<string, any>;
  errors?: string[];
}

export function submitBackgroundTask(payload: {
  task: string;
  doc_id?: string;
  document_ids?: string[];
  session_id?: string;
  requested_formats?: string[];
}): Promise<TaskRecord & { success: boolean }> {
  return request("/tasks", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listBackgroundTasks(): Promise<{
  success: boolean;
  tasks: TaskRecord[];
  total: number;
}> {
  return request("/tasks");
}

export function getBackgroundTaskStatus(
  executionId: string
): Promise<TaskRecord & { success: boolean }> {
  return request(`/tasks/${encodeURIComponent(executionId)}`);
}

export function cancelBackgroundTask(
  executionId: string
): Promise<{ success: boolean; execution_id: string; state: string }> {
  return request(`/tasks/${encodeURIComponent(executionId)}/cancel`, {
    method: "POST",
  });
}

export function resumeBackgroundTask(
  executionId: string
): Promise<TaskRecord & { success: boolean }> {
  return request(`/tasks/${encodeURIComponent(executionId)}/resume`, {
    method: "POST",
  });
}

// ── Dataset Analytics Profiling API ───────────────────────────────────

export interface DatasetProfileResult {
  success: boolean;
  document_id: string;
  filename: string;
  profile: Record<string, any>;
  charts: ChartData[];
  tables: TableData[];
  error?: string;
}

export function profileDataset(documentId: string): Promise<DatasetProfileResult> {
  return request(`/analytics/profile/${encodeURIComponent(documentId)}`, {
    method: "POST",
  });
}

// ── SSE Streaming ─────────────────────────────────────────────────────

export interface SSEEvent {
  agent: string;
  type:
    | "agent_start"
    | "agent_output"
    | "agent_complete"
    | "task_state"
    | "pipeline_complete"
    | "error";
  output?: string;
  status?: string;
  execution_id?: string;
  session_id?: string;
  task_type?: string;
  task_complexity?: string;
  report_title?: string;
  report_markdown?: string;
  action_items?: string[];
  charts?: string[];
  chart_data?: ChartData[];
  table_data?: TableData[];
  subtasks?: SubtaskItem[];
  execution_plan?: any;
  agent_work_illustrations?: AgentWorkIllustration[];
  response_metrics?: ResponseMetrics;
  analysis_results?: Record<string, any>;
  artifacts?: ArtifactItem[];
  evaluation?: Record<string, any>;
}

export function streamPipeline(
  task: string,
  onEvent: (event: SSEEvent) => void,
  onError?: (error: string) => void,
  onComplete?: () => void,
  docId?: string,
  options?: { documentIds?: string[]; formats?: string[]; sessionId?: string }
): AbortController {
  const url = new URL(apiUrl("/pipeline/stream"));
  url.searchParams.set("task", task);
  if (docId) {
    url.searchParams.set("doc_id", docId);
  }
  if (options?.documentIds && options.documentIds.length > 0) {
    url.searchParams.set("document_ids", options.documentIds.join(","));
  }
  if (options?.formats && options.formats.length > 0) {
    url.searchParams.set("formats", options.formats.join(","));
  }
  if (options?.sessionId) {
    url.searchParams.set("session_id", options.sessionId);
  }

  const controller = new AbortController();

  fetch(url, { signal: controller.signal })
    .then(async (response) => {
      if (!response.ok) {
        onError?.(`HTTP ${response.status}`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onError?.("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent: Partial<SSEEvent> = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, "");
          if (line.startsWith("event: ")) {
            currentEvent.type = line.slice(7).trim() as SSEEvent["type"];
          } else if (line.startsWith("data: ")) {
            try {
              const savedType = currentEvent.type;
              const data = JSON.parse(line.slice(6));
              currentEvent = { ...currentEvent, ...data };
              if (savedType) currentEvent.type = savedType;
            } catch {
              // ignore malformed SSE chunk
            }
          } else if (line === "") {
            if (currentEvent.type) {
              const sseEvent: SSEEvent = {
                agent: (currentEvent.agent as string) || "",
                type: (currentEvent.type as SSEEvent["type"]) || "agent_start",
                output: (currentEvent.output as string) || "",
                status: (currentEvent.status as string) || "",
                execution_id: currentEvent.execution_id,
                session_id: currentEvent.session_id,
                task_type: currentEvent.task_type,
                task_complexity: currentEvent.task_complexity,
                report_title: (currentEvent.report_title as string) || "",
                report_markdown: (currentEvent.report_markdown as string) || "",
                action_items: (currentEvent.action_items as string[]) || [],
                charts: (currentEvent.charts as string[]) || [],
                chart_data: (currentEvent.chart_data as ChartData[]) || [],
                table_data: (currentEvent.table_data as TableData[]) || [],
                subtasks: (currentEvent.subtasks as SubtaskItem[]) || [],
                execution_plan: currentEvent.execution_plan,
                agent_work_illustrations:
                  (currentEvent.agent_work_illustrations as AgentWorkIllustration[]) || [],
                response_metrics: currentEvent.response_metrics,
                analysis_results: currentEvent.analysis_results,
                artifacts: (currentEvent.artifacts as ArtifactItem[]) || [],
                evaluation: (currentEvent.evaluation as Record<string, any>) || {},
              };

              if (sseEvent.type === "pipeline_complete") {
                onEvent(sseEvent);
                onComplete?.();
                return;
              }

              if (sseEvent.type === "error") {
                onError?.(sseEvent.output || "Pipeline error");
                return;
              }

              onEvent(sseEvent);
              currentEvent = {};
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError?.(err.message);
      }
    });

  return controller;
}

// ── Health & History API ──────────────────────────────────────────────

export function getHealth(): Promise<HealthInfo> {
  return request("/health");
}

export interface HistoryRun {
  execution_id?: string;
  task: string;
  report_title: string;
  report_markdown: string;
  timestamp: string;
  agent_count: number;
  response_metrics?: ResponseMetrics;
  artifacts?: ArtifactItem[];
}

export interface HistoryResult {
  success: boolean;
  runs: HistoryRun[];
  error?: string;
}

export function getHistory(): Promise<HistoryResult> {
  return request("/pipeline/history", { method: "POST" });
}

// ── Documents API ─────────────────────────────────────────────────────

export interface DocItem {
  id: string;
  filename: string;
  uploaded_at: string;
  status: string;
  chunks: number;
  error: string;
  source_type: string;
  file_size_bytes: number;
  parser_used?: string;
  is_tabular?: boolean;
}

export interface ChunkMeta {
  filename: string;
  type: string;
  source_type: string;
  chunk_index: number;
  doc_id: string;
  content_hash: string;
  token_count: number;
  word_count?: number;
  char_offset: number;
  page_number: number | null;
  section_path: string;
  parser_used?: string;
  chunk_strategy?: string;
  text: string;
}

export interface ChunkListResponse {
  success: boolean;
  doc_id: string;
  chunks: ChunkMeta[];
  total_chunks: number;
  page: number;
  page_size: number;
  total_pages: number;
  error?: string;
}

export interface DocListResult {
  success: boolean;
  documents: DocItem[];
  error?: string;
}

export function listDocuments(): Promise<DocListResult> {
  return request("/documents");
}

export function getDocumentChunks(
  documentId: string,
  page = 1,
  pageSize = 50
): Promise<ChunkListResponse> {
  return request(
    `/documents/${encodeURIComponent(documentId)}/chunks?page=${page}&page_size=${pageSize}`
  );
}

export interface UploadResult {
  success: boolean;
  filename?: string;
  chunks?: number;
  error?: string;
  source_type?: string;
  document_id?: string;
  status?: "ready" | "processing" | "error";
  parser_used?: string;
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(apiUrl("/documents/upload"), {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const textBody = await res.text();
    throw new Error(`Upload failed (${res.status}): ${textBody}`);
  }
  return res.json();
}

export interface BatchUploadItem {
  filename: string;
  success: boolean;
  chunks: number;
  error: string;
  document_id?: string;
  status?: string;
}

export interface BatchUploadResult {
  success: boolean;
  total_files: number;
  successful: number;
  failed: number;
  total_chunks: number;
  results: BatchUploadItem[];
  error?: string;
}

export function isStillProcessing(result: BatchUploadResult): boolean {
  return (result.results ?? []).some(
    (item) => item.status === "processing" && (item.chunks ?? 0) === 0
  );
}

export async function batchUpload(files: File[]): Promise<BatchUploadResult> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }

  const res = await fetch(apiUrl("/documents/batch-upload"), {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const textBody = await res.text();
    throw new Error(`Batch upload failed (${res.status}): ${textBody}`);
  }
  return res.json();
}

export interface SourceIngestResult {
  success: boolean;
  source: string;
  source_type: string;
  chunks: number;
  document_id?: string;
  error?: string;
}

export async function ingestSource(
  source: string,
  sourceType: "url" | "repo"
): Promise<SourceIngestResult> {
  return request("/documents/ingest", {
    method: "POST",
    body: JSON.stringify({ source, source_type: sourceType }),
  });
}

export async function deleteDocument(
  id: string
): Promise<{ success: boolean; error?: string }> {
  return request(`/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

// ── Conversational Chat API ───────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface ChatRequest {
  message: string;
  history: ChatMessage[];
  mode: "auto" | "plan" | "research" | "analysis";
  context_documents: string[];
  document_ids?: string[];
  session_id?: string;
  requested_formats?: string[];
  tool_settings?: Record<string, any>;
}

export interface Evidence {
  id: string;
  claim: string;
  source: string;
  supporting_passage: string;
  confidence: number;
  relevance: number;
}

export interface ChatResponse {
  success: boolean;
  execution_id?: string;
  session_id?: string;
  response: string;
  plan?: Record<string, any>;
  subtasks?: SubtaskItem[];
  agent_trace: Array<{ agent: string; status: string }>;
  agent_work_illustrations?: AgentWorkIllustration[];
  response_metrics?: ResponseMetrics;
  artifacts?: ArtifactItem[];
  analysis_results?: Record<string, any>;
  error?: string;
  evidence?: Evidence[];
  citations?: string[];
  charts?: ChartData[];
  tables?: TableData[];
  document_metrics?: Record<string, any>;
  evaluation_results?: Record<string, any>;
  report_markdown?: string;
}

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export interface ChatSSEEvent {
  agent: string;
  type: "agent_start" | "agent_output" | "agent_complete" | "chat_complete" | "error";
  status?: string;
  output?: string;
  illustration?: AgentWorkIllustration;
  success?: boolean;
  execution_id?: string;
  session_id?: string;
  response?: string;
  plan?: Record<string, any>;
  subtasks?: SubtaskItem[];
  agent_trace?: Array<{ agent: string; status: string }>;
  agent_work_illustrations?: AgentWorkIllustration[];
  response_metrics?: ResponseMetrics;
  artifacts?: ArtifactItem[];
  charts?: ChartData[];
  tables?: TableData[];
  evidence?: Evidence[];
  citations?: string[];
  report_markdown?: string;
  evaluation_results?: Record<string, any>;
}

export function streamChat(
  req: ChatRequest,
  onEvent: (event: ChatSSEEvent) => void,
  onError?: (error: string) => void,
  onComplete?: (data: ChatSSEEvent) => void
): AbortController {
  const controller = new AbortController();

  fetch(apiUrl("/chat/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        onError?.(`HTTP ${response.status}`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        onError?.("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent: Partial<ChatSSEEvent> = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, "");
          if (line.startsWith("event: ")) {
            currentEvent.type = line.slice(7).trim() as ChatSSEEvent["type"];
          } else if (line.startsWith("data: ")) {
            try {
              const savedType = currentEvent.type;
              const data = JSON.parse(line.slice(6));
              currentEvent = { ...currentEvent, ...data };
              if (savedType) currentEvent.type = savedType;
            } catch {
              // ignore malformed chunk
            }
          } else if (line === "") {
            if (currentEvent.type) {
              const sseEvent: ChatSSEEvent = {
                agent: (currentEvent.agent as string) || "",
                type: currentEvent.type,
                output: currentEvent.output || "",
                status: currentEvent.status || "",
                illustration: currentEvent.illustration,
                success: currentEvent.success,
                execution_id: currentEvent.execution_id,
                session_id: currentEvent.session_id,
                response: currentEvent.response,
                plan: currentEvent.plan,
                subtasks: currentEvent.subtasks,
                agent_trace: currentEvent.agent_trace,
                agent_work_illustrations: currentEvent.agent_work_illustrations,
                response_metrics: currentEvent.response_metrics,
                artifacts: currentEvent.artifacts,
                charts: currentEvent.charts,
                tables: currentEvent.tables,
                evidence: currentEvent.evidence,
                citations: currentEvent.citations,
                report_markdown: currentEvent.report_markdown,
                evaluation_results: currentEvent.evaluation_results,
              };

              if (sseEvent.type === "chat_complete") {
                onEvent(sseEvent);
                onComplete?.(sseEvent);
                return;
              }

              if (sseEvent.type === "error") {
                onError?.(sseEvent.output || "Chat pipeline error");
                return;
              }

              onEvent(sseEvent);
              currentEvent = {};
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError?.(err.message);
      }
    });

  return controller;
}

// ── Document Analysis API ─────────────────────────────────────────────

export interface AnalyzeRequest {
  document_id: string;
  question: string;
  analysis_type:
    | "general"
    | "summary"
    | "risks"
    | "recommendations"
    | "contradictions";
}

export interface AnalyzeResponse {
  success: boolean;
  answer: string;
  evidence: string[];
  confidence: number;
  error?: string;
}

export async function analyzeDocument(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request("/documents/analyze", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/**
 * Strips raw UUIDs, numeric timestamp prefixes, and directory path clutter from filenames.
 * Example: "9eb51991-4870-5f52-b1db-c39a6165ce3b_customers.csv" -> "customers.csv"
 */
export function formatCleanFilename(name: string): string {
  if (!name) return "";
  let clean = name.trim();
  // Strip UUID prefix (standard 8-4-4-4-12 hex format)
  clean = clean.replace(/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_?/, "");
  // Strip Unix timestamp prefix (10+ digits)
  clean = clean.replace(/^\d{10,}_?/, "");
  // Extract basename from paths
  clean = clean.split(/[/\\]/).pop() || clean;
  return clean || name;
}

