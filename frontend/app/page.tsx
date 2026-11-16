"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  Activity,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronRight,
  Clock,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  Info,
  Layers,
  Lightbulb,
  Loader2,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  Tag,
  UploadCloud,
  Workflow,
  XCircle,
  Zap,
} from "lucide-react";
import { Sidebar, tabs, type TabId } from "@/components/ui/Sidebar";
import {
  ContactDeveloper,
  NavbarSocialIcons,
  type DeveloperLinkKey,
} from "@/components/ui/DeveloperLinks";
import { AgentTrace } from "@/components/workflow/AgentTrace";
import { ReportView } from "@/components/report/ReportView";
import { FileUpload } from "@/components/ui/FileUpload";
import { DocumentList } from "@/components/documents/DocumentList";
import { HistoryList } from "@/components/history/HistoryList";
import { Chat } from "@/components/workflow/Chat";
import {
  cancelBackgroundTask,
  downloadReportPdf,
  formatCleanFilename,
  generateArtifactsOnDemand,
  getArtifactDownloadUrl,
  getHealth,
  getHistory,
  listAllArtifacts,
  listBackgroundTasks,
  listDocuments,
  profileDataset,
  resumeBackgroundTask,
  submitBackgroundTask,
  type ArtifactItem,
  type DatasetProfileResult,
  type DocItem,
  type HealthInfo,
  type HistoryRun,
  type PipelineResult,
  type ResponseMetrics,
  type TaskRecord,
} from "@/lib/api";

const SUPPORTED_FORMATS_MATRIX = [
  { type: "PDF & Scanned PDF", extensions: ".pdf", limit: "Up to 500 MB per file · Up to 100 files / 1 GB batch", badge: "badge-purple" },
  { type: "Images (OCR)", extensions: ".png, .jpg, .jpeg, .webp, .bmp, .tiff", limit: "Up to 500 MB per file · Scanned text & tables supported", badge: "badge-lemon" },
  { type: "Word Documents", extensions: ".docx, .doc", limit: "Up to 500 MB per file · Sections & tables preserved", badge: "badge-blue" },
  { type: "Presentations", extensions: ".pptx", limit: "Up to 500 MB per file · Slides & speaker notes", badge: "badge-amber" },
  { type: "Spreadsheets", extensions: ".xlsx, .xls, .csv", limit: "Up to 500 MB per file · Multi-sheet analytics & charts", badge: "badge-green" },
  { type: "Structured & Web", extensions: ".json, .html, .htm, .md, .txt", limit: "Up to 500 MB per file · Stored permanently until deleted", badge: "badge-cyan" },
  { type: "Archives, URLs & Repos", extensions: ".zip, https://, git repo", limit: "Up to 500 MB archive / URL · Public GitHub & GitLab repos", badge: "badge-rose" },
];

const ALL_DELIVERABLE_FORMATS = [
  { id: "pdf", label: "PDF" },
  { id: "docx", label: "DOCX" },
  { id: "pptx", label: "PPTX" },
  { id: "xlsx", label: "XLSX" },
  { id: "csv", label: "CSV" },
  { id: "html", label: "HTML" },
  { id: "md", label: "Markdown" },
];

const EIGHT_AGENT_ROSTER = [
  {
    step: "01",
    name: "Supervisor Agent",
    role: "Intent triage, task complexity scoring, deliverable detection, and dynamic multi-agent routing.",
    accent: "badge-purple",
  },
  {
    step: "02",
    name: "Planner Agent",
    role: "Dependency-aware subtask decomposition with assigned specialists and expected outputs.",
    accent: "badge-lemon",
  },
  {
    step: "03",
    name: "Retriever Agent",
    role: "Qdrant hybrid dense (BGE-M3 / MiniLM) + lexical BM25 Reciprocal Rank Fusion (RRF) search.",
    accent: "badge-purple",
  },
  {
    step: "04",
    name: "Researcher Agent",
    role: "Multi-source synthesis combining grounded corpus chunks with external research tools.",
    accent: "badge-lemon",
  },
  {
    step: "05",
    name: "Validator Agent",
    role: "Claim-by-claim factual verification, citation cross-checking, and contradiction flagging.",
    accent: "badge-purple",
  },
  {
    step: "06",
    name: "Reasoner & Analyst Agent",
    role: "Quantitative statistical computation (pandas/scipy/sklearn), charts, and multi-step synthesis.",
    accent: "badge-lemon",
  },
  {
    step: "07",
    name: "Critic Agent",
    role: "Adversarial quality gate enforcing completeness and triggering bounded revision loops.",
    accent: "badge-purple",
  },
  {
    step: "08",
    name: "Report & Deliverable Engine",
    role: "Generates executive reports and verified PDF, DOCX, PPTX, XLSX, CSV, HTML, and MD artifacts.",
    accent: "badge-lemon",
  },
];

const QUICK_PROMPTS = [
  "Analyze all uploaded documents, extract key quantitative metrics, and generate an executive summary with PDF and PPTX deliverables.",
  "Compare risks, financial trends, and strategic recommendations across our indexed corpus with statistical charts.",
  "Perform a deep technical audit of our uploaded dataset, identify outliers and correlations, and export an XLSX + PDF report.",
];

function formatBytes(bytes: number): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

function isTabularDocument(doc: DocItem): boolean {
  if (doc.is_tabular) return true;
  const lower = (doc.filename || "").toLowerCase();
  return (
    lower.endsWith(".csv") ||
    lower.endsWith(".xlsx") ||
    lower.endsWith(".xls") ||
    lower.endsWith(".json")
  );
}

function formatChartTypeBadge(type: string): string {
  switch (type?.toLowerCase()) {
    case "data_quality":
      return "Data Quality Audit";
    case "correlation_matrix":
      return "Correlation Heatmap";
    case "distribution":
      return "Distribution (KDE)";
    case "bar":
      return "Categorical Bar";
    case "boxplot":
      return "Outlier & Boxplot";
    case "scatter":
      return "Bivariate Scatter";
    case "line":
      return "Trend Line";
    default:
      return type ? type.toUpperCase() : "Visual";
  }
}

export default function Home() {
  const [section, setSection] = useState<TabId>("home");
  const [activeContact, setActiveContact] = useState<DeveloperLinkKey | null>(null);
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  // Live Sidebar state & per-response evaluation metrics
  const [liveMetrics, setLiveMetrics] = useState<ResponseMetrics | null>(null);
  const [taskState, setTaskState] = useState<string>("IDLE");
  const [activeAgent, setActiveAgent] = useState<string>("");
  const [lastExecutionId, setLastExecutionId] = useState<string>("");

  // Platform metadata
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [docsRefreshCounter, setDocsRefreshCounter] = useState<number>(0);
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [requestedFormats, setRequestedFormats] = useState<string[]>(["pdf", "md"]);

  // Background Long-Running Tasks state
  const [bgTasks, setBgTasks] = useState<TaskRecord[]>([]);
  const [submittingBg, setSubmittingBg] = useState(false);

  // Analytics Lab state
  const [analyticsDocId, setAnalyticsDocId] = useState<string>("");
  const [profiling, setProfiling] = useState(false);
  const [profileResult, setProfileResult] = useState<DatasetProfileResult | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);

  // Deliverables Hub state
  const [allArtifacts, setAllArtifacts] = useState<ArtifactItem[]>([]);
  const [artifactsLoading, setArtifactsLoading] = useState(false);
  const [customTitle, setCustomTitle] = useState("");
  const [customMarkdown, setCustomMarkdown] = useState("");
  const [customGenerating, setCustomGenerating] = useState(false);

  const mainRef = useRef<HTMLElement | null>(null);

  const fetchMeta = useCallback(async () => {
    try {
      const [h, d, r, t] = await Promise.all([
        getHealth().catch(() => null),
        listDocuments().catch(() => ({ success: false, documents: [] as DocItem[] })),
        getHistory().catch(() => ({ success: false, runs: [] as HistoryRun[] })),
        listBackgroundTasks().catch(() => ({ success: false, tasks: [] as TaskRecord[], total: 0 })),
      ]);
      if (h) setHealth(h);
      if (d.success) {
        setDocs(d.documents);
        setDocsRefreshCounter((c) => c + 1);
        setAnalyticsDocId((prev) => {
          if (prev) return prev;
          const firstTab = d.documents.find(isTabularDocument);
          return firstTab ? firstTab.id : "";
        });
      }
      if (r.success) {
        setRuns(r.runs);
        if (r.runs.length > 0 && r.runs[0].response_metrics) {
          setLiveMetrics((prev) => prev ?? r.runs[0].response_metrics ?? null);
        }
      }
      if (t.success) setBgTasks(t.tasks);
    } catch {
      // non-fatal
    }
  }, []);

  const refreshArtifacts = useCallback(async () => {
    setArtifactsLoading(true);
    try {
      const res = await listAllArtifacts();
      if (res.success) setAllArtifacts(res.artifacts);
    } catch {
      // ignore
    } finally {
      setArtifactsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMeta();
  }, [fetchMeta]);

  useEffect(() => {
    if (section === "deliverables") {
      refreshArtifacts();
    }
  }, [section, refreshArtifacts]);

  // Poll active background tasks automatically while any task is non-terminal
  useEffect(() => {
    const hasActiveBg = bgTasks.some((t) =>
      [
        "PENDING",
        "PLANNING",
        "RUNNING",
        "WAITING_FOR_TOOL",
        "VALIDATING",
        "REVISING",
        "GENERATING_ARTIFACT",
      ].includes(t.state)
    );
    if (!hasActiveBg) return;

    const timer = setInterval(async () => {
      try {
        const res = await listBackgroundTasks();
        if (res.success) {
          setBgTasks(res.tasks);
          const newestDone = res.tasks.find((item) => item.state === "COMPLETED" && item.response_metrics);
          if (newestDone?.response_metrics) {
            setLiveMetrics(newestDone.response_metrics);
          }
        }
      } catch {
        // ignore transient poll error
      }
    }, 2500);

    return () => clearInterval(timer);
  }, [bgTasks]);

  const toggleFormat = (fmt: string) => {
    setRequestedFormats((prev) =>
      prev.includes(fmt) ? prev.filter((f) => f !== fmt) : [...prev, fmt]
    );
  };

  const startStreamingPipeline = () => {
    if (!task.trim() || running) return;
    setResult(null);
    setPipelineError(null);
    setTaskState("PLANNING");
    setActiveAgent("supervisor");
    setRunning(true);
  };

  const stopStreamingPipeline = () => {
    setRunning(false);
    setTaskState("CANCELLED");
    setActiveAgent("");
  };

  const handleQueueBackgroundTask = async () => {
    if (!task.trim() || submittingBg) return;
    setSubmittingBg(true);
    setPipelineError(null);
    try {
      const created = await submitBackgroundTask({
        task: task.trim(),
        doc_id: selectedDocId || undefined,
        document_ids: selectedDocId ? [selectedDocId] : [],
        requested_formats: requestedFormats,
      });
      if (created.execution_id) {
        setLastExecutionId(created.execution_id);
        setTaskState(created.state || "PENDING");
      }
      const listRes = await listBackgroundTasks();
      if (listRes.success) setBgTasks(listRes.tasks);
    } catch (err: any) {
      setPipelineError(err?.message || "Failed to queue background task");
    } finally {
      setSubmittingBg(false);
    }
  };

  const handlePipelineComplete = useCallback(
    (pipelineResult: PipelineResult) => {
      setResult(pipelineResult);
      setRunning(false);
      setTaskState("COMPLETED");
      setActiveAgent("");
      if (pipelineResult.execution_id) {
        setLastExecutionId(pipelineResult.execution_id);
      }
      if (pipelineResult.response_metrics) {
        setLiveMetrics(pipelineResult.response_metrics);
      }
      fetchMeta();
    },
    [fetchMeta]
  );

  const handlePipelineError = useCallback((error: string) => {
    setPipelineError(error);
    setRunning(false);
    setTaskState("FAILED");
    setActiveAgent("");
  }, []);

  const handleLoadTaskIntoWorkspace = (t: TaskRecord) => {
    setResult({
      success: t.state === "COMPLETED",
      execution_id: t.execution_id,
      session_id: t.session_id,
      report_title: t.report_title || t.user_query,
      report_markdown: t.report_markdown || "",
      action_items: [],
      pipeline_trace: t.agent_steps || [],
      chart_data: t.chart_data || [],
      table_data: t.table_data || [],
      subtasks: t.subtasks || [],
      agent_work_illustrations: t.agent_work_illustrations || [],
      response_metrics: t.response_metrics,
      analysis_results: t.analysis_results,
      artifacts: t.artifacts || [],
    });
    if (t.response_metrics) setLiveMetrics(t.response_metrics);
    setLastExecutionId(t.execution_id);
    setTaskState(t.state);
    setSection("chat");
  };

  const handleRunProfile = useCallback(async (docIdToRun?: string) => {
    const targetId = docIdToRun || analyticsDocId;
    if (!targetId) return;
    setProfiling(true);
    setProfileError(null);
    try {
      const res = await profileDataset(targetId);
      if (!res.success) {
        setProfileError(res.error || "Dataset profiling returned an error.");
      } else {
        setProfileResult(res);
      }
    } catch (err: any) {
      setProfileError(err?.message || "Failed to profile dataset.");
    } finally {
      setProfiling(false);
    }
  }, [analyticsDocId]);

  const handleCustomArtifactGenerate = async () => {
    if (!customMarkdown.trim() || customGenerating) return;
    setCustomGenerating(true);
    try {
      const res = await generateArtifactsOnDemand({
        title: customTitle.trim() || "AegisAI Executive Deliverable",
        markdown: customMarkdown,
        formats: ["pdf", "docx", "pptx", "xlsx", "csv", "html", "md"],
      });
      if (res.success) {
        setCustomTitle("");
        setCustomMarkdown("");
        await refreshArtifacts();
      }
    } catch (err) {
      console.error("Deliverable generation error:", err);
    } finally {
      setCustomGenerating(false);
    }
  };

  const tabularDocs = docs.filter(isTabularDocument);

  const renderSection = () => {
    switch (section) {
      case "home":
        return (
          <div className="space-y-8 animate-fade-in-up">
            {/* Hero Workspace Banner */}
            <div className="relative overflow-hidden rounded-3xl bg-white border border-[#EAE7EF] shadow-sm p-7 md:p-10">
              <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-[#FFF9CC] opacity-80 blur-3xl pointer-events-none" />
              <div className="absolute -left-16 -bottom-16 h-64 w-64 rounded-full bg-[#F0EAFB] opacity-90 blur-3xl pointer-events-none" />

              <div className="relative z-10 flex flex-col gap-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="badge badge-purple px-3 py-1 font-semibold text-xs flex items-center gap-1.5 shadow-sm">
                    <Sparkles className="w-3.5 h-3.5 text-[#F3E66B]" /> Autonomous Multi-Agent AI Orchestration System — 8 Agents for Knowledge Retrieval
                  </span>
                  <span className="badge badge-lemon">
                    Docling + BGE-M3 + Qdrant
                  </span>
                  <span className="badge badge-green">
                    Max 500 MB / File · Permanent Storage
                  </span>
                </div>

                <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-[#272436] leading-tight max-w-4xl">
                  Autonomous Multi-Agent AI Orchestration System
                  <span className="block text-xl md:text-3xl font-bold text-[#7454B8] mt-2">
                    8 Agents for Knowledge Retrieval, Deep Research &amp; Verified Execution
                  </span>
                </h1>

                <p className="max-w-3xl text-sm md:text-base text-[#777383] leading-relaxed">
                  AegisAI is an autonomous multi-agent AI orchestration system featuring 8 specialized agents for knowledge retrieval, document intelligence, reasoning, and quality-controlled reporting. Upload scanned or native PDFs, spreadsheets, presentations, images, or archives to decompose complex queries, execute hybrid Qdrant vector retrieval, verify factual grounding, and generate ready-to-share PDF, DOCX, PPTX, XLSX, CSV, HTML, and Markdown artifacts.
                </p>

                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    className="btn-primary inline-flex items-center gap-2"
                    onClick={() => setSection("chat")}
                  >
                    Open Conversational Swarm <ChevronRight className="w-4 h-4" />
                  </button>
                  <button
                    className="btn-lemon inline-flex items-center gap-2"
                    onClick={() => setSection("documents")}
                  >
                    <UploadCloud className="w-4 h-4" /> Upload Corpus &amp; Datasets
                  </button>
                  <button
                    className="btn-secondary inline-flex items-center gap-2"
                    onClick={() => setSection("analytics")}
                  >
                    <FileSpreadsheet className="w-4 h-4 text-[#7454B8]" /> Analytics Lab
                  </button>
                </div>
              </div>
            </div>

            {/* Live Platform Telemetry Strip */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="glass-panel rounded-2xl p-4">
                <div className="text-xs font-semibold text-[#777383]">Active Agents</div>
                <div className="text-2xl font-extrabold text-[#4F3685] mt-1">8 Specialists</div>
                <div className="text-[11px] text-[#777383] mt-0.5">Parallel High-Speed Swarm</div>
              </div>
              <div className="glass-panel rounded-2xl p-4">
                <div className="text-xs font-semibold text-[#777383]">Indexed Documents</div>
                <div className="text-2xl font-extrabold text-[#272436] mt-1">{docs.length}</div>
                <div className="text-[11px] text-[#777383] mt-0.5">
                  Max 500 MB / file · Saved until deleted
                </div>
              </div>
              <div className="glass-panel rounded-2xl p-4">
                <div className="text-xs font-semibold text-[#777383]">Completed Executions</div>
                <div className="text-2xl font-extrabold text-[#272436] mt-1">{runs.length}</div>
                <div className="text-[11px] text-[#777383] mt-0.5">
                  Live Ragas &amp; DeepEval scored
                </div>
              </div>
              <div className="glass-panel rounded-2xl p-4">
                <div className="text-xs font-semibold text-[#777383]">System Health</div>
                <div
                  className={`text-2xl font-extrabold mt-1 ${
                    health?.status === "ok" ? "text-emerald-700" : "text-amber-700"
                  }`}
                >
                  {health?.status === "ok" ? "Operational" : "Connecting..."}
                </div>
                <div className="text-[11px] text-[#777383] mt-0.5">
                  {health?.models?.primary_llm || "gemini-2.5-flash"}
                </div>
              </div>
            </div>

            {/* Quick-Launch Prompts */}
            <div className="glass-panel rounded-3xl p-6">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h2 className="text-lg font-bold text-[#272436]">
                    One-Click Executive Objectives
                  </h2>
                  <p className="text-xs text-[#777383]">
                    Select a structured objective to prefill the 8-agent execution workspace.
                  </p>
                </div>
                <span className="badge badge-lemon">Fast Launch</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {QUICK_PROMPTS.map((promptText, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setTask(promptText);
                      setSection("chat");
                    }}
                    className="text-left p-4 rounded-2xl bg-[#FAFAF4] hover:bg-[#F0EAFB] border border-[#EAE7EF] hover:border-[#D8CCF2] transition-all group"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="badge badge-purple">Template {idx + 1}</span>
                      <ChevronRight className="w-4 h-4 text-[#777383] group-hover:text-[#4F3685]" />
                    </div>
                    <p className="text-xs text-[#272436] font-medium leading-relaxed">
                      {promptText}
                    </p>
                  </button>
                ))}
              </div>
            </div>

            {/* Resumable Background Tasks Queue */}
            {bgTasks.length > 0 && (
              <div className="glass-panel rounded-3xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-bold text-[#272436]">
                      Resumable Long-Running Task Queue
                    </h2>
                    <p className="text-xs text-[#777383]">
                      Asynchronous executions managed by TaskExecutionManager with live progress, cancel, and resume controls.
                    </p>
                  </div>
                  <button
                    onClick={fetchMeta}
                    className="btn-secondary inline-flex items-center gap-1.5 text-xs py-1.5 px-3"
                  >
                    <RefreshCw className="w-3.5 h-3.5" /> Refresh
                  </button>
                </div>

                <div className="space-y-2.5">
                  {bgTasks.slice(0, 6).map((item) => {
                    const isRunningTask = [
                      "PENDING",
                      "PLANNING",
                      "RUNNING",
                      "WAITING_FOR_TOOL",
                      "VALIDATING",
                      "REVISING",
                      "GENERATING_ARTIFACT",
                    ].includes(item.state);

                    return (
                      <div
                        key={item.execution_id}
                        className="p-3.5 rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] flex flex-col md:flex-row md:items-center justify-between gap-3"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2 mb-1">
                            <span className="font-mono text-[11px] font-bold text-[#4F3685]">
                              {item.execution_id}
                            </span>
                            <span
                              className={`badge ${
                                item.state === "COMPLETED"
                                  ? "badge-green"
                                  : item.state === "FAILED" || item.state === "CANCELLED"
                                  ? "badge-rose"
                                  : "badge-lemon"
                              }`}
                            >
                              {item.state}
                            </span>
                            <span className="text-[11px] text-[#777383]">
                              {Math.round(item.progress_pct || 0)}%
                              {item.current_agent ? ` · ${item.current_agent}` : ""}
                            </span>
                          </div>
                          <p className="text-xs font-semibold text-[#272436] truncate">
                            {item.user_query}
                          </p>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          {isRunningTask && (
                            <button
                              onClick={async () => {
                                await cancelBackgroundTask(item.execution_id);
                                fetchMeta();
                              }}
                              className="btn-secondary text-xs py-1.5 px-3 text-rose-700"
                            >
                              <XCircle className="w-3.5 h-3.5 inline mr-1" /> Cancel
                            </button>
                          )}
                          {(item.state === "FAILED" || item.state === "CANCELLED") && (
                            <button
                              onClick={async () => {
                                await resumeBackgroundTask(item.execution_id);
                                fetchMeta();
                              }}
                              className="btn-lemon text-xs py-1.5 px-3"
                            >
                              <RotateCcw className="w-3.5 h-3.5 inline mr-1" /> Resume
                            </button>
                          )}
                          {item.state === "COMPLETED" && (
                            <button
                              onClick={() => handleLoadTaskIntoWorkspace(item)}
                              className="btn-primary text-xs py-1.5 px-3"
                            >
                              Open Report
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* 8-Agent Architecture Grid */}
            <div className="glass-panel rounded-3xl p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-bold text-[#272436]">
                    Autonomous Multi-Agent AI Orchestration System — 8 Agents for Knowledge Retrieval
                  </h2>
                  <p className="text-xs text-[#777383]">
                    Coordinated via LangGraph conditional edges with Critic quality gates and verified artifact generation.
                  </p>
                </div>
                <span className="badge badge-purple">LangGraph StateGraph</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                {EIGHT_AGENT_ROSTER.map((ag) => (
                  <div
                    key={ag.step}
                    className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4 hover:border-[#D8CCF2] transition-all"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`badge ${ag.accent}`}>Agent {ag.step}</span>
                    </div>
                    <h3 className="text-sm font-bold text-[#272436]">{ag.name}</h3>
                    <p className="mt-1 text-xs text-[#777383] leading-relaxed">
                      {ag.role}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case "chat":
        return (
          <div className="space-y-4 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-4 md:p-6 h-[calc(100vh-120px)] min-h-[640px]">
              <Chat
                onDocumentUploaded={fetchMeta}
                onMetricsUpdate={(metrics, _illustrations, execId) => {
                  setLiveMetrics(metrics);
                  if (execId) setLastExecutionId(execId);
                }}
                onStateChange={(st, ag) => {
                  setTaskState(st);
                  setActiveAgent(ag);
                }}
              />
            </div>
          </div>
        );

      case "documents":
        return (
          <div className="space-y-5 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                <div>
                  <h2 className="text-2xl font-extrabold text-[#272436]">
                    Document Intelligence, URL &amp; Repository Ingestion
                  </h2>
                  <p className="text-xs text-[#777383]">
                    Upload files up to 500 MB each (up to 100 files / 1 GB per batch), ingest web URLs, or clone Git repositories. Indexed files remain saved permanently until you delete them.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="badge badge-green">Max 500 MB / File</span>
                  <span className="badge badge-lemon">{docs.length} Indexed</span>
                </div>
              </div>
              <FileUpload onUploaded={fetchMeta} />
            </div>

            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <DocumentList
                refreshKey={docsRefreshCounter}
                onUploaded={fetchMeta}
                onRunPipeline={(doc) => {
                  setSelectedDocId(doc.id);
                  if (!task.trim()) {
                    setTask(
                      `Analyze "${doc.filename}" thoroughly, summarize key findings, verify claims, and generate an executive report.`
                    );
                  }
                  setSection("chat");
                }}
                onProfileDataset={(doc) => {
                  setAnalyticsDocId(doc.id);
                  setSection("analytics");
                  handleRunProfile(doc.id);
                }}
              />
            </div>

            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <h3 className="text-base font-bold text-[#272436] mb-3">
                Supported Formats &amp; File Size Limits (Max 500 MB per file · 1 GB per batch)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-[#EAE7EF] text-[#777383]">
                      <th className="pb-2.5 font-bold">Category</th>
                      <th className="pb-2.5 font-bold">Supported Formats</th>
                      <th className="pb-2.5 font-bold">File Size Limit &amp; Storage Policy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#EAE7EF]">
                    {SUPPORTED_FORMATS_MATRIX.map((row) => (
                      <tr key={row.type}>
                        <td className="py-2.5">
                          <span className={`badge ${row.badge}`}>{row.type}</span>
                        </td>
                        <td className="py-2.5 font-mono text-[#4F3685]">
                          {row.extensions}
                        </td>
                        <td className="py-2.5 text-[#272436]">{row.limit}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        );

      case "analytics": {
        const cleanDatasetFilename = formatCleanFilename(profileResult?.filename || "");
        return (
          <div className="space-y-6 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="badge badge-purple text-[11px] font-semibold">Autonomous Analytics Hub</span>
                    <span className="badge badge-lemon text-[11px]">CSV &amp; Tabular Intelligence</span>
                  </div>
                  <h2 className="text-2xl font-extrabold text-[#272436]">
                    Dataset Analytics &amp; Visual Intelligence
                  </h2>
                  <p className="text-xs text-[#777383] mt-0.5">
                    Deterministic statistical profiling, outlier detection, correlations, and executive visual intelligence with full structured explanations.
                  </p>
                </div>
                <span className="badge badge-lemon">Auto-Statistics &amp; Charts · Max 500 MB</span>
              </div>

              {tabularDocs.length === 0 ? (
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-6 text-center">
                  <FileSpreadsheet className="w-8 h-8 text-[#7454B8] mx-auto mb-2" />
                  <p className="text-sm font-bold text-[#272436]">
                    No Structured Datasets Uploaded Yet
                  </p>
                  <p className="text-xs text-[#777383] mt-1 mb-4">
                    Upload a CSV, XLSX, XLS, or JSON file in Documents &amp; OCR to unlock automated statistical profiling and chart generation.
                  </p>
                  <button
                    onClick={() => setSection("documents")}
                    className="btn-primary text-xs"
                  >
                    Upload Dataset
                  </button>
                </div>
              ) : (
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <select
                    value={analyticsDocId}
                    onChange={(e) => setAnalyticsDocId(e.target.value)}
                    className="input-field flex-1"
                  >
                    <option value="">Select an indexed dataset...</option>
                    {tabularDocs.map((d) => (
                      <option key={d.id} value={d.id}>
                        {formatCleanFilename(d.filename)} ({formatBytes(d.file_size_bytes)})
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={() => handleRunProfile()}
                    disabled={!analyticsDocId || profiling}
                    className="btn-primary inline-flex items-center justify-center gap-2 whitespace-nowrap"
                  >
                    {profiling ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Play className="w-4 h-4" />
                    )}
                    Run Statistical Profile &amp; Charts
                  </button>
                </div>
              )}

              {profileError && (
                <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-800">
                  {profileError}
                </div>
              )}
            </div>

            {profileResult && (
              <div className="space-y-6">
                {/* Summary KPI Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="glass-panel rounded-2xl p-4">
                    <div className="text-xs text-[#777383]">Dataset File</div>
                    <div className="text-base font-extrabold text-[#272436] truncate mt-1" title={cleanDatasetFilename}>
                      {cleanDatasetFilename}
                    </div>
                  </div>
                  <div className="glass-panel rounded-2xl p-4">
                    <div className="text-xs text-[#777383]">Total Records</div>
                    <div className="text-2xl font-extrabold text-[#4F3685] mt-1">
                      {profileResult.profile?.row_count?.toLocaleString() ?? "—"}
                    </div>
                  </div>
                  <div className="glass-panel rounded-2xl p-4">
                    <div className="text-xs text-[#777383]">Total Features</div>
                    <div className="text-2xl font-extrabold text-[#272436] mt-1">
                      {profileResult.profile?.column_count ?? "—"}
                    </div>
                  </div>
                  <div className="glass-panel rounded-2xl p-4">
                    <div className="text-xs text-[#777383]">Continuous Metrics</div>
                    <div className="text-2xl font-extrabold text-emerald-700 mt-1">
                      {profileResult.profile?.numeric_columns?.length ?? 0}
                    </div>
                  </div>
                </div>

                {/* Generated Charts with Headings, Labels, and Explanations */}
                {profileResult.charts && profileResult.charts.length > 0 && (
                  <div className="glass-panel rounded-3xl p-6">
                    <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
                      <div>
                        <h3 className="text-xl font-bold text-[#272436]">
                          Dataset Visualizations &amp; Interpretations ({profileResult.charts.length})
                        </h3>
                        <p className="text-xs text-[#777383] mt-0.5">
                          Visual profiles with statistical explanations and strategic takeaways derived from {cleanDatasetFilename}.
                        </p>
                      </div>
                      <span className="badge badge-purple text-xs">
                        {cleanDatasetFilename}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {profileResult.charts.map((chart, idx) => {
                        const imgSrc = chart.image_data
                          ? chart.image_data.startsWith("data:")
                            ? chart.image_data
                            : `data:image/png;base64,${chart.image_data}`
                          : "";
                        const meta = chart.metadata || {};
                        const chartHeading = meta.heading || formatCleanFilename(chart.title);
                        const chartTypeLabel = formatChartTypeBadge(chart.chart_type);
                        const keyMetric = meta.key_metric;
                        const explanation = meta.explanation;
                        const takeaway = meta.takeaway;
                        const labels = chart.labels || [];

                        return (
                          <div
                            key={idx}
                            className="flex flex-col rounded-2xl bg-white border border-[#EAE7EF] shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden"
                          >
                            {/* Card Header */}
                            <div className="px-5 pt-4 pb-3 border-b border-[#EAE7EF] bg-[#FAFAF4]">
                              <div className="flex items-start justify-between gap-2">
                                <div>
                                  <div className="flex items-center gap-1.5 mb-1">
                                    <span className="badge badge-purple text-[10px] font-semibold">
                                      {chartTypeLabel}
                                    </span>
                                    <span className="badge badge-lemon text-[10px] font-mono">
                                      {cleanDatasetFilename}
                                    </span>
                                  </div>
                                  <h4 className="text-sm font-bold text-[#272436] leading-snug">
                                    {chartHeading}
                                  </h4>
                                </div>
                              </div>
                            </div>

                            {/* Chart Image */}
                            <div className="p-4 bg-[#FAFAF4]/40 flex items-center justify-center">
                              {imgSrc ? (
                                <img
                                  src={imgSrc}
                                  alt={chartHeading}
                                  className="w-full h-auto rounded-xl border border-[#EAE7EF] bg-white shadow-xs object-contain"
                                />
                              ) : (
                                <div className="h-48 w-full flex items-center justify-center text-xs text-[#777383] bg-white rounded-xl border border-dashed border-[#EAE7EF]">
                                  No visual preview available
                                </div>
                              )}
                            </div>

                            {/* Labels & Features Pill Strip */}
                            {labels.length > 0 && (
                              <div className="px-5 py-2.5 border-t border-[#EAE7EF] bg-white">
                                <div className="text-[11px] font-semibold text-[#777383] mb-1.5 flex items-center gap-1">
                                  <Tag className="w-3 h-3 text-[#7454B8]" />
                                  <span>Plotted Labels &amp; Categories ({labels.length}):</span>
                                </div>
                                <div className="flex flex-wrap gap-1 max-h-16 overflow-y-auto pr-1">
                                  {labels.map((lbl, lIdx) => (
                                    <span
                                      key={lIdx}
                                      className="inline-flex items-center text-[10px] font-medium bg-[#FAFAF4] text-[#4F3685] border border-[#EAE7EF] rounded-md px-2 py-0.5"
                                    >
                                      {lbl}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Explanation and Statistical Insights Box */}
                            <div className="p-5 border-t border-[#EAE7EF] bg-[#FAFAF4] space-y-3 flex-1 flex flex-col justify-between">
                              <div className="space-y-2.5">
                                {/* Visual Explanation */}
                                {explanation && (
                                  <div className="space-y-1">
                                    <div className="text-[11px] font-bold uppercase tracking-wider text-[#4F3685] flex items-center gap-1">
                                      <Info className="w-3.5 h-3.5 text-[#7454B8]" />
                                      <span>Visual Explanation</span>
                                    </div>
                                    <p className="text-xs text-[#272436] leading-relaxed bg-white/80 rounded-xl p-3 border border-[#EAE7EF]/80">
                                      {explanation}
                                    </p>
                                  </div>
                                )}

                                {/* Key Statistical Metric */}
                                {keyMetric && (
                                  <div className="rounded-xl bg-[#FFF9CC] border border-[#E5C832]/40 p-3">
                                    <div className="text-[10px] font-bold uppercase tracking-wider text-[#7454B8] flex items-center gap-1 mb-1">
                                      <Sparkles className="w-3 h-3 text-[#7454B8]" />
                                      <span>Key Statistical Metric</span>
                                    </div>
                                    <div className="text-xs font-semibold text-[#272436]">
                                      {keyMetric}
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Actionable Strategic Takeaway */}
                              {takeaway && (
                                <div className="rounded-xl bg-purple-50/60 border border-[#7454B8]/20 p-3 mt-1">
                                  <div className="text-[10px] font-bold uppercase tracking-wider text-[#4F3685] flex items-center gap-1 mb-1">
                                    <Lightbulb className="w-3 h-3 text-[#7454B8]" />
                                    <span>Actionable Takeaway</span>
                                  </div>
                                  <p className="text-xs text-[#272436] leading-relaxed">
                                    {takeaway}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Structured Summary Tables */}
                {profileResult.tables && profileResult.tables.length > 0 && (
                  <div className="glass-panel rounded-3xl p-6 space-y-5">
                    <h3 className="text-lg font-bold text-[#272436]">
                      Descriptive Statistics &amp; Data Preview — {cleanDatasetFilename}
                    </h3>
                    {profileResult.tables.map((tbl, idx) => (
                      <div key={idx} className="space-y-2">
                        <h4 className="text-xs font-bold uppercase tracking-wider text-[#4F3685]">
                          {formatCleanFilename(tbl.title)}
                        </h4>
                        <div className="overflow-x-auto rounded-2xl border border-[#EAE7EF]">
                          <table className="w-full text-left text-xs">
                            <thead className="bg-[#FAFAF4] border-b border-[#EAE7EF]">
                              <tr>
                                {tbl.headers.map((h, hIdx) => (
                                  <th
                                    key={hIdx}
                                    className="px-3 py-2 font-bold text-[#272436]"
                                  >
                                    {h}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#EAE7EF] bg-white">
                              {tbl.rows.slice(0, 15).map((r, rIdx) => (
                                <tr key={rIdx}>
                                  {r.map((cell, cIdx) => (
                                    <td
                                      key={cIdx}
                                      className="px-3 py-2 text-[#272436] whitespace-nowrap"
                                    >
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      }

      case "deliverables":
        return (
          <div className="space-y-5 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-2xl font-extrabold text-[#272436]">
                    Deliverables Hub (7 Verified Formats)
                  </h2>
                  <p className="text-xs text-[#777383]">
                    Download generated PDF reports, DOCX documents, PPTX executive slide decks, XLSX multi-sheet workbooks, CSV exports, HTML reports, and Markdown files.
                  </p>
                </div>
                <button
                  onClick={refreshArtifacts}
                  className="btn-secondary inline-flex items-center gap-1.5 text-xs"
                >
                  <RefreshCw
                    className={`w-3.5 h-3.5 ${artifactsLoading ? "animate-spin" : ""}`}
                  />
                  Refresh Hub
                </button>
              </div>

              {allArtifacts.length === 0 ? (
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-8 text-center">
                  <Download className="w-8 h-8 text-[#7454B8] mx-auto mb-2" />
                  <p className="text-sm font-bold text-[#272436]">
                    No Deliverables Generated Yet
                  </p>
                  <p className="text-xs text-[#777383] mt-1">
                    Run an 8-Agent Execution with selected formats or create an instant 7-format bundle below.
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
                  {allArtifacts.map((art) => (
                    <div
                      key={`${art.execution_id}-${art.filename}`}
                      className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4 flex flex-col justify-between gap-3 hover:border-[#D8CCF2] transition-all"
                    >
                      <div>
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <span className="badge badge-purple uppercase">
                            {art.format}
                          </span>
                          <span className="text-[11px] font-semibold text-[#777383]">
                            {formatBytes(art.size_bytes)}
                          </span>
                        </div>
                        <p className="text-xs font-bold text-[#272436] break-all">
                          {art.filename}
                        </p>
                        <p className="text-[10px] font-mono text-[#777383] mt-1">
                          Exec: {art.execution_id}
                        </p>
                      </div>
                      <a
                        href={getArtifactDownloadUrl(art)}
                        download={art.filename}
                        target="_blank"
                        rel="noreferrer"
                        className="btn-primary text-xs py-2 inline-flex items-center justify-center gap-1.5 w-full"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download {art.format.toUpperCase()}
                      </a>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Instant On-Demand 7-Format Deliverable Builder */}
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <h3 className="text-base font-bold text-[#272436] mb-1">
                Instant 7-Format Deliverable Packager
              </h3>
              <p className="text-xs text-[#777383] mb-4">
                Convert any Markdown brief or executive memo directly into verified PDF, DOCX, PPTX, XLSX, CSV, HTML, and Markdown artifacts.
              </p>
              <div className="space-y-3">
                <input
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Deliverable Title (e.g., Q4 Strategic Intelligence Brief)"
                  className="input-field"
                />
                <textarea
                  value={customMarkdown}
                  onChange={(e) => setCustomMarkdown(e.target.value)}
                  rows={5}
                  placeholder="# Executive Summary&#10;&#10;Enter structured Markdown sections, bullet points, or findings to compile into all 7 formats..."
                  className="input-field"
                />
                <button
                  onClick={handleCustomArtifactGenerate}
                  disabled={!customMarkdown.trim() || customGenerating}
                  className="btn-lemon inline-flex items-center gap-2 text-xs"
                >
                  {customGenerating ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                  Generate All 7 Deliverable Formats
                </button>
              </div>
            </div>
          </div>
        );

      case "history":
        return (
          <div className="space-y-4 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-extrabold text-[#272436]">
                    Execution History
                  </h2>
                  <p className="text-xs text-[#777383]">
                    Persistent archive of completed 8-agent runs, reports, and live evaluation scores.
                  </p>
                </div>
                <span className="badge badge-purple">{runs.length} Runs</span>
              </div>
            </div>
            <div className="glass-panel rounded-3xl p-5 md:p-7">
              <HistoryList />
            </div>
          </div>
        );

      case "docs":
        return (
          <div className="space-y-5 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-6 md:p-8">
              <h2 className="text-2xl font-extrabold text-[#272436] mb-2">
                Architecture &amp; Technical Reference
              </h2>
              <p className="text-xs text-[#777383] mb-6">
                Verified configuration of Document Intelligence, Chunking, Multi-Model Embeddings, Qdrant Hybrid Retrieval, OpenLM Gateway, and FastAPI Endpoints.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-5">
                  <h3 className="text-sm font-bold text-[#272436] mb-2">
                    Chunking &amp; Provenance Engine
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[#777383]">
                    <li>
                      <strong className="text-[#272436]">Default Chunk Size:</strong>{" "}
                      {health?.chunking_defaults?.chunk_size_words ?? 900} words
                    </li>
                    <li>
                      <strong className="text-[#272436]">Dynamic Overlap:</strong>{" "}
                      {health?.chunking_defaults?.chunk_overlap_pct ?? 10}% (
                      {health?.chunking_defaults?.chunk_overlap_words ?? 90} words)
                    </li>
                    <li>
                      <strong className="text-[#272436]">7 Supported Strategies:</strong>{" "}
                      word, adaptive, document_aware, section_aware, heading_aware, table_aware, page_aware
                    </li>
                    <li>
                      <strong className="text-[#272436]">Metadata:</strong> page_number, section_title, parser_used, ocr_used, word_count, overlap_words
                    </li>
                  </ul>
                </div>

                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-5">
                  <h3 className="text-sm font-bold text-[#272436] mb-2">
                    Embeddings, Qdrant &amp; OpenLM
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[#777383]">
                    <li>
                      <strong className="text-[#272436]">Default Embedder:</strong>{" "}
                      BAAI/bge-m3 (1024 dimensions, normalized)
                    </li>
                    <li>
                      <strong className="text-[#272436]">Fast Secondary Embedder:</strong>{" "}
                      sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
                    </li>
                    <li>
                      <strong className="text-[#272436]">Vector Retrieval:</strong>{" "}
                      Qdrant Hybrid Dense + Lexical BM25 Reciprocal Rank Fusion (RRF)
                    </li>
                    <li>
                      <strong className="text-[#272436]">OpenLM Provider Layer:</strong>{" "}
                      Modular OpenAI-compatible LLM gateway / CausalLM adapter (Gemini 2.5 Flash primary)
                    </li>
                  </ul>
                </div>

                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-5">
                  <h3 className="text-sm font-bold text-[#272436] mb-2">
                    Core REST &amp; SSE Endpoints
                  </h3>
                  <ul className="space-y-1.5 text-xs font-mono text-[#4F3685]">
                    <li>POST /api/documents/upload &amp; /batch-upload</li>
                    <li>POST /api/pipeline/run &amp; GET /api/pipeline/stream</li>
                    <li>POST /api/chat &amp; POST /api/chat/stream</li>
                    <li>POST /api/tasks · GET /api/tasks/&#123;id&#125; · /cancel · /resume</li>
                    <li>POST /api/analytics/profile/&#123;document_id&#125;</li>
                    <li>GET /api/artifacts/&#123;execution_id&#125;/&#123;filename&#125;</li>
                  </ul>
                </div>

                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-5">
                  <h3 className="text-sm font-bold text-[#272436] mb-2">
                    Live Sidebar Evaluation Suite
                  </h3>
                  <ul className="space-y-1.5 text-xs text-[#777383]">
                    <li>1. Hallucination Rate (inverse groundedness check)</li>
                    <li>2. Accuracy &amp; 3. Correctness</li>
                    <li>4. Context Precision &amp; 5. Context Recall</li>
                    <li>6. Task Success Rate &amp; 7. Tool Call Accuracy</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        );

      case "about":
        return (
          <div className="space-y-5 animate-fade-in-up">
            <div className="glass-panel rounded-3xl p-6 md:p-8">
              <div className="flex flex-wrap items-center gap-2 mb-3">
                <span className="badge badge-lemon">AegisAI AgentSwarm</span>
                <span className="badge badge-purple">Production Release</span>
              </div>
              <h2 className="text-2xl font-extrabold text-[#272436] mb-3">
                About the AegisAI Autonomous Execution Platform
              </h2>
              <p className="text-sm text-[#777383] leading-relaxed">
                AegisAI is an enterprise multi-agent AI execution workspace engineered around LangGraph, Gemini 2.5 Flash, SentenceTransformers (BAAI/bge-m3 &amp; MiniLM), Qdrant vector search, real Python data science libraries, and multi-format deliverable synthesis.
              </p>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4">
                  <div className="font-bold text-[#272436] mb-1">Primary LLM &amp; OpenLM</div>
                  <p className="text-[#777383]">
                    Primary: {health?.models?.primary_llm || "gemini-2.5-flash"} via google-genai SDK. OpenLM configured as optional LLM gateway ({health?.openlm?.configured ? "Configured" : "Standby"}).
                  </p>
                </div>
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4">
                  <div className="font-bold text-[#272436] mb-1">Vector Store Status</div>
                  <p className="text-[#777383]">
                    Qdrant collection active ({health?.qdrant?.status || "ready"}) with strict vector dimension validation and zero Neo4j/graph dependencies.
                  </p>
                </div>
                <div className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4">
                  <div className="font-bold text-[#272436] mb-1">Verified Deliverables</div>
                  <p className="text-[#777383]">
                    Every generated PDF, DOCX, PPTX, XLSX, CSV, HTML, and Markdown file is validated non-empty on disk and linked to its execution_id.
                  </p>
                </div>
              </div>
            </div>

            <div className="glass-panel rounded-3xl p-6 md:p-8">
              <h3 className="text-lg font-bold text-[#272436] mb-4">
                8 Specialized Agents
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                {EIGHT_AGENT_ROSTER.map((ag) => (
                  <div
                    key={ag.step}
                    className="rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] p-4"
                  >
                    <span className={`badge ${ag.accent} mb-2`}>{ag.step}</span>
                    <div className="text-sm font-bold text-[#272436]">{ag.name}</div>
                    <p className="text-xs text-[#777383] mt-1 leading-relaxed">
                      {ag.role}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-panel rounded-3xl p-6 md:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="text-lg font-bold text-[#272436]">
                    Contact Developer
                  </h3>
                  <p className="text-xs text-[#777383] mt-1">
                    Connect via LinkedIn or Gmail for feedback, collaboration, and deployment inquiries.
                  </p>
                </div>
                <ContactDeveloper
                  activeKey={activeContact}
                  onSelect={setActiveContact}
                  variant="card"
                />
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const activeTabLabel =
    tabs.find((t) => t.id === section)?.label || "Workspace";

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-[#FAFAF4] text-[#272436]">
      <Sidebar
        activeTab={section}
        onTabChange={(id) => setSection(id)}
        responseMetrics={liveMetrics}
        taskState={taskState}
        activeAgent={activeAgent}
        lastExecutionId={lastExecutionId}
        recentUploads={docs}
        healthStatus={health?.status || "ok"}
        activeContact={activeContact}
        onContactClick={setActiveContact}
        onSelectDocument={(doc) => {
          setSelectedDocId(doc.id);
          setSection("documents");
        }}
      />

      <main ref={mainRef} className="flex-1 min-w-0 min-h-screen">
        <header className="hidden lg:flex items-center justify-between px-8 py-3 bg-white/95 backdrop-blur-sm border-b border-[#EAE7EF] sticky top-0 z-10">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="text-sm font-bold text-[#272436] truncate">
              {activeTabLabel}
            </span>
            <span className="badge badge-purple text-[11px]">
              8-Agent Orchestration
            </span>
          </div>
          <div className="flex items-center gap-2.5 shrink-0">
            <span className="badge badge-lemon text-[11px]">
              <Sparkles className="w-3 h-3" /> v1.0
            </span>
            <NavbarSocialIcons
              activeKey={activeContact}
              onSelect={setActiveContact}
            />
          </div>
        </header>

        <div className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8 md:py-8">
          {renderSection()}
        </div>
      </main>
    </div>
  );
}
