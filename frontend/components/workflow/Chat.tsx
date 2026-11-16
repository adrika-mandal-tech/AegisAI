"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Loader2,
  Brain,
  FileText,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  Search,
  Lightbulb,
  X,
  Download,
  Zap,
  Activity,
  Layers,
  CheckCircle2,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  streamChat,
  getArtifactDownloadUrl,
  downloadReportPdf,
  type ChatMessage,
  type ChatSSEEvent,
  type Evidence,
  type ChartData,
  type ArtifactItem,
  type AgentWorkIllustration,
  type ResponseMetrics,
} from "@/lib/api";
import {
  ToolExecutionCheckbox,
  type ToolExecutionSettings,
} from "@/components/ui/ToolExecutionCheckbox";

interface ChatProps {
  onDocumentUploaded?: () => void;
  onMetricsUpdate?: (
    metrics: ResponseMetrics,
    illustrations?: AgentWorkIllustration[],
    executionId?: string
  ) => void;
  onStateChange?: (state: string, activeAgent: string) => void;
}

const defaultToolSettings: ToolExecutionSettings = {
  enableWebSearch: true,
  enableDocumentAnalysis: true,
  enableCodeExecution: true,
  enableExternalAPIs: false,
  maxToolCalls: 10,
  toolTimeout: 30,
};

const AGENT_STAGES = [
  { agent: "supervisor", label: "Supervisor" },
  { agent: "planner", label: "Planner" },
  { agent: "retriever", label: "Retriever" },
  { agent: "researcher", label: "Researcher" },
  { agent: "validator", label: "Validator" },
  { agent: "reasoner", label: "Reasoner & Analyst" },
  { agent: "critic", label: "Critic" },
  { agent: "report_generator", label: "Report & Deliverables" },
] as const;

type AgentStatus = "pending" | "running" | "done";

interface AgentProgress {
  agent: string;
  label: string;
  status: AgentStatus;
  output: string;
}

function cleanChatContent(raw: string): string {
  if (!raw) return "";
  let text = raw;

  // Clean decorative horizontal divider lines like ***, ___, ###, ===, *#*#*
  text = text.replace(/^[ \t]*[*#_=-]{3,}[ \t]*$/gm, "");

  // Clean trailing hashes from markdown headers (e.g. "## Heading ##" -> "## Heading")
  text = text.replace(/^(#{1,6}\s+.*?)\s+#+\s*$/gm, "$1");

  // Fix and normalize markdown tables
  const lines = text.split("\n");
  const cleanedLines: string[] = [];
  let inTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isPipeLine = /^\s*\|.*\|\s*$/.test(line);

    if (isPipeLine) {
      inTable = true;
      if (/^\s*\|[\s\-:]+(\|[\s\-:]+)+\|\s*$/.test(line)) {
        const segments = line.trim().replace(/^\||\|$/g, "").split("|");
        cleanedLines.push("| " + segments.map(() => "---").join(" | ") + " |");
      } else {
        cleanedLines.push(line);
      }
    } else {
      if (inTable && line.trim() === "") {
        inTable = false;
      }
      let cleanLine = line;
      if (!inTable) {
        // Outside of tables, replace scattered pipe characters with clean middle dots
        cleanLine = cleanLine.replace(/(\S)\s*\|\s*(\S)/g, "$1 • $2");
        cleanLine = cleanLine.replace(/^\s*\|\s*/, "").replace(/\s*\|\s*$/, "");
      }
      cleanedLines.push(cleanLine);
    }
  }

  text = cleanedLines.join("\n");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

export function Chat({
  onDocumentUploaded,
  onMetricsUpdate,
  onStateChange,
}: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPlan, setShowPlan] = useState<Record<string, any> | null>(null);
  const [showEvidence, setShowEvidence] = useState<Record<number, boolean>>({});
  const [currentResponseEvidence, setCurrentResponseEvidence] = useState<Evidence[]>([]);
  const [currentCitations, setCurrentCitations] = useState<string[]>([]);
  const [currentCharts, setCurrentCharts] = useState<ChartData[]>([]);
  const [currentArtifacts, setCurrentArtifacts] = useState<ArtifactItem[]>([]);
  const [currentIllustrations, setCurrentIllustrations] = useState<AgentWorkIllustration[]>([]);
  const [currentReportMarkdown, setCurrentReportMarkdown] = useState<string | null>(null);
  const [toolSettings, setToolSettings] = useState<ToolExecutionSettings>(defaultToolSettings);
  const [activeCommand, setActiveCommand] = useState<string | null>(null);
  const [sessionId] = useState<string>(() => `sess_${Math.random().toString(36).slice(2, 10)}`);

  const [agentProgress, setAgentProgress] = useState<AgentProgress[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showAgentPanel, setShowAgentPanel] = useState(true);
  const [showIllustrations, setShowIllustrations] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const initAgentProgress = (): AgentProgress[] =>
    AGENT_STAGES.map((s) => ({
      agent: s.agent,
      label: s.label,
      status: "pending" as AgentStatus,
      output: "",
    }));

  const submitPrompt = useCallback(
    (promptText: string) => {
      if (!promptText.trim() || isLoading) return;

      const userMessage: ChatMessage = {
        role: "user",
        content: promptText.trim(),
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setIsStreaming(true);
      setError(null);
      setShowAgentPanel(true);
      setAgentProgress(initAgentProgress());
      setInput("");
      onStateChange?.("RUNNING", "supervisor");

      controllerRef.current = streamChat(
        {
          message: promptText.trim(),
          history: messages,
          mode: "auto",
          context_documents: [],
          session_id: sessionId,
          tool_settings: toolSettings,
        },
        (event: ChatSSEEvent) => {
          if (event.type === "agent_start") {
            onStateChange?.("RUNNING", event.agent);
            setAgentProgress((prev) =>
              prev.map((a) =>
                a.agent === event.agent ? { ...a, status: "running" as AgentStatus } : a
              )
            );
          } else if (event.type === "agent_output") {
            onStateChange?.("RUNNING", event.agent);
            setAgentProgress((prev) =>
              prev.map((a) =>
                a.agent === event.agent
                  ? { ...a, status: "running" as AgentStatus, output: event.output || "" }
                  : a
              )
            );
          } else if (event.type === "agent_complete") {
            setAgentProgress((prev) =>
              prev.map((a) =>
                a.agent === event.agent ? { ...a, status: "done" as AgentStatus } : a
              )
            );
          } else if (event.type === "chat_complete") {
            const responseText = event.response || "Execution completed.";
            if (event.plan) setShowPlan(event.plan);
            if (event.evidence) setCurrentResponseEvidence(event.evidence);
            if (event.citations) setCurrentCitations(event.citations);
            if (event.charts) setCurrentCharts(event.charts);
            if (event.artifacts) setCurrentArtifacts(event.artifacts);
            if (event.agent_work_illustrations) {
              setCurrentIllustrations(event.agent_work_illustrations);
            }
            if (event.report_markdown) setCurrentReportMarkdown(event.report_markdown);
            if (event.response_metrics) {
              onMetricsUpdate?.(
                event.response_metrics,
                event.agent_work_illustrations,
                event.execution_id
              );
            }
            onStateChange?.("COMPLETED", "");

            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: responseText,
                timestamp: new Date().toISOString(),
              },
            ]);
            setIsLoading(false);
            setIsStreaming(false);
            setActiveCommand(null);
            onDocumentUploaded?.();
          }
        },
        (errorMsg) => {
          setError(errorMsg);
          onStateChange?.("FAILED", "");
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: `Execution encountered an error: ${errorMsg}`,
              timestamp: new Date().toISOString(),
            },
          ]);
          setIsLoading(false);
          setIsStreaming(false);
          setActiveCommand(null);
        },
        () => {
          setIsLoading(false);
          setIsStreaming(false);
          setActiveCommand(null);
        }
      );
    },
    [isLoading, messages, sessionId, toolSettings, onDocumentUploaded, onMetricsUpdate, onStateChange]
  );

  const handleSend = (e?: React.FormEvent) => {
    e?.preventDefault();
    submitPrompt(input);
  };

  const handleStop = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsLoading(false);
    setIsStreaming(false);
    setActiveCommand(null);
    onStateChange?.("CANCELLED", "");
  };

  const clearChat = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setMessages([]);
    setError(null);
    setShowPlan(null);
    setCurrentResponseEvidence([]);
    setCurrentCitations([]);
    setCurrentCharts([]);
    setCurrentArtifacts([]);
    setCurrentIllustrations([]);
    setCurrentReportMarkdown(null);
    setActiveCommand(null);
    setAgentProgress([]);
    setIsStreaming(false);
    onStateChange?.("IDLE", "");
  };

  const pipelineCommands = [
    {
      id: "explain-report",
      label: "Explain Documents",
      prompt:
        "Investigate all uploaded documents, retrieve grounded evidence, validate claims, reason across sections, and produce a comprehensive report with citations.",
    },
    {
      id: "generate-charts",
      label: "Dataset Charts",
      prompt:
        "Analyze uploaded structured datasets or quantitative document metrics, run statistical profiling, and generate visual charts and summary tables.",
    },
    {
      id: "export-deliverables",
      label: "Generate PDF, DOCX & PPTX",
      prompt:
        "Create an executive intelligence report with key findings, risk assessment, and export downloadable PDF, DOCX, PPTX, and XLSX deliverables.",
    },
    {
      id: "calculate-evals",
      label: "Audit & Validate",
      prompt:
        "Retrieve document evidence, verify every factual claim for contradictions or gaps, critique analytical rigor, and compute full evaluation metrics.",
    },
  ];

  const completedCount = agentProgress.filter((a) => a.status === "done").length;
  const progressPct =
    agentProgress.length > 0 ? (completedCount / agentProgress.length) * 100 : 0;

  return (
    <div className="flex flex-col h-full bg-white rounded-3xl border border-[#EAE7EF] shadow-soft overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-[#EAE7EF] bg-[#FAFAF4]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-[#F0EAFB] border border-[#D8CCF2] flex items-center justify-center">
            <Brain className="w-5 h-5 text-[#7454B8]" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-[#272436]">
              Conversational 8-Agent Workspace
            </h2>
            <p className="text-[11px] text-[#777383]">
              Multi-turn memory · Grounded Qdrant citations · Live deliverables
            </p>
          </div>
          {isStreaming && (
            <span className="badge badge-lemon animate-pulse">
              <Zap className="w-3 h-3" /> Streaming Swarm
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ToolExecutionCheckbox
            settings={toolSettings}
            onSettingsChange={setToolSettings}
          />
          {isStreaming && (
            <button
              onClick={handleStop}
              className="px-3 py-1.5 text-xs font-semibold rounded-xl bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100 transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            onClick={clearChat}
            className="p-2 rounded-xl text-[#777383] hover:text-[#272436] hover:bg-white border border-transparent hover:border-[#EAE7EF] transition-colors"
            title="Reset conversation"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Live 8-Agent Progress Bar */}
      {(isStreaming || agentProgress.some((a) => a.status !== "pending")) && (
        <div className="mx-4 mt-3 rounded-2xl border border-[#EAE7EF] bg-[#FAFAF4] overflow-hidden">
          <button
            onClick={() => setShowAgentPanel((p) => !p)}
            className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-[#F0EAFB]/40 transition-colors"
          >
            <div className="flex items-center gap-2.5">
              <Activity
                className={`w-4 h-4 ${
                  isStreaming ? "text-[#7454B8] animate-pulse" : "text-emerald-600"
                }`}
              />
              <span className="text-xs font-bold text-[#272436]">
                {isStreaming ? "Collaborating Across Agents..." : "8-Agent Turn Complete"}
              </span>
              <span className="badge badge-purple">
                {completedCount}/{agentProgress.length}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-28 h-1.5 bg-[#EAE7EF] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#7454B8] to-[#F3E66B] transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              {showAgentPanel ? (
                <ChevronUp className="w-4 h-4 text-[#777383]" />
              ) : (
                <ChevronDown className="w-4 h-4 text-[#777383]" />
              )}
            </div>
          </button>
          {showAgentPanel && (
            <div className="px-3 pb-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-1.5">
              {agentProgress.map((ag) => (
                <div
                  key={ag.agent}
                  className={`px-2.5 py-2 rounded-xl border text-xs flex items-center gap-2 ${
                    ag.status === "running"
                      ? "bg-[#FFF9CC] border-[#E6D54F] text-[#5C4900]"
                      : ag.status === "done"
                      ? "bg-white border-[#EAE7EF] text-[#272436]"
                      : "bg-white/60 border-[#EAE7EF] text-[#777383] opacity-60"
                  }`}
                >
                  {ag.status === "running" ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-[#7454B8] shrink-0" />
                  ) : ag.status === "done" ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-[#D8CCF2] shrink-0" />
                  )}
                  <span className="font-semibold truncate">{ag.label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="mx-4 mt-3 p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-2 text-rose-700 text-xs">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)}>
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto p-5 space-y-5">
        {messages.length === 0 && !isStreaming ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-8">
            <div className="w-16 h-16 rounded-2xl bg-[#FFF9CC] border border-[#E6D54F] flex items-center justify-center mb-4">
              <Brain className="w-8 h-8 text-[#4F3685]" />
            </div>
            <h3 className="text-lg font-extrabold text-[#272436] mb-1">
              Autonomous Multi-Agent AI Orchestration System
            </h3>
            <p className="text-xs text-[#777383] max-w-md mb-5 leading-relaxed">
              8 specialized agents for knowledge retrieval, deep research, dataset analytics, and verified deliverable generation. Ask complex questions, interrogate uploaded documents, or generate exportable reports.
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-2xl">
              {[
                "Summarize key findings and risks across uploaded documents",
                "Profile the uploaded dataset, detect anomalies, and plot trends",
                "Create a strategic AI architecture plan and export PPTX & PDF",
                "Compare methodologies and verify factual consistency",
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => submitPrompt(suggestion)}
                  className="px-3.5 py-2 text-xs font-medium bg-[#FAFAF4] border border-[#EAE7EF] rounded-xl text-[#272436] hover:bg-[#F0EAFB] hover:border-[#7454B8] hover:text-[#4F3685] transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const isLastAssistant =
              msg.role === "assistant" && idx === messages.length - 1;

            return (
              <div
                key={idx}
                className={`flex gap-3 ${
                  msg.role === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 border ${
                    msg.role === "user"
                      ? "bg-[#FFF9CC] border-[#E6D54F] text-[#5C4900]"
                      : "bg-[#F0EAFB] border-[#D8CCF2] text-[#4F3685]"
                  }`}
                >
                  {msg.role === "user" ? (
                    <MessageSquare className="w-4 h-4" />
                  ) : (
                    <Brain className="w-4 h-4" />
                  )}
                </div>

                <div className="max-w-[86%] space-y-3">
                  <div
                    className={`px-4 py-3 rounded-2xl border ${
                      msg.role === "user"
                        ? "bg-[#FFF9CC] border-[#E6D54F] text-[#272436] font-medium text-sm"
                        : "bg-[#FAFAF4] border-[#EAE7EF] text-[#272436]"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    ) : (
                      <div className="max-w-none text-sm">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            h1: ({ children }) => (
                              <h1 className="text-lg font-extrabold text-[#272436] mt-3 mb-2 border-b border-[#EAE7EF] pb-1.5">
                                {children}
                              </h1>
                            ),
                            h2: ({ children }) => (
                              <h2 className="text-base font-bold text-[#4F3685] mt-4 mb-2 flex items-center gap-2">
                                <span className="w-1.5 h-4 rounded-full bg-[#7454B8]" />
                                {children}
                              </h2>
                            ),
                            h3: ({ children }) => (
                              <h3 className="text-sm font-bold text-[#272436] mt-3 mb-1.5">
                                {children}
                              </h3>
                            ),
                            p: ({ children }) => (
                              <p className="text-sm text-[#272436] leading-relaxed mb-3">
                                {children}
                              </p>
                            ),
                            ul: ({ children }) => (
                              <ul className="space-y-1.5 text-sm mb-3 pl-1">
                                {children}
                              </ul>
                            ),
                            ol: ({ children }) => (
                              <ol className="list-decimal pl-5 space-y-1.5 text-sm mb-3 text-[#272436]">
                                {children}
                              </ol>
                            ),
                            li: ({ children }) => (
                              <li className="flex items-start gap-2 text-sm text-[#272436] leading-relaxed">
                                <span className="w-1.5 h-1.5 rounded-full bg-[#7454B8] mt-2 shrink-0" />
                                <span className="flex-1">{children}</span>
                              </li>
                            ),
                            table: ({ children }) => (
                              <div className="my-4 overflow-x-auto rounded-2xl border border-[#D8CCF2] shadow-sm bg-white">
                                <table className="w-full text-left text-xs border-collapse">
                                  {children}
                                </table>
                              </div>
                            ),
                            thead: ({ children }) => (
                              <thead className="bg-gradient-to-r from-[#F0EAFB] to-[#FFF9CC] border-b border-[#D8CCF2]">
                                {children}
                              </thead>
                            ),
                            tbody: ({ children }) => (
                              <tbody className="divide-y divide-[#EAE7EF] bg-white">
                                {children}
                              </tbody>
                            ),
                            tr: ({ children }) => (
                              <tr className="hover:bg-[#FAFAF4] transition-colors">
                                {children}
                              </tr>
                            ),
                            th: ({ children }) => (
                              <th className="px-3.5 py-2.5 font-extrabold text-[#4F3685] text-xs tracking-wide">
                                {children}
                              </th>
                            ),
                            td: ({ children }) => (
                              <td className="px-3.5 py-2.5 text-[#272436] text-xs font-medium leading-relaxed align-top">
                                {children}
                              </td>
                            ),
                            strong: ({ children }) => (
                              <strong className="font-bold text-[#272436]">
                                {children}
                              </strong>
                            ),
                            em: ({ children }) => (
                              <em className="italic text-[#4F3685]">
                                {children}
                              </em>
                            ),
                            code: ({ children }) => (
                              <code className="px-1.5 py-0.5 rounded-md bg-[#F0EAFB] text-[#4F3685] font-mono text-xs border border-[#D8CCF2]/50">
                                {children}
                              </code>
                            ),
                            blockquote: ({ children }) => (
                              <blockquote className="my-3 pl-4 border-l-4 border-[#F3E66B] bg-[#FFF9CC]/40 rounded-r-xl py-2 px-3 text-xs text-[#5C4900] italic">
                                {children}
                              </blockquote>
                            ),
                          }}
                        >
                          {cleanChatContent(msg.content)}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Attachments on Latest Assistant Turn */}
                  {isLastAssistant && (
                    <div className="space-y-2.5">
                      {/* Deliverable Downloads */}
                      {currentArtifacts.length > 0 && (
                        <div className="p-3 rounded-xl bg-[#FFF9CC]/70 border border-[#E6D54F] flex flex-wrap items-center gap-2">
                          <span className="text-xs font-bold text-[#5C4900]">
                            Deliverables:
                          </span>
                          {currentArtifacts.map((art) => (
                            <a
                              key={art.artifact_id || art.filename}
                              href={getArtifactDownloadUrl(art)}
                              download={art.filename}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white border border-[#E6D54F] text-xs font-bold text-[#272436] hover:bg-[#F0EAFB] hover:border-[#7454B8]"
                            >
                              <Download className="w-3 h-3 text-[#7454B8]" />
                              {art.format.toUpperCase()}
                            </a>
                          ))}
                        </div>
                      )}

                      {/* Inline Charts */}
                      {currentCharts.filter((c) => c.image_data || c.png_base64).length > 0 && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {currentCharts
                            .filter((c) => c.image_data || c.png_base64)
                            .slice(0, 4)
                            .map((ch, ci) => {
                              const rawImg = ch.image_data || ch.png_base64 || "";
                              const imgSrc = rawImg.startsWith("data:")
                                ? rawImg
                                : `data:image/png;base64,${rawImg}`;
                              return (
                                <div
                                  key={ci}
                                  className="p-2.5 rounded-xl bg-white border border-[#EAE7EF]"
                                >
                                  <div className="text-xs font-bold text-[#272436] mb-1.5">
                                    {ch.title}
                                  </div>
                                  <img
                                    src={imgSrc}
                                    alt={ch.title}
                                    className="w-full h-auto rounded-lg"
                                  />
                                </div>
                              );
                            })}
                        </div>
                      )}

                      {/* Collapsible Per-Agent Work Illustrations */}
                      {currentIllustrations.length > 0 && (
                        <div className="rounded-xl border border-[#EAE7EF] bg-white p-3">
                          <button
                            onClick={() => setShowIllustrations((s) => !s)}
                            className="w-full flex items-center justify-between text-xs font-bold text-[#4F3685]"
                          >
                            <span className="flex items-center gap-1.5">
                              <Layers className="w-3.5 h-3.5" />
                              Agent Work Illustrations ({currentIllustrations.length} agents)
                            </span>
                            {showIllustrations ? (
                              <ChevronUp className="w-4 h-4" />
                            ) : (
                              <ChevronDown className="w-4 h-4" />
                            )}
                          </button>
                          {showIllustrations && (
                            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                              {currentIllustrations.map((ill, ii) => (
                                <div
                                  key={ii}
                                  className="p-2.5 rounded-xl bg-[#FAFAF4] border border-[#EAE7EF]"
                                >
                                  <div className="flex items-center justify-between text-xs font-bold text-[#272436]">
                                    <span>{ill.label || ill.agent_label || ill.agent}</span>
                                    {ill.duration_ms ? (
                                      <span className="text-[10px] text-[#777383]">
                                        {Math.round(ill.duration_ms)} ms
                                      </span>
                                    ) : null}
                                  </div>
                                  <p className="text-[11px] text-[#777383] mt-1 line-clamp-2">
                                    {ill.summary}
                                  </p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Evidence & Sources */}
                      {currentResponseEvidence.length > 0 && (
                        <div className="rounded-xl border border-[#EAE7EF] bg-white p-3">
                          <button
                            onClick={() =>
                              setShowEvidence((p) => ({ ...p, [idx]: !p[idx] }))
                            }
                            className="w-full flex items-center justify-between text-xs font-bold text-[#272436]"
                          >
                            <span className="flex items-center gap-1.5">
                              <Search className="w-3.5 h-3.5 text-[#7454B8]" />
                              Grounded Evidence ({currentResponseEvidence.length} items)
                            </span>
                            {showEvidence[idx] ? (
                              <ChevronUp className="w-4 h-4" />
                            ) : (
                              <ChevronDown className="w-4 h-4" />
                            )}
                          </button>
                          {showEvidence[idx] && (
                            <div className="mt-2.5 space-y-2">
                              {currentResponseEvidence.map((ev, ei) => (
                                <div
                                  key={ei}
                                  className="p-2.5 rounded-lg bg-[#FAFAF4] border border-[#EAE7EF] text-xs"
                                >
                                  <div className="flex items-center justify-between font-bold text-[#4F3685] mb-1">
                                    <span>{ev.source}</span>
                                    <span>
                                      {Math.round(ev.confidence * 100)}% conf
                                    </span>
                                  </div>
                                  <p className="text-[#272436]">{ev.claim}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={handleSend}
        className="p-4 border-t border-[#EAE7EF] bg-[#FAFAF4]"
      >
        <div className="flex flex-wrap gap-1.5 mb-2.5">
          {pipelineCommands.map((cmd) => (
            <button
              key={cmd.id}
              type="button"
              disabled={isLoading}
              onClick={() => {
                setActiveCommand(cmd.id);
                submitPrompt(cmd.prompt);
              }}
              className={`px-3 py-1 text-xs font-semibold rounded-lg border transition-colors ${
                activeCommand === cmd.id
                  ? "bg-[#FFF9CC] border-[#E6D54F] text-[#5C4900]"
                  : "bg-white border-[#EAE7EF] text-[#272436] hover:border-[#7454B8] hover:text-[#4F3685]"
              } disabled:opacity-50`}
            >
              {cmd.label}
            </button>
          ))}
        </div>
        <div className="flex items-end gap-2.5">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Describe your research objective, dataset analysis, or deliverable request..."
            className="input-field flex-1 min-h-[46px] max-h-36 resize-none"
            disabled={isLoading}
            rows={1}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="btn-primary h-[46px] px-4 inline-flex items-center justify-center gap-1.5 shrink-0"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
