"use client";

import { useState } from "react";
import {
  BarChart3,
  CheckCircle2,
  Clock,
  Download,
  FileSpreadsheet,
  FileText,
  Image as ImageIcon,
  Layers,
  ListChecks,
  Loader2,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { EvidenceGapAnalysis } from "@/components/evaluation/EvidenceGapAnalysis";
import {
  generateArtifactsOnDemand,
  getArtifactDownloadUrl,
  type AgentWorkIllustration,
  type ArtifactItem,
  type ChartData,
  type SubtaskItem,
  type TableData,
} from "@/lib/api";

interface ReportViewProps {
  executionId?: string;
  markdown: string;
  title: string;
  actionItems?: string[];
  charts?: string[];
  chartData?: ChartData[];
  tableData?: TableData[];
  subtasks?: SubtaskItem[];
  illustrations?: AgentWorkIllustration[];
  artifacts?: ArtifactItem[];
  analysisResults?: Record<string, any>;
  evaluation?: Record<string, any>;
  onDownloadPdf?: () => void;
  onArtifactsGenerated?: (items: ArtifactItem[]) => void;
}

type ResultTab = "report" | "illustrations" | "planning" | "analytics" | "deliverables";

function formatSize(bytes: number): string {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(2)} MB`;
}

export function ReportView({
  executionId,
  markdown,
  title,
  actionItems = [],
  charts = [],
  chartData = [],
  tableData = [],
  subtasks = [],
  illustrations = [],
  artifacts = [],
  analysisResults = {},
  evaluation,
  onDownloadPdf,
  onArtifactsGenerated,
}: ReportViewProps) {
  const [activeTab, setActiveTab] = useState<ResultTab>("report");
  const [exportingAll, setExportingAll] = useState(false);
  const [localArtifacts, setLocalArtifacts] = useState<ArtifactItem[]>([]);

  const mergedArtifacts =
    localArtifacts.length > 0
      ? localArtifacts
      : artifacts;

  const hasContent =
    Boolean(markdown) ||
    charts.length > 0 ||
    chartData.length > 0 ||
    illustrations.length > 0;

  if (!hasContent) {
    return (
      <div className="text-center py-16 text-[#777383] flex flex-col items-center justify-center h-full">
        <div className="w-14 h-14 rounded-2xl bg-[#F0EAFB] border border-[#D8CCF2] flex items-center justify-center mb-4">
          <FileText className="w-7 h-7 text-[#7454B8]" />
        </div>
        <p className="text-base font-bold text-[#272436]">
          Executive Workspace Ready
        </p>
        <p className="text-xs text-[#777383] max-w-md mt-1.5 leading-relaxed">
          Enter a complex research, document analysis, dataset profiling, or deliverable task above. Your synthesized report, per-agent work illustrations, subtask plan, charts, and 7-format downloads will appear here.
        </p>
      </div>
    );
  }

  const allChartImages: { title: string; src: string; type?: string }[] = [];
  for (const cd of chartData) {
    if (cd.image_data) {
      allChartImages.push({
        title: cd.title || "Analytical Chart",
        src: cd.image_data.startsWith("data:")
          ? cd.image_data
          : `data:image/png;base64,${cd.image_data}`,
        type: cd.chart_type,
      });
    }
  }
  if (allChartImages.length === 0) {
    charts.forEach((raw, idx) => {
      if (raw) {
        allChartImages.push({
          title: `Visualization ${idx + 1}`,
          src: raw.startsWith("data:") ? raw : `data:image/png;base64,${raw}`,
        });
      }
    });
  }

  const handleGenerateAllFormats = async () => {
    if (!markdown || exportingAll) return;
    setExportingAll(true);
    try {
      const res = await generateArtifactsOnDemand({
        execution_id: executionId,
        title: title || "AegisAI Intelligence Report",
        markdown,
        action_items: actionItems,
        formats: ["pdf", "docx", "pptx", "xlsx", "csv", "html", "md"],
      });
      if (res.success && res.artifacts) {
        setLocalArtifacts(res.artifacts);
        onArtifactsGenerated?.(res.artifacts);
      }
    } catch (err) {
      console.error("Failed to generate deliverables:", err);
    } finally {
      setExportingAll(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Header + Primary Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#EAE7EF]">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="badge badge-purple">8-Agent Verified Output</span>
            {executionId && (
              <span className="badge badge-lemon font-mono">{executionId}</span>
            )}
          </div>
          {title && (
            <h2 className="text-xl md:text-2xl font-extrabold text-[#272436]">
              {title}
            </h2>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {onDownloadPdf && (
            <button
              onClick={onDownloadPdf}
              className="btn-lemon inline-flex items-center gap-1.5 text-xs"
            >
              <Download className="w-3.5 h-3.5" />
              Quick PDF
            </button>
          )}
          <button
            onClick={() => setActiveTab("deliverables")}
            className="btn-primary inline-flex items-center gap-1.5 text-xs"
          >
            <Download className="w-3.5 h-3.5" />
            Deliverables ({mergedArtifacts.length})
          </button>
        </div>
      </div>

      {/* Sub-View Tabs */}
      <div className="flex flex-wrap gap-1.5 p-1.5 rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF]">
        {[
          { id: "report", label: "Final Report", icon: FileText },
          {
            id: "illustrations",
            label: `Agent Work (${illustrations.length})`,
            icon: Layers,
          },
          {
            id: "planning",
            label: `Planning (${subtasks.length})`,
            icon: ListChecks,
          },
          {
            id: "analytics",
            label: `Charts & Tables (${allChartImages.length + tableData.length})`,
            icon: BarChart3,
          },
          {
            id: "deliverables",
            label: `Deliverables (${mergedArtifacts.length})`,
            icon: Download,
          },
        ].map((t) => {
          const isCurrent = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id as ResultTab)}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                isCurrent
                  ? "bg-white text-[#4F3685] border border-[#D8CCF2] shadow-sm"
                  : "text-[#777383] hover:text-[#272436]"
              }`}
            >
              <t.icon className="w-3.5 h-3.5" />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: FINAL REPORT */}
      {activeTab === "report" && (
        <div className="space-y-6 animate-fade-in">
          {/* Quick Deliverable Strip */}
          {mergedArtifacts.length > 0 && (
            <div className="p-3.5 rounded-2xl bg-[#FFF9CC]/65 border border-[#E6D54F] flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-bold text-[#5C4900] flex items-center gap-1.5">
                <Sparkles className="w-4 h-4" />
                Verified Downloadable Deliverables Ready:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {mergedArtifacts.map((art) => (
                  <a
                    key={art.artifact_id || art.filename}
                    href={getArtifactDownloadUrl(art)}
                    download={art.filename}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white border border-[#E6D54F] text-xs font-bold text-[#272436] hover:bg-[#F0EAFB] hover:border-[#7454B8] transition-colors"
                  >
                    <Download className="w-3 h-3 text-[#7454B8]" />
                    {art.format.toUpperCase()}
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Inline Preview Charts */}
          {allChartImages.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {allChartImages.slice(0, 2).map((ch, i) => (
                <div
                  key={i}
                  className="rounded-2xl border border-[#EAE7EF] bg-[#FAFAF4] p-3 overflow-hidden"
                >
                  <div className="text-xs font-bold text-[#272436] mb-2 px-1">
                    {ch.title}
                  </div>
                  <img
                    src={ch.src}
                    alt={ch.title}
                    className="w-full h-auto rounded-xl border border-[#EAE7EF] bg-white"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Markdown Content */}
          {markdown && (
            <div className="max-w-none text-[#272436]">
              <ReactMarkdown
                components={{
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-extrabold text-[#272436] mb-4 pb-2 border-b border-[#EAE7EF]">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-lg font-bold text-[#4F3685] mt-6 mb-2.5 pb-1.5 border-b border-[#EAE7EF]">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-base font-bold text-[#272436] mt-4 mb-2">
                      {children}
                    </h3>
                  ),
                  p: ({ children }) => (
                    <p className="text-sm text-[#272436] leading-relaxed mb-3.5">
                      {children}
                    </p>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-5 space-y-1.5 text-sm text-[#272436] mb-4">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-5 space-y-1.5 text-sm text-[#272436] mb-4">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  strong: ({ children }) => (
                    <strong className="font-bold text-[#272436]">{children}</strong>
                  ),
                  code: ({ children }) => (
                    <code className="bg-[#F0EAFB] text-[#4F3685] rounded px-1.5 py-0.5 text-xs font-mono">
                      {children}
                    </code>
                  ),
                  pre: ({ children }) => (
                    <pre className="bg-[#FAFAF4] text-[#272436] p-4 rounded-xl border border-[#EAE7EF] text-xs overflow-x-auto mb-4">
                      {children}
                    </pre>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-[#7454B8] bg-[#F0EAFB]/50 pl-4 py-2 my-3 rounded-r-xl text-sm text-[#4F3685]">
                      {children}
                    </blockquote>
                  ),
                }}
              >
                {markdown}
              </ReactMarkdown>
            </div>
          )}

          {evaluation?.evidence_gap_analysis && (
            <EvidenceGapAnalysis analysis={evaluation.evidence_gap_analysis} />
          )}
        </div>
      )}

      {/* TAB 2: PER-AGENT WORK ILLUSTRATIONS */}
      {activeTab === "illustrations" && (
        <div className="space-y-4 animate-fade-in">
          <p className="text-xs text-[#777383]">
            Transparent breakdown of the cognitive work, key outputs, and execution latency measured across each specialized agent in this run.
          </p>
          {illustrations.length === 0 ? (
            <div className="p-8 text-center rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] text-xs text-[#777383]">
              No agent illustration cards recorded for this run yet.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {illustrations.map((ill, idx) => (
                <div
                  key={`${ill.agent}-${idx}`}
                  className="rounded-2xl border border-[#EAE7EF] bg-[#FAFAF4] p-4 flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <div>
                        <span className="text-xs font-extrabold text-[#4F3685] uppercase tracking-wider">
                          {ill.label || ill.agent_label || ill.agent}
                        </span>
                        <p className="text-[11px] font-medium text-[#777383]">
                          {ill.role}
                        </p>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {typeof ill.duration_ms === "number" && ill.duration_ms > 0 && (
                          <span className="badge badge-lemon">
                            <Clock className="w-3 h-3" />
                            {Math.round(ill.duration_ms)} ms
                          </span>
                        )}
                        <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      </div>
                    </div>

                    <p className="text-xs text-[#272436] leading-relaxed mb-3 bg-white p-3 rounded-xl border border-[#EAE7EF]">
                      {ill.summary}
                    </p>

                    {ill.key_outputs && ill.key_outputs.length > 0 && (
                      <div className="space-y-1 mb-3">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-[#777383]">
                          Key Outputs
                        </div>
                        {ill.key_outputs.slice(0, 4).map((ko, ki) => (
                          <div
                            key={ki}
                            className="text-[11px] text-[#272436] flex items-start gap-1.5"
                          >
                            <span className="text-[#7454B8] font-bold">•</span>
                            <span className="truncate">{ko}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {ill.metrics && Object.keys(ill.metrics).length > 0 && (
                    <div className="pt-2.5 border-t border-[#EAE7EF] flex flex-wrap gap-1.5">
                      {Object.entries(ill.metrics).map(([mk, mv]) => (
                        <span
                          key={mk}
                          className="text-[10px] font-semibold px-2 py-0.5 rounded-md bg-white border border-[#EAE7EF] text-[#4F3685]"
                        >
                          {mk.replace(/_/g, " ")}: {String(mv)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 3: PLANNING & SUBTASKS */}
      {activeTab === "planning" && (
        <div className="space-y-4 animate-fade-in">
          {subtasks.length === 0 ? (
            <div className="p-8 text-center rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] text-xs text-[#777383]">
              Planner subtask graph will populate on multi-agent execution.
            </div>
          ) : (
            <div className="overflow-x-auto rounded-2xl border border-[#EAE7EF]">
              <table className="w-full text-left text-xs">
                <thead className="bg-[#F0EAFB] text-[#4F3685]">
                  <tr>
                    <th className="px-4 py-3 font-bold">ID</th>
                    <th className="px-4 py-3 font-bold">Assigned Agent</th>
                    <th className="px-4 py-3 font-bold">Subtask Objective</th>
                    <th className="px-4 py-3 font-bold">Dependencies</th>
                    <th className="px-4 py-3 font-bold">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#EAE7EF] bg-white">
                  {subtasks.map((st, idx) => (
                    <tr key={st.id || idx} className="hover:bg-[#FAFAF4]">
                      <td className="px-4 py-3 font-mono font-bold text-[#4F3685]">
                        {st.id || `T${idx + 1}`}
                      </td>
                      <td className="px-4 py-3">
                        <span className="badge badge-purple capitalize">
                          {st.assigned_agent}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-[#272436]">
                        {st.description || st.title}
                      </td>
                      <td className="px-4 py-3 font-mono text-[#777383]">
                        {st.dependencies && st.dependencies.length > 0
                          ? st.dependencies.join(", ")
                          : "root"}
                      </td>
                      <td className="px-4 py-3">
                        <span className="badge badge-green">
                          {st.status || "completed"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* TAB 4: ANALYTICS, CHARTS & TABLES */}
      {activeTab === "analytics" && (
        <div className="space-y-6 animate-fade-in">
          {allChartImages.length > 0 ? (
            <div>
              <h3 className="text-sm font-bold text-[#272436] mb-3 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-[#7454B8]" />
                Generated Visualizations ({allChartImages.length})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {allChartImages.map((ch, idx) => (
                  <div
                    key={idx}
                    className="rounded-2xl border border-[#EAE7EF] bg-[#FAFAF4] p-4"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-[#272436]">
                        {ch.title}
                      </span>
                      {ch.type && (
                        <span className="badge badge-lemon uppercase">
                          {ch.type}
                        </span>
                      )}
                    </div>
                    <img
                      src={ch.src}
                      alt={ch.title}
                      className="w-full h-auto rounded-xl bg-white border border-[#EAE7EF]"
                    />
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="p-6 text-center rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF] text-xs text-[#777383]">
              No quantitative charts were rendered for this query. Upload a CSV/XLSX dataset or include numeric metrics to generate statistical charts.
            </div>
          )}

          {tableData.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-bold text-[#272436] flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-[#7454B8]" />
                Structured Data Tables ({tableData.length})
              </h3>
              {tableData.map((tbl, ti) => (
                <div
                  key={ti}
                  className="rounded-2xl border border-[#EAE7EF] overflow-hidden bg-white"
                >
                  <div className="px-4 py-2.5 bg-[#FAFAF4] border-b border-[#EAE7EF] text-xs font-bold text-[#272436]">
                    {tbl.title || `Table ${ti + 1}`}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-[#F0EAFB] text-[#4F3685]">
                        <tr>
                          {tbl.headers.map((h, hi) => (
                            <th key={hi} className="px-3.5 py-2.5 font-bold">
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#EAE7EF]">
                        {tbl.rows.slice(0, 25).map((r, ri) => (
                          <tr key={ri} className="hover:bg-[#FAFAF4]">
                            {r.map((cell, ci) => (
                              <td key={ci} className="px-3.5 py-2 text-[#272436]">
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

      {/* TAB 5: DELIVERABLES (7 FORMATS) */}
      {activeTab === "deliverables" && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-[#FAFAF4] border border-[#EAE7EF]">
            <div>
              <h3 className="text-sm font-bold text-[#272436]">
                Multi-Format Deliverable Artifacts
              </h3>
              <p className="text-xs text-[#777383]">
                Download verified PDF reports, Word documents, PowerPoint decks, Excel workbooks, CSV exports, HTML reports, and Markdown files.
              </p>
            </div>
            <button
              onClick={handleGenerateAllFormats}
              disabled={exportingAll || !markdown}
              className="btn-primary inline-flex items-center gap-1.5 text-xs shrink-0"
            >
              {exportingAll ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              {exportingAll ? "Generating All 7 Formats..." : "Generate All 7 Formats"}
            </button>
          </div>

          {mergedArtifacts.length === 0 ? (
            <div className="p-8 text-center rounded-2xl border border-dashed border-[#D8CCF2] bg-[#FAFAF4] text-xs text-[#777383]">
              Click <strong>Generate All 7 Formats</strong> above to export PDF, DOCX, PPTX, XLSX, CSV, HTML, and Markdown deliverables for this report.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {mergedArtifacts.map((art) => (
                <div
                  key={art.artifact_id || art.filename}
                  className="rounded-2xl border border-[#EAE7EF] bg-white p-4 flex flex-col justify-between hover:border-[#7454B8] transition-colors shadow-sm"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="badge badge-lemon uppercase font-bold">
                        {art.format}
                      </span>
                      <span className="text-[11px] font-medium text-[#777383]">
                        {formatSize(art.size_bytes)}
                      </span>
                    </div>
                    <p
                      className="text-xs font-bold text-[#272436] truncate mb-1"
                      title={art.filename}
                    >
                      {art.filename}
                    </p>
                    <p className="text-[11px] text-[#777383] font-mono truncate">
                      {art.execution_id}
                    </p>
                  </div>
                  <a
                    href={getArtifactDownloadUrl(art)}
                    download={art.filename}
                    className="mt-3 btn-secondary w-full inline-flex items-center justify-center gap-1.5 text-xs py-2"
                  >
                    <Download className="w-3.5 h-3.5 text-[#7454B8]" />
                    Download {art.format.toUpperCase()}
                  </a>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
