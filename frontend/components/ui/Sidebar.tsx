"use client";

import {
  Activity,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  History,
  LayoutDashboard,
  Shield,
  Sparkles,
  UploadCloud,
  Workflow,
} from "lucide-react";
import {
  ContactDeveloper,
  NavbarSocialIcons,
  type DeveloperLinkKey,
} from "@/components/ui/DeveloperLinks";
import type { DocItem, ResponseMetrics } from "@/lib/api";

export const tabs = [
  { id: "home", label: "Workspace", icon: LayoutDashboard },
  { id: "chat", label: "Conversational Swarm", icon: Brain },
  { id: "documents", label: "Documents & OCR", icon: UploadCloud },
  { id: "analytics", label: "Analytics Lab", icon: FileSpreadsheet },
  { id: "deliverables", label: "Deliverables Hub", icon: Download },
  { id: "history", label: "Execution History", icon: History },
  { id: "docs", label: "Architecture & Docs", icon: BookOpen },
  { id: "about", label: "About Platform", icon: Shield },
] as const;

export type TabId = (typeof tabs)[number]["id"];

interface SidebarProps {
  activeTab: TabId;
  onTabChange: (tabId: TabId) => void;
  responseMetrics?: ResponseMetrics | null;
  taskState?: string;
  activeAgent?: string;
  lastExecutionId?: string;
  recentUploads?: DocItem[];
  onSelectDocument?: (doc: DocItem) => void;
  healthStatus?: string;
  activeContact?: DeveloperLinkKey | null;
  onContactClick?: (key: DeveloperLinkKey) => void;
}

function pct(value: number | undefined): string {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export function Sidebar({
  activeTab,
  onTabChange,
  responseMetrics,
  taskState = "IDLE",
  activeAgent = "",
  lastExecutionId = "",
  recentUploads = [],
  onSelectDocument,
  healthStatus = "ok",
  activeContact = null,
  onContactClick,
}: SidebarProps) {
  const hasMetrics =
    responseMetrics &&
    Object.values(responseMetrics).some((v) => typeof v === "number");

  const metricRows: {
    key: keyof ResponseMetrics;
    label: string;
    inverse?: boolean;
  }[] = [
    { key: "hallucination_rate", label: "Hallucination Rate", inverse: true },
    { key: "accuracy", label: "Accuracy" },
    { key: "correctness", label: "Correctness" },
    { key: "context_precision", label: "Context Precision" },
    { key: "context_recall", label: "Context Recall" },
    { key: "task_success_rate", label: "Task Success Rate" },
    { key: "tool_call_accuracy", label: "Tool Call Accuracy" },
  ];

  const isRunningState = [
    "PENDING",
    "PLANNING",
    "RUNNING",
    "WAITING_FOR_TOOL",
    "VALIDATING",
    "REVISING",
    "GENERATING_ARTIFACT",
  ].includes(taskState);

  return (
    <aside className="w-full lg:w-80 lg:min-h-screen lg:sticky lg:top-0 bg-white border-b lg:border-b-0 lg:border-r border-[#EAE7EF] flex flex-col shrink-0 z-20">
      {/* Brand Header */}
      <div className="px-5 py-4 border-b border-[#EAE7EF] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#7454B8] to-[#4F3685] flex items-center justify-center shadow-sm relative">
            <Shield className="w-5 h-5 text-[#F3E66B]" />
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-[#F3E66B] border-2 border-white" />
          </div>
          <div>
            <h1 className="font-extrabold text-[#272436] tracking-tight text-lg leading-tight">
              AegisAI
            </h1>
            <p className="text-[10px] font-semibold text-[#7454B8] tracking-tight leading-tight">
              Autonomous Multi-Agent AI (8 Agents)
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden sm:inline-flex badge badge-lemon">
            <Sparkles className="w-3 h-3" /> v1.0
          </span>
          <NavbarSocialIcons
            activeKey={activeContact}
            onSelect={onContactClick}
            size="sm"
            className="lg:hidden"
          />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex lg:flex-col gap-1 px-3 py-3 overflow-x-auto lg:overflow-x-visible border-b border-[#EAE7EF]">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`whitespace-nowrap flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                isActive
                  ? "bg-[#F0EAFB] text-[#4F3685] border border-[#D8CCF2] shadow-sm"
                  : "text-[#777383] hover:text-[#272436] hover:bg-[#FAFAF4] border border-transparent"
              }`}
            >
              <tab.icon
                className={`w-4 h-4 shrink-0 ${
                  isActive ? "text-[#7454B8]" : "text-[#777383]"
                }`}
              />
              <span>{tab.label}</span>
              {isActive && (
                <span className="ml-auto hidden lg:inline-block w-2 h-2 rounded-full bg-[#F3E66B] border border-[#C9B62C]" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Live Execution State Indicator */}
      <div className="px-4 py-3 border-b border-[#EAE7EF] bg-[#FAFAF4]">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[11px] font-bold uppercase tracking-wider text-[#777383] flex items-center gap-1.5">
            <Activity
              className={`w-3.5 h-3.5 ${
                isRunningState ? "text-[#7454B8] animate-pulse" : "text-emerald-600"
              }`}
            />
            Orchestration State
          </span>
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
              isRunningState
                ? "bg-[#FFF9CC] text-[#5C4900] border-[#E6D54F]"
                : taskState === "COMPLETED"
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : taskState === "FAILED"
                ? "bg-rose-50 text-rose-700 border-rose-200"
                : "bg-white text-[#777383] border-[#EAE7EF]"
            }`}
          >
            {taskState}
          </span>
        </div>
        {activeAgent ? (
          <p className="text-xs text-[#4F3685] font-medium truncate">
            Active agent: <span className="font-bold capitalize">{activeAgent.replace(/_/g, " ")}</span>
          </p>
        ) : lastExecutionId ? (
          <p className="text-[11px] text-[#777383] font-mono truncate">
            ID: {lastExecutionId}
          </p>
        ) : (
          <p className="text-[11px] text-[#777383]">
            8-Agent Swarm Ready · Max 500 MB / file
          </p>
        )}
      </div>

      {/* Live Per-Response Evaluation Metrics Panel */}
      <div className="px-4 py-3.5 border-b border-[#EAE7EF] flex-1 overflow-y-auto">
        <div className="flex items-center justify-between mb-2.5">
          <div className="flex items-center gap-1.5">
            <BarChart3 className="w-4 h-4 text-[#7454B8]" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-[#272436]">
              Live Response Metrics
            </h2>
          </div>
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[#FFF9CC] text-[#5C4900] border border-[#E6D54F]">
            {hasMetrics ? "Verified" : "Standby"}
          </span>
        </div>

        <div className="space-y-2.5">
          {metricRows.map((row) => {
            const rawVal = responseMetrics?.[row.key];
            const numericVal =
              typeof rawVal === "number" && !Number.isNaN(rawVal)
                ? Math.max(0, Math.min(1, rawVal))
                : null;

            let barColor = "bg-[#7454B8]";
            let pillClass = "bg-[#F0EAFB] text-[#4F3685]";
            if (numericVal !== null) {
              if (row.inverse) {
                if (numericVal <= 0.15) {
                  barColor = "bg-emerald-500";
                  pillClass = "bg-emerald-50 text-emerald-700";
                } else if (numericVal <= 0.3) {
                  barColor = "bg-[#E2C83B]";
                  pillClass = "bg-[#FFF9CC] text-[#5C4900]";
                } else {
                  barColor = "bg-rose-500";
                  pillClass = "bg-rose-50 text-rose-700";
                }
              } else {
                if (numericVal >= 0.8) {
                  barColor = "bg-[#7454B8]";
                  pillClass = "bg-[#F0EAFB] text-[#4F3685]";
                } else if (numericVal >= 0.6) {
                  barColor = "bg-[#E2C83B]";
                  pillClass = "bg-[#FFF9CC] text-[#5C4900]";
                } else {
                  barColor = "bg-rose-500";
                  pillClass = "bg-rose-50 text-rose-700";
                }
              }
            }

            return (
              <div
                key={row.key}
                className="p-2 rounded-xl bg-[#FAFAF4] border border-[#EAE7EF]"
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-[#272436]">{row.label}</span>
                  <span
                    className={`text-[11px] font-bold px-1.5 py-0.5 rounded-md ${
                      numericVal !== null
                        ? pillClass
                        : "text-[#777383] bg-white border border-[#EAE7EF]"
                    }`}
                  >
                    {numericVal !== null ? pct(numericVal) : "—"}
                  </span>
                </div>
                <div className="w-full h-1.5 rounded-full bg-[#EAE7EF] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                    style={{
                      width: numericVal !== null ? `${Math.max(4, numericVal * 100)}%` : "0%",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {!hasMetrics && (
          <p className="mt-2.5 text-[11px] text-[#777383] leading-relaxed">
            Run any query in <strong>Agent Execution</strong> or <strong>Conversational Swarm</strong> to populate live Ragas &amp; DeepEval metrics.
          </p>
        )}
      </div>

      {/* Recent Indexed Documents */}
      {recentUploads.length > 0 && (
        <div className="px-4 py-3 border-b border-[#EAE7EF] hidden lg:block">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-[#777383]">
              Indexed Corpus ({recentUploads.length})
            </span>
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {recentUploads.slice(0, 5).map((doc) => (
              <button
                key={doc.id}
                onClick={() => onSelectDocument?.(doc)}
                className="w-full text-left px-2.5 py-1.5 rounded-lg hover:bg-[#F0EAFB] transition-colors border border-[#EAE7EF] bg-[#FAFAF4] flex items-center justify-between gap-2"
              >
                <span className="text-xs font-medium text-[#272436] truncate">
                  {doc.filename}
                </span>
                <span className="text-[10px] font-semibold text-[#4F3685] shrink-0">
                  {doc.chunks}c
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Contact Developer */}
      <div className="px-4 py-3 border-b border-[#EAE7EF] bg-white">
        <ContactDeveloper
          activeKey={activeContact}
          onSelect={onContactClick}
          variant="sidebar"
        />
      </div>

      {/* Footer Status */}
      <div className="px-4 py-3 bg-[#FAFAF4] text-xs text-[#777383] flex items-center justify-between">
        <span className="font-medium text-[#272436]">Max 500 MB / File</span>
        <span
          className={`inline-flex items-center gap-1 text-[11px] font-semibold ${
            healthStatus === "ok" ? "text-emerald-700" : "text-amber-700"
          }`}
        >
          <CheckCircle2 className="w-3.5 h-3.5" />
          {healthStatus === "ok" ? "Online" : "Checking"}
        </span>
      </div>
    </aside>
  );
}
