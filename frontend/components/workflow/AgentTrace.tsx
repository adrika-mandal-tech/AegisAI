"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { CheckCircle2, Clock, Loader2, Sparkles } from "lucide-react";
import type { SSEEvent, PipelineResult } from "@/lib/api";
import { streamPipeline } from "@/lib/api";

const PIPELINE_STAGES = [
  {
    agent: "supervisor",
    label: "1. Supervisor",
    verb: "Triaging intent, complexity & dynamic agent delegation",
    accent: "plum",
  },
  {
    agent: "planner",
    label: "2. Planner",
    verb: "Constructing dependency-aware subtask execution plan",
    accent: "lemon",
  },
  {
    agent: "retriever",
    label: "3. Retriever",
    verb: "Running Qdrant hybrid dense + BM25 RRF vector search",
    accent: "plum",
  },
  {
    agent: "researcher",
    label: "4. Researcher",
    verb: "Synthesizing multi-source domain & external evidence",
    accent: "lemon",
  },
  {
    agent: "validator",
    label: "5. Validator",
    verb: "Verifying factual grounding, citations & contradictions",
    accent: "plum",
  },
  {
    agent: "reasoner",
    label: "6. Reasoner & Analyst",
    verb: "Executing quantitative computation & multi-step synthesis",
    accent: "lemon",
  },
  {
    agent: "critic",
    label: "7. Critic",
    verb: "Adversarial quality review & bounded revision gate",
    accent: "plum",
  },
  {
    agent: "report_generator",
    label: "8. Report & Deliverable Engine",
    verb: "Generating 12-section report, charts & downloadable files",
    accent: "lemon",
  },
  {
    agent: "evaluation",
    label: "Evaluation Suite",
    verb: "Computing Ragas & DeepEval live response metrics",
    accent: "plum",
  },
] as const;

type StageStatus = "pending" | "active" | "done";

function agentStageForEvent(agent: string): number | null {
  const idx = PIPELINE_STAGES.findIndex((s) => s.agent === agent);
  return idx >= 0 ? idx : null;
}

export function AgentTrace({
  task,
  isRunning,
  onComplete,
  onError,
  onAgentStateChange,
  docId,
  documentIds,
  requestedFormats,
}: {
  task: string;
  isRunning: boolean;
  onComplete: (result: PipelineResult) => void;
  onError: (error: string) => void;
  onAgentStateChange?: (state: string, activeAgent: string) => void;
  docId?: string;
  documentIds?: string[];
  requestedFormats?: string[];
}) {
  const [stages, setStages] = useState(() =>
    PIPELINE_STAGES.map(() => ({ status: "pending" as StageStatus, output: "" }))
  );
  const [agentOutputs, setAgentOutputs] = useState<Record<string, string>>({});
  const [activeIndex, setActiveIndex] = useState<number>(-1);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setStages(PIPELINE_STAGES.map(() => ({ status: "pending", output: "" })));
    setAgentOutputs({});
    setActiveIndex(-1);
  }, []);

  useEffect(() => {
    if (!isRunning || !task) {
      if (!isRunning && activeIndex !== -1) {
        setActiveIndex(-1);
      }
      return;
    }

    reset();
    setActiveIndex(0);
    onAgentStateChange?.("PLANNING", "supervisor");

    controllerRef.current = streamPipeline(
      task,
      (event: SSEEvent) => {
        const stageIndex = agentStageForEvent(event.agent);

        if (event.type === "task_state") {
          onAgentStateChange?.(event.status || "RUNNING", event.agent || "");
        } else if (event.type === "agent_start") {
          const mappedState =
            event.agent === "planner" || event.agent === "supervisor"
              ? "PLANNING"
              : event.agent === "validator"
              ? "VALIDATING"
              : event.agent === "critic"
              ? "REVISING"
              : event.agent === "report_generator"
              ? "GENERATING_ARTIFACT"
              : "RUNNING";
          onAgentStateChange?.(mappedState, event.agent);

          if (stageIndex !== null) {
            setActiveIndex(stageIndex);
            setStages((prev) =>
              prev.map((s, i) => {
                if (i === stageIndex) return { ...s, status: "active" as StageStatus };
                return s;
              })
            );
          }
        } else if (event.type === "agent_output") {
          if (event.output) {
            setAgentOutputs((prev) => ({
              ...prev,
              [event.agent]: event.output ?? "",
            }));
          }
        } else if (event.type === "agent_complete") {
          if (stageIndex !== null) {
            setStages((prev) => {
              const next = [...prev];
              if (stageIndex < next.length) {
                next[stageIndex] = { ...next[stageIndex], status: "done" };
              }
              return next;
            });
          }
        } else if (event.type === "pipeline_complete") {
          setStages((prev) => prev.map((s) => ({ ...s, status: "done" as StageStatus })));
          setActiveIndex(-1);
          onAgentStateChange?.("COMPLETED", "");
          onComplete({
            success: true,
            execution_id: event.execution_id,
            session_id: event.session_id,
            task_type: event.task_type,
            task_complexity: event.task_complexity,
            report_title: event.report_title || "",
            report_markdown: event.report_markdown || "",
            action_items: event.action_items || [],
            pipeline_trace: [],
            charts: event.charts || [],
            chart_data: event.chart_data || [],
            table_data: event.table_data || [],
            subtasks: event.subtasks || [],
            execution_plan: event.execution_plan,
            agent_work_illustrations: event.agent_work_illustrations || [],
            response_metrics: event.response_metrics,
            analysis_results: event.analysis_results,
            artifacts: event.artifacts || [],
            evaluation: event.evaluation || {},
          });
        } else if (event.type === "error") {
          setActiveIndex(-1);
          onAgentStateChange?.("FAILED", "");
          onError(event.output || "Unknown pipeline error");
        }
      },
      (error) => {
        setActiveIndex(-1);
        onAgentStateChange?.("FAILED", "");
        onError(error);
      },
      undefined,
      docId,
      {
        documentIds,
        formats: requestedFormats,
      }
    );

    return () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
    };
  }, [isRunning, task, docId, documentIds, requestedFormats, onComplete, onError, onAgentStateChange, reset]);

  const doneCount = stages.filter((s) => s.status === "done").length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-[#EAE7EF]">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-[#7454B8]" />
          <h3 className="text-sm font-bold text-[#272436]">
            8-Agent Execution Stream
          </h3>
        </div>
        <span className="badge badge-lemon">
          {doneCount}/{PIPELINE_STAGES.length} Stages
        </span>
      </div>

      {!isRunning && stages.every((s) => s.status === "pending") ? (
        <div className="rounded-2xl border border-dashed border-[#D8CCF2] bg-[#FAFAF4] p-6 text-center">
          <Clock className="w-8 h-8 text-[#7454B8] mx-auto mb-2 opacity-75" />
          <p className="text-xs font-semibold text-[#272436]">
            Agent Swarm Standby
          </p>
          <p className="text-[11px] text-[#777383] mt-1">
            Submit an objective above to stream Supervisor, Planner, Retriever, Researcher, Validator, Reasoner, Critic, and Deliverable generation live.
          </p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[620px] overflow-y-auto pr-1">
          {PIPELINE_STAGES.map((stage, index) => {
            const s = stages[index];
            const isActive = s.status === "active" || (isRunning && activeIndex === index);
            const isDone = s.status === "done";
            const outputText = agentOutputs[stage.agent] || "";

            return (
              <div
                key={stage.agent}
                className={`transition-all duration-300 rounded-xl border p-3 ${
                  isActive
                    ? stage.accent === "lemon"
                      ? "bg-[#FFF9CC] border-[#E6D54F] shadow-sm"
                      : "bg-[#F0EAFB] border-[#D8CCF2] shadow-sm"
                    : isDone
                    ? "bg-white border-[#EAE7EF]"
                    : "bg-[#FAFAF4] border-[#EAE7EF] opacity-65"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-xs font-bold ${
                      isActive
                        ? "text-[#4F3685]"
                        : isDone
                        ? "text-[#272436]"
                        : "text-[#777383]"
                    }`}
                  >
                    {stage.label}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {isActive && (
                      <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-[#4F3685]">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Running
                      </span>
                    )}
                    {isDone && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                    )}
                  </div>
                </div>
                <p className="text-[11px] text-[#777383] mt-1">{stage.verb}</p>

                {(isActive || isDone) && outputText && (
                  <div className="mt-2 p-2.5 rounded-lg bg-[#FAFAF4] border border-[#EAE7EF] text-[11px] text-[#272436] font-mono whitespace-pre-wrap max-h-28 overflow-y-auto leading-relaxed">
                    {outputText}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
