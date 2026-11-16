"use client";

import { useEffect, useState, useCallback } from "react";
import {
  History,
  Loader2,
  AlertCircle,
  ChevronRight,
  Download,
} from "lucide-react";
import {
  getHistory,
  getArtifactDownloadUrl,
  type HistoryRun,
} from "@/lib/api";

export function HistoryList() {
  const [runs, setRuns] = useState<HistoryRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory();
      if (res.success) {
        setRuns(res.runs);
      } else {
        setError(res.error || "Failed to load history");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-bold text-[#272436]">
            Multi-Agent Execution History
          </h3>
          <p className="text-xs text-[#777383]">
            Inspect completed runs, evaluation metrics, reports, and downloadable artifacts.
          </p>
        </div>
        <button
          onClick={fetchHistory}
          disabled={loading}
          className="btn-secondary inline-flex items-center gap-1.5 text-xs"
        >
          {loading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <History className="w-3.5 h-3.5" />
          )}
          Refresh
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-xs text-rose-700 bg-rose-50 p-3 rounded-xl border border-rose-200">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {runs.length === 0 && !loading && (
        <div className="text-center py-12 rounded-2xl bg-[#FAFAF4] border border-dashed border-[#D8CCF2] text-xs text-[#777383]">
          <History className="w-10 h-10 mx-auto mb-2 text-[#7454B8] opacity-60" />
          No execution runs recorded yet.
        </div>
      )}

      <div className="space-y-2.5">
        {runs.map((run, index) => {
          const isExpanded = expandedId === index;
          return (
            <div
              key={run.execution_id || index}
              className="bg-[#FAFAF4] rounded-2xl border border-[#EAE7EF] overflow-hidden"
            >
              <button
                onClick={() => setExpandedId(isExpanded ? null : index)}
                className="w-full text-left p-4 flex items-start justify-between gap-3 hover:bg-[#F0EAFB]/40 transition-colors"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <span className="text-sm font-bold text-[#272436] truncate">
                      {run.report_title || run.task.slice(0, 80)}
                    </span>
                    {run.execution_id && (
                      <span className="badge badge-lemon font-mono">
                        {run.execution_id}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-[#777383] line-clamp-2 mb-2">
                    {run.task}
                  </p>
                  <div className="flex flex-wrap items-center gap-3 text-[11px] text-[#777383]">
                    <span>{run.timestamp?.slice(0, 19).replace("T", " ")}</span>
                    <span>·</span>
                    <span className="font-semibold text-[#4F3685]">
                      {run.agent_count} agents
                    </span>
                    {run.response_metrics?.accuracy != null && (
                      <>
                        <span>·</span>
                        <span className="font-semibold text-emerald-700">
                          Accuracy {(run.response_metrics.accuracy * 100).toFixed(1)}%
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <ChevronRight
                  className={`w-4 h-4 text-[#777383] shrink-0 transition-transform ${
                    isExpanded ? "rotate-90" : ""
                  }`}
                />
              </button>

              {isExpanded && (
                <div className="px-4 pb-4 space-y-3">
                  {run.artifacts && run.artifacts.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {run.artifacts.map((art) => (
                        <a
                          key={art.artifact_id || art.filename}
                          href={getArtifactDownloadUrl(art)}
                          download={art.filename}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white border border-[#EAE7EF] text-xs font-bold text-[#4F3685] hover:border-[#7454B8]"
                        >
                          <Download className="w-3 h-3" />
                          {art.format.toUpperCase()}
                        </a>
                      ))}
                    </div>
                  )}
                  {run.report_markdown && (
                    <div className="rounded-xl bg-white border border-[#EAE7EF] p-4 text-xs text-[#272436] whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto">
                      {run.report_markdown}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
