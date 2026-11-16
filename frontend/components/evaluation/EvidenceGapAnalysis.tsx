"use client";

import { AlertTriangle, Lightbulb, BarChart3 } from "lucide-react";

interface EvidenceGapData {
  gap_score: number;
  identified_gaps: string[];
  coverage_metrics: {
    coverage_ratio: number;
    total_claims: number;
    supported_claims: number;
    unsupported_claims: number;
    partially_supported_claims: number;
    evidence_density: number;
  };
  recommendations: string[];
}

export function EvidenceGapAnalysis({
  analysis,
  className = "",
}: {
  analysis: EvidenceGapData | null;
  className?: string;
}) {
  if (!analysis) return null;

  return (
    <div
      className={`rounded-2xl border border-[#EAE7EF] bg-[#FAFAF4] p-5 ${className}`}
    >
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-4 h-4 text-[#7454B8]" />
        <h3 className="text-xs font-bold text-[#272436] uppercase tracking-wider">
          Evidence Coverage &amp; Gap Verification
        </h3>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="rounded-xl bg-white border border-[#EAE7EF] p-3">
          <div className="text-[#777383] text-xs">Gap Score</div>
          <div className="mt-1 font-extrabold text-lg text-[#4F3685]">
            {(analysis.gap_score * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-xl bg-white border border-[#EAE7EF] p-3">
          <div className="text-[#777383] text-xs">Claim Coverage</div>
          <div className="mt-1 font-extrabold text-lg text-emerald-700">
            {(analysis.coverage_metrics.coverage_ratio * 100).toFixed(0)}%
          </div>
        </div>
        <div className="rounded-xl bg-white border border-[#EAE7EF] p-3">
          <div className="text-[#777383] text-xs">Evidence Density</div>
          <div className="mt-1 font-extrabold text-lg text-[#272436]">
            {analysis.coverage_metrics.evidence_density.toFixed(1)}x
          </div>
        </div>
        <div className="rounded-xl bg-white border border-[#EAE7EF] p-3">
          <div className="text-[#777383] text-xs">Flagged Gaps</div>
          <div className="mt-1 font-extrabold text-lg text-amber-700">
            {analysis.identified_gaps.length}
          </div>
        </div>
      </div>

      {analysis.identified_gaps.length > 0 && (
        <div className="space-y-1.5 mb-3">
          {analysis.identified_gaps.map((gap, i) => (
            <div
              key={i}
              className="flex items-start gap-2 p-2.5 bg-[#FFF9CC]/60 border border-[#E6D54F] rounded-xl text-xs text-[#5C4900]"
            >
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{gap}</span>
            </div>
          ))}
        </div>
      )}

      {analysis.recommendations.length > 0 && (
        <div className="space-y-1.5">
          {analysis.recommendations.map((rec, i) => (
            <div
              key={i}
              className="flex items-start gap-2 p-2.5 bg-white border border-[#EAE7EF] rounded-xl text-xs text-[#272436]"
            >
              <Lightbulb className="w-3.5 h-3.5 text-[#7454B8] shrink-0 mt-0.5" />
              <span>{rec}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
