"use client";

import { useState } from "react";
import {
  Search,
  AlertTriangle,
  Lightbulb,
  FileText,
  Flag,
  X,
  Loader2,
  Brain,
  MessageSquare,
} from "lucide-react";
import {
  analyzeDocument,
  type AnalyzeRequest,
  type AnalyzeResponse,
} from "@/lib/api";

interface DocumentAnalysisProps {
  documentId: string;
  documentName: string;
  onClose: () => void;
}

const ANALYSIS_TYPES = [
  {
    value: "general",
    label: "General Q&A",
    icon: MessageSquare,
    description: "Ask any grounded question",
  },
  {
    value: "summary",
    label: "Executive Summary",
    icon: FileText,
    description: "Synthesize core takeaways",
  },
  {
    value: "risks",
    label: "Risk Audit",
    icon: AlertTriangle,
    description: "Identify operational & financial risks",
  },
  {
    value: "recommendations",
    label: "Action Items",
    icon: Lightbulb,
    description: "Extract concrete recommendations",
  },
  {
    value: "contradictions",
    label: "Contradictions",
    icon: Flag,
    description: "Spot conflicts or missing data",
  },
] as const;

export function DocumentAnalysis({
  documentId,
  documentName,
  onClose,
}: DocumentAnalysisProps) {
  const [question, setQuestion] = useState("");
  const [analysisType, setAnalysisType] = useState<
    "general" | "summary" | "risks" | "recommendations" | "contradictions"
  >("general");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const defaultQuestionForType: Record<string, string> = {
    summary: "Provide a comprehensive executive summary of this document.",
    risks: "Identify all risks, anomalies, and vulnerabilities in this document.",
    recommendations: "Extract actionable recommendations from this document.",
    contradictions: "Identify any contradictions, inconsistencies, or data gaps.",
  };

  const handleAnalyze = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const effectiveQuestion =
      analysisType === "general"
        ? question.trim()
        : question.trim() || defaultQuestionForType[analysisType];
    if (!effectiveQuestion) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const req: AnalyzeRequest = {
        document_id: documentId,
        question: effectiveQuestion,
        analysis_type: analysisType,
      };
      const response = await analyzeDocument(req);
      if (response.success) {
        setResult(response);
      } else {
        setError(response.error || "Analysis failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#272436]/40 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-3xl max-h-[90vh] bg-white border border-[#EAE7EF] rounded-3xl shadow-card overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-5 py-4 border-b border-[#EAE7EF] bg-[#FAFAF4]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#F0EAFB] border border-[#D8CCF2] flex items-center justify-center">
              <Brain className="w-5 h-5 text-[#7454B8]" />
            </div>
            <div>
              <h2 className="font-bold text-[#272436]">
                Document Intelligence Inspector
              </h2>
              <p className="text-xs text-[#777383] truncate max-w-[320px]">
                {documentName}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-[#777383] hover:text-[#272436] hover:bg-white border border-transparent hover:border-[#EAE7EF]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {ANALYSIS_TYPES.map((type) => (
              <button
                key={type.value}
                onClick={() => setAnalysisType(type.value)}
                className={`p-3 rounded-xl border transition-all text-left ${
                  analysisType === type.value
                    ? "bg-[#F0EAFB] border-[#7454B8] text-[#4F3685]"
                    : "bg-[#FAFAF4] border-[#EAE7EF] text-[#272436] hover:border-[#D8CCF2]"
                }`}
              >
                <div className="flex items-center gap-2 text-xs font-bold">
                  <type.icon className="w-4 h-4 shrink-0" />
                  <span>{type.label}</span>
                </div>
                <p className="text-[11px] text-[#777383] mt-1">
                  {type.description}
                </p>
              </button>
            ))}
          </div>

          {analysisType === "general" && (
            <div>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleAnalyze();
                  }
                }}
                placeholder="Ask a specific question grounded in this document..."
                className="input-field min-h-[84px] resize-none"
                disabled={isLoading}
                rows={3}
              />
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={
              isLoading || (analysisType === "general" && !question.trim())
            }
            className="btn-primary w-full inline-flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing Grounded Passages...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                Run Grounded Analysis
              </>
            )}
          </button>

          {error && (
            <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-2 text-rose-700 text-xs">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="space-y-3 animate-fade-in">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-[#272436]">
                  Grounded Answer
                </h3>
                <span className="badge badge-lemon">
                  Retrieval Score: {Math.round(result.confidence * 100)}%
                </span>
              </div>
              <div className="bg-[#FAFAF4] border border-[#EAE7EF] rounded-2xl p-4 text-sm text-[#272436] whitespace-pre-wrap leading-relaxed max-h-80 overflow-y-auto">
                {result.answer}
              </div>
              {result.evidence && result.evidence.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs font-bold text-[#777383] uppercase">
                    Supporting Passages ({result.evidence.length})
                  </div>
                  {result.evidence.map((ev, i) => (
                    <div
                      key={i}
                      className="p-3 bg-white border border-[#EAE7EF] rounded-xl text-xs text-[#272436]"
                    >
                      {ev}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
