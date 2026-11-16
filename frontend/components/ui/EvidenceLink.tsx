"use client";

import { useState } from "react";
import { Search, FileText, ChevronDown, ChevronUp, X } from "lucide-react";
import { type Evidence } from "@/lib/api";

interface EvidenceLinkProps {
  evidence?: Evidence[];
  citations?: string[];
  className?: string;
}

export function EvidenceLink({
  evidence = [],
  citations = [],
  className = "",
}: EvidenceLinkProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (evidence.length === 0) return null;

  return (
    <div className={`relative inline-block ${className}`}>
      <button
        type="button"
        onClick={() => setIsExpanded((v) => !v)}
        className="flex items-center gap-2 px-3 py-1.5 bg-[#FFF9CC] border border-[#E6D54F] rounded-xl text-[#5C4900] text-xs font-semibold hover:bg-[#F3E66B] transition-colors"
      >
        <Search className="w-3.5 h-3.5" />
        <span>Evidence: {evidence.length}</span>
        {citations.length > 0 && <span>· {citations.length} citations</span>}
        {isExpanded ? (
          <ChevronUp className="w-3.5 h-3.5" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5" />
        )}
      </button>

      {isExpanded && (
        <div className="absolute left-0 top-full mt-2 w-96 z-50 bg-white border border-[#EAE7EF] rounded-2xl p-4 shadow-card animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-bold text-[#272436] flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-[#7454B8]" />
              Grounded Sources
            </h4>
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              className="text-[#777383] hover:text-[#272436]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {evidence.map((ev, idx) => (
              <div
                key={idx}
                className="p-2.5 bg-[#FAFAF4] border border-[#EAE7EF] rounded-xl text-xs"
              >
                <div className="font-bold text-[#4F3685] truncate">
                  {ev.source}
                </div>
                <p className="text-[#272436] mt-1">{ev.claim}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
