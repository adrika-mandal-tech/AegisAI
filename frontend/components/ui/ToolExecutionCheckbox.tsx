"use client";

import { useState } from "react";
import { Settings2, Check, X } from "lucide-react";

export interface ToolExecutionSettings {
  enableWebSearch: boolean;
  enableDocumentAnalysis: boolean;
  enableCodeExecution: boolean;
  enableExternalAPIs: boolean;
  maxToolCalls: number;
  toolTimeout: number;
}

interface ToolExecutionCheckboxProps {
  settings: ToolExecutionSettings;
  onSettingsChange: (settings: ToolExecutionSettings) => void;
  disabled?: boolean;
}

export function ToolExecutionCheckbox({
  settings,
  onSettingsChange,
  disabled = false,
}: ToolExecutionCheckboxProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const updateSetting = <K extends keyof ToolExecutionSettings>(
    key: K,
    value: ToolExecutionSettings[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  const toolOptions = [
    {
      key: "enableWebSearch" as const,
      label: "Web Research",
      description: "Enable multi-source web search",
    },
    {
      key: "enableDocumentAnalysis" as const,
      label: "Qdrant Retrieval",
      description: "Hybrid semantic + BM25 search",
    },
    {
      key: "enableCodeExecution" as const,
      label: "Dataset Computation",
      description: "Safe Python statistical profiling",
    },
    {
      key: "enableExternalAPIs" as const,
      label: "MCP Connectors",
      description: "Query configured MCP servers",
    },
  ];

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        disabled={disabled}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-colors border ${
          isExpanded
            ? "bg-[#F0EAFB] text-[#4F3685] border-[#7454B8]"
            : "bg-white text-[#272436] hover:bg-[#F0EAFB] border-[#EAE7EF]"
        }`}
      >
        <Settings2 className="w-3.5 h-3.5 text-[#7454B8]" />
        <span>Tools</span>
        {isExpanded ? <X className="w-3 h-3" /> : <Check className="w-3 h-3" />}
      </button>

      {isExpanded && (
        <div className="absolute right-0 top-full mt-2 w-72 z-50 bg-white border border-[#EAE7EF] rounded-2xl p-4 shadow-card animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-bold text-[#272436] uppercase tracking-wider">
              Controlled Tool Layer
            </h4>
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              className="p-1 rounded text-[#777383] hover:text-[#272436]"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-3">
            {toolOptions.map((option) => (
              <div
                key={option.key}
                className="flex items-center justify-between gap-2"
              >
                <div>
                  <div className="text-xs font-semibold text-[#272436]">
                    {option.label}
                  </div>
                  <div className="text-[11px] text-[#777383]">
                    {option.description}
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={settings[option.key]}
                  onChange={(e) => updateSetting(option.key, e.target.checked)}
                  disabled={disabled}
                  className="w-4 h-4 accent-[#7454B8] rounded cursor-pointer"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
