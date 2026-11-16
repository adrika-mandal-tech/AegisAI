"use client";

import { useState, useRef, useCallback, useMemo } from "react";
import {
  Upload,
  File as FileIcon,
  X,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Globe,
  Github,
  Files,
} from "lucide-react";
import { uploadDocument, batchUpload, ingestSource } from "@/lib/api";

interface UploadHistoryItem {
  filename: string;
  success: boolean;
  chunks: number;
  parser?: string;
  error?: string;
  fileSizeBytes?: number;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

const SUPPORTED_FORMATS = [
  ".pdf",
  ".docx",
  ".doc",
  ".pptx",
  ".xls",
  ".xlsx",
  ".csv",
  ".json",
  ".html",
  ".htm",
  ".txt",
  ".md",
  ".zip",
  ".png",
  ".jpg",
  ".jpeg",
  ".tiff",
  ".bmp",
  ".webp",
];

const MAX_SINGLE_FILE_BYTES = 500 * 1024 * 1024; // 500 MB per file
const MAX_BATCH_FILES = 100;
const MAX_BATCH_BYTES = 1024 * 1024 * 1024; // 1 GB per batch

export function FileUpload({ onUploaded }: { onUploaded: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceType, setSourceType] = useState<"url" | "repo">("url");
  const [activeTab, setActiveTab] = useState<"files" | "url" | "repo">("files");
  const [isUploading, setIsUploading] = useState(false);
  const [history, setHistory] = useState<UploadHistoryItem[]>([]);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const totalSelectedSize = useMemo(
    () => files.reduce((sum, f) => sum + f.size, 0),
    [files]
  );

  const addFiles = useCallback((newFiles: FileList | File[] | null) => {
    if (!newFiles || newFiles.length === 0) return;
    const valid: File[] = [];
    const oversized: string[] = [];
    for (const f of Array.from(newFiles)) {
      const ext = "." + f.name.split(".").pop()?.toLowerCase();
      if (f.size > MAX_SINGLE_FILE_BYTES) {
        oversized.push(`${f.name} (${formatBytes(f.size)})`);
        continue;
      }
      if (SUPPORTED_FORMATS.includes(ext)) {
        valid.push(f);
      }
    }
    setFiles((prev) => [...prev, ...valid].slice(0, MAX_BATCH_FILES));
    if (oversized.length > 0) {
      setMessage({
        type: "error",
        text: `File exceeds the 500 MB per-file limit: ${oversized.join(", ")}`,
      });
    } else {
      setMessage(null);
    }
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const clearFiles = useCallback(() => {
    setFiles([]);
    setHistory([]);
    setMessage(null);
  }, []);

  const handleFileUpload = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setMessage(null);
    const newHistory: UploadHistoryItem[] = [];

    try {
      if (files.length === 1) {
        const result = await uploadDocument(files[0]);
        newHistory.push({
          filename: files[0].name,
          success: result.success,
          chunks: result.chunks || 0,
          parser: result.parser_used,
          error: result.error,
          fileSizeBytes: files[0].size,
        });
        if (result.success && result.status === "processing") {
          setMessage({
            type: "success",
            text: `"${files[0].name}" (${formatBytes(files[0].size)}) accepted and indexing in the background.`,
          });
        } else if (result.success) {
          setMessage({
            type: "success",
            text: `"${files[0].name}" (${formatBytes(files[0].size)}) indexed (${result.chunks} chunks · stored permanently until deleted).`,
          });
        } else {
          setMessage({
            type: "error",
            text: `${result.error || "Upload failed"}`,
          });
        }
      } else {
        const result = await batchUpload(files);
        for (const item of result.results || []) {
          const sourceFile = files.find((f) => f.name === item.filename);
          newHistory.push({
            filename: item.filename,
            success: item.success,
            chunks: item.chunks,
            error: item.error,
            fileSizeBytes: sourceFile?.size || 0,
          });
        }
        const succeeded = result.successful || 0;
        const failed = result.failed || 0;
        const stillProcessing = (result.results || []).filter(
          (item) => item.status === "processing"
        ).length;

        if (failed === 0 && stillProcessing === 0) {
          setMessage({
            type: "success",
            text: `${succeeded} file(s) indexed (${result.total_chunks} chunks).`,
          });
        } else if (failed === 0) {
          setMessage({
            type: "success",
            text: `${succeeded} file(s) accepted (${stillProcessing} finishing in background).`,
          });
        } else {
          setMessage({
            type: "error",
            text: `${succeeded} succeeded, ${failed} failed.`,
          });
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Upload error";
      setMessage({ type: "error", text: errorMessage });
    }

    setHistory(newHistory);
    setFiles([]);
    setIsUploading(false);
    onUploaded();
  };

  const handleUrlIngest = async () => {
    let rawSource = sourceUrl.trim();
    if (!rawSource) return;
    if (
      !/^https?:\/\//i.test(rawSource) &&
      !rawSource.startsWith("git@") &&
      !rawSource.startsWith("ssh://")
    ) {
      rawSource = `https://${rawSource}`;
    }
    setIsUploading(true);
    setMessage(null);
    try {
      const result = await ingestSource(rawSource, sourceType);
      if (result.success) {
        setMessage({
          type: "success",
          text: `${sourceType === "url" ? "Web page" : "Git repository"} indexed (${result.chunks} chunks · stored permanently until deleted).`,
        });
        setHistory((prev) => [
          {
            filename: rawSource,
            success: true,
            chunks: result.chunks || 0,
          },
          ...prev.slice(0, 9),
        ]);
        setSourceUrl("");
        onUploaded();
      } else {
        setMessage({
          type: "error",
          text: `${result.error || "Ingestion failed"}`,
        });
      }
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "Ingestion error",
      });
    }
    setIsUploading(false);
  };

  return (
    <div className="w-full space-y-3">
      {/* Mode Switcher */}
      <div className="flex gap-1.5 p-1.5 bg-[#FAFAF4] rounded-2xl border border-[#EAE7EF]">
        <button
          onClick={() => setActiveTab("files")}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl transition-all ${
            activeTab === "files"
              ? "bg-white text-[#4F3685] border border-[#D8CCF2] shadow-sm"
              : "text-[#777383] hover:text-[#272436]"
          }`}
        >
          <Files className="w-3.5 h-3.5" />
          Multi-Format Files &amp; OCR
        </button>
        <button
          onClick={() => {
            setActiveTab("url");
            setSourceType("url");
          }}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl transition-all ${
            activeTab === "url"
              ? "bg-white text-[#4F3685] border border-[#D8CCF2] shadow-sm"
              : "text-[#777383] hover:text-[#272436]"
          }`}
        >
          <Globe className="w-3.5 h-3.5" />
          Web URL
        </button>
        <button
          onClick={() => {
            setActiveTab("repo");
            setSourceType("repo");
          }}
          className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl transition-all ${
            activeTab === "repo"
              ? "bg-white text-[#4F3685] border border-[#D8CCF2] shadow-sm"
              : "text-[#777383] hover:text-[#272436]"
          }`}
        >
          <Github className="w-3.5 h-3.5" />
          Git Repository
        </button>
      </div>

      {activeTab === "files" && (
        <>
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => inputRef.current?.click()}
            className={`flex flex-col items-center justify-center p-7 border-2 border-dashed rounded-2xl cursor-pointer transition-all ${
              files.length > 0
                ? "border-[#7454B8] bg-[#F0EAFB]/50"
                : "border-[#D8CCF2] hover:border-[#7454B8] bg-[#FAFAF4]"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={SUPPORTED_FORMATS.join(",")}
              multiple
              className="hidden"
              onChange={(e) => addFiles(e.target.files)}
            />
            <div className="w-12 h-12 rounded-2xl bg-[#FFF9CC] border border-[#E6D54F] flex items-center justify-center mb-3">
              <Upload className="w-6 h-6 text-[#4F3685]" />
            </div>
            <p className="text-sm font-bold text-[#272436]">
              {files.length > 0
                ? `${files.length} file(s) staged (${formatBytes(totalSelectedSize)} / ${formatBytes(MAX_BATCH_BYTES)} batch limit)`
                : "Drop documents, datasets, or scanned images here — or click to browse"}
            </p>
            <p className="text-xs text-[#777383] mt-1 text-center">
              PDF, Scanned PDF &amp; Images, DOCX, PPTX, XLSX, XLS, CSV, JSON, HTML, MD, TXT, ZIP
            </p>
            <p className="text-[11px] font-semibold text-[#4F3685] mt-1.5 text-center">
              Max file size: 500 MB per file · Up to 100 files (1 GB batch) · Stored permanently until deleted
            </p>
          </div>

          {files.length > 0 && (
            <div className="space-y-2">
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {files.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className="flex items-center justify-between bg-[#FAFAF4] rounded-xl p-2.5 border border-[#EAE7EF]"
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <FileIcon className="w-4 h-4 text-[#7454B8] shrink-0" />
                      <span className="text-xs font-semibold text-[#272436] truncate">
                        {file.name}
                      </span>
                      <span className="text-xs text-[#777383]">
                        ({formatBytes(file.size)})
                      </span>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 text-[#777383] hover:text-rose-600 rounded-md"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={handleFileUpload}
                  disabled={isUploading}
                  className="btn-primary flex-1 inline-flex items-center justify-center gap-2"
                >
                  {isUploading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Upload className="w-4 h-4" />
                  )}
                  {isUploading
                    ? "Uploading & Indexing..."
                    : `Upload & Index ${files.length} File(s)`}
                </button>
                <button onClick={clearFiles} className="btn-secondary">
                  Clear
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {(activeTab === "url" || activeTab === "repo") && (
        <div className="space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={sourceUrl}
              onChange={(e) => setSourceUrl(e.target.value)}
              placeholder={
                activeTab === "url"
                  ? "https://example.com/article-or-document.pdf"
                  : "https://github.com/organization/repository"
              }
              className="input-field flex-1"
              onKeyDown={(e) => e.key === "Enter" && handleUrlIngest()}
            />
            <button
              onClick={handleUrlIngest}
              disabled={isUploading || !sourceUrl.trim()}
              className="btn-primary inline-flex items-center gap-1.5 shrink-0"
            >
              {isUploading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : activeTab === "url" ? (
                <Globe className="w-4 h-4" />
              ) : (
                <Github className="w-4 h-4" />
              )}
              {isUploading ? "Ingesting..." : activeTab === "url" ? "Ingest URL" : "Clone & Index Repo"}
            </button>
          </div>
          <p className="text-[11px] text-[#777383]">
            {activeTab === "url"
              ? "Supports web articles, documentation pages, GitHub file links, and direct PDF URLs (up to 500 MB · stored permanently until deleted)."
              : "Clones public GitHub / GitLab repositories (README, docs, Python, TypeScript, configs, SQL · stored permanently until deleted)."}
          </p>
        </div>
      )}

      {message && (
        <div
          className={`flex items-start gap-2 text-xs p-3 rounded-xl border ${
            message.type === "error"
              ? "bg-rose-50 text-rose-700 border-rose-200"
              : "bg-emerald-50 text-emerald-800 border-emerald-200"
          }`}
        >
          {message.type === "error" ? (
            <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          ) : (
            <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
          )}
          <p className="font-medium">{message.text}</p>
        </div>
      )}

      {history.length > 0 && (
        <div className="space-y-1">
          {history.map((item, i) => (
            <div
              key={i}
              className={`flex items-center gap-2 text-xs px-3 py-2 rounded-xl border ${
                item.success
                  ? "bg-[#FAFAF4] text-[#272436] border-[#EAE7EF]"
                  : "bg-rose-50 text-rose-700 border-rose-200"
              }`}
            >
              {item.success ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
              )}
              <span className="font-semibold truncate">{item.filename}</span>
              {item.fileSizeBytes ? (
                <span className="badge badge-lemon ml-auto">
                  {formatBytes(item.fileSizeBytes)}
                </span>
              ) : null}
              {item.success && (
                <span className={`badge badge-purple ${item.fileSizeBytes ? "" : "ml-auto"}`}>
                  {item.chunks} chunks
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
