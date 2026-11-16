"use client";

import { useEffect, useState, useCallback } from "react";
import {
  FileText,
  Trash2,
  RefreshCw,
  Loader2,
  AlertCircle,
  Search,
  Brain,
  Eye,
  X,
  BarChart3,
} from "lucide-react";
import {
  listDocuments,
  deleteDocument,
  getDocumentChunks,
  formatCleanFilename,
  type DocItem,
  type ChunkMeta,
} from "@/lib/api";
import { DocumentAnalysis } from "./DocumentAnalysis";

function formatBytes(bytes: number) {
  if (!bytes) return "—";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

interface DocumentListProps {
  refreshKey?: number;
  onUploaded?: () => void;
  onRunPipeline?: (doc: DocItem) => void;
  onProfileDataset?: (doc: DocItem) => void;
}

export function DocumentList({
  refreshKey = 0,
  onUploaded,
  onRunPipeline,
  onProfileDataset,
}: DocumentListProps = {}) {
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyzingDoc, setAnalyzingDoc] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [viewingChunksDoc, setViewingChunksDoc] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [chunks, setChunks] = useState<ChunkMeta[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listDocuments();
      if (res.success) {
        setDocs(res.documents);
      } else {
        setError(res.error || "Failed to load documents");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs, refreshKey]);

  const handleDelete = async (id: string) => {
    await deleteDocument(id);
    fetchDocs();
    onUploaded?.();
  };

  const handleViewChunks = async (doc: DocItem) => {
    setViewingChunksDoc({ id: doc.id, name: doc.filename });
    setChunksLoading(true);
    setChunks([]);
    try {
      const res = await getDocumentChunks(doc.id, 1, 100);
      if (res.success) {
        setChunks(res.chunks);
      } else {
        setError(res.error || "Failed to load chunks");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load chunks");
    } finally {
      setChunksLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-bold text-[#272436]">
            Indexed Document &amp; Dataset Registry
          </h3>
          <p className="text-xs text-[#777383]">
            Stored permanently until deleted · Max 500 MB per file · Inspect chunks, run 8-agent analysis, or profile datasets.
          </p>
        </div>
        <button
          onClick={fetchDocs}
          disabled={loading}
          className="btn-secondary inline-flex items-center gap-1.5 text-xs"
        >
          {loading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <RefreshCw className="w-3.5 h-3.5" />
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

      {docs.length === 0 && !loading && (
        <div className="text-center py-10 rounded-2xl bg-[#FAFAF4] border border-dashed border-[#D8CCF2] text-xs text-[#777383]">
          <FileText className="w-9 h-9 mx-auto mb-2 text-[#7454B8] opacity-60" />
          No documents indexed yet. Upload PDFs, DOCX, PPTX, scanned images, or CSV/XLSX datasets above.
        </div>
      )}

      <div className="space-y-2">
        {docs.map((doc) => {
          const isReady = doc.status === "ready" || doc.status === "success";
          const isTabular =
            doc.is_tabular ||
            /\.(csv|xlsx|xls|json)$/i.test(doc.filename);

          return (
            <div
              key={doc.id}
              className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#FAFAF4] p-3.5 rounded-2xl border border-[#EAE7EF] hover:border-[#D8CCF2] transition-colors"
            >
              <div className="flex items-center gap-3 overflow-hidden">
                <div className="w-9 h-9 rounded-xl bg-white border border-[#EAE7EF] flex items-center justify-center shrink-0">
                  <FileText className="w-4 h-4 text-[#7454B8]" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-bold text-[#272436] truncate" title={formatCleanFilename(doc.filename)}>
                    {formatCleanFilename(doc.filename)}
                  </p>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-[#777383]">
                    <span>{formatBytes(doc.file_size_bytes)}</span>
                    <span>·</span>
                    <span className="font-semibold text-[#4F3685]">
                      {doc.chunks} chunks
                    </span>
                    <span>·</span>
                    <span className="text-[11px] font-medium text-emerald-700">
                      {doc.source_type === "repo"
                        ? "Git Repository"
                        : doc.source_type === "url"
                        ? "Web URL"
                        : "Saved until deleted"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                <button
                  onClick={() => handleViewChunks(doc)}
                  disabled={!isReady}
                  className="px-2.5 py-1.5 text-xs font-semibold text-[#272436] bg-white border border-[#EAE7EF] hover:border-[#7454B8] hover:text-[#4F3685] rounded-lg disabled:opacity-45 inline-flex items-center gap-1"
                  title="Inspect stored Qdrant chunks"
                >
                  <Eye className="w-3.5 h-3.5" />
                  Chunks
                </button>

                {isTabular && onProfileDataset && (
                  <button
                    onClick={() => onProfileDataset(doc)}
                    disabled={!isReady}
                    className="px-2.5 py-1.5 text-xs font-semibold bg-[#FFF9CC] text-[#5C4900] border border-[#E6D54F] hover:bg-[#F3E66B] rounded-lg disabled:opacity-45 inline-flex items-center gap-1"
                    title="Profile dataset statistics & charts"
                  >
                    <BarChart3 className="w-3.5 h-3.5" />
                    Profile Data
                  </button>
                )}

                {onRunPipeline && (
                  <button
                    onClick={() => onRunPipeline(doc)}
                    disabled={!isReady}
                    className="px-2.5 py-1.5 text-xs font-semibold bg-[#F0EAFB] text-[#4F3685] border border-[#D8CCF2] hover:bg-[#7454B8] hover:text-white rounded-lg disabled:opacity-45 inline-flex items-center gap-1 transition-colors"
                    title="Run 8-agent swarm scoped to this document"
                  >
                    <Search className="w-3.5 h-3.5" />
                    Run Swarm
                  </button>
                )}

                <button
                  onClick={() =>
                    setAnalyzingDoc({ id: doc.id, name: formatCleanFilename(doc.filename) })
                  }
                  disabled={!isReady}
                  className="px-2.5 py-1.5 text-xs font-semibold text-[#272436] bg-white border border-[#EAE7EF] hover:border-[#7454B8] rounded-lg disabled:opacity-45 inline-flex items-center gap-1"
                  title="Targeted Q&A / Risk / Contradiction analysis"
                >
                  <Brain className="w-3.5 h-3.5 text-[#7454B8]" />
                  Q&amp;A
                </button>

                <span
                  className={`badge ${isReady
                    ? "badge-green"
                    : doc.status === "processing"
                      ? "badge-lemon"
                      : "badge-rose"
                    }`}
                >
                  {doc.status}
                </span>

                <button
                  onClick={() => handleDelete(doc.id)}
                  className="p-1.5 text-[#777383] hover:text-rose-600 hover:bg-rose-50 rounded-lg"
                  title="Delete from registry & Qdrant"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Chunks Viewer Modal */}
      {viewingChunksDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#272436]/40 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-4xl max-h-[88vh] bg-white border border-[#EAE7EF] rounded-3xl shadow-card overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#EAE7EF] bg-[#FAFAF4]">
              <div>
                <h3 className="font-bold text-[#272436]">
                  Stored Qdrant Chunks &amp; Provenance
                </h3>
                <p className="text-xs text-[#777383] truncate max-w-[460px]">
                  {viewingChunksDoc.name}
                </p>
              </div>
              <button
                onClick={() => setViewingChunksDoc(null)}
                className="p-2 rounded-xl text-[#777383] hover:text-[#272436] hover:bg-white border border-transparent hover:border-[#EAE7EF]"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-3">
              {chunksLoading && (
                <div className="flex items-center justify-center py-12 text-[#777383] text-sm">
                  <Loader2 className="w-5 h-5 animate-spin mr-2 text-[#7454B8]" />
                  Loading chunks from Qdrant...
                </div>
              )}
              {!chunksLoading && chunks.length === 0 && (
                <div className="text-center py-12 text-xs text-[#777383]">
                  No chunks found for this document.
                </div>
              )}
              {chunks.map((chunk, idx) => (
                <div
                  key={chunk.content_hash + idx}
                  className="p-4 bg-[#FAFAF4] border border-[#EAE7EF] rounded-2xl"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <span className="badge badge-purple">
                      Chunk #{chunk.chunk_index + 1}
                    </span>
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#777383]">
                      <span>{chunk.word_count || chunk.token_count} words</span>
                      {chunk.page_number != null && (
                        <span>· Page {chunk.page_number}</span>
                      )}
                      {chunk.section_path && (
                        <span>· {chunk.section_path}</span>
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-[#272436] whitespace-pre-wrap leading-relaxed">
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {analyzingDoc && (
        <DocumentAnalysis
          documentId={analyzingDoc.id}
          documentName={analyzingDoc.name}
          onClose={() => setAnalyzingDoc(null)}
        />
      )}
    </div>
  );
}
