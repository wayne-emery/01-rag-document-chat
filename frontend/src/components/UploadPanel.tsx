import { useCallback, useRef, useState } from 'react'
import { clearDocuments, uploadFile } from '../api/client'
import type { DocumentSummary, IngestResult } from '../types'

interface Props {
  documents: DocumentSummary[]
  onChanged: () => void | Promise<void>
}

const ACCEPTED_EXTS = ['.pdf', '.txt', '.md']
const ACCEPT_ATTR = ACCEPTED_EXTS.join(',')
const MAX_BYTES = 10 * 1024 * 1024

type Status =
  | { kind: 'idle' }
  | { kind: 'uploading'; filename: string }
  | { kind: 'success'; result: IngestResult }
  | { kind: 'error'; message: string }

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

function isAcceptedFile(file: File): boolean {
  const name = file.name.toLowerCase()
  return ACCEPTED_EXTS.some((ext) => name.endsWith(ext))
}

export function UploadPanel({ documents, onChanged }: Props) {
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [isDragging, setIsDragging] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    async (file: File) => {
      if (!isAcceptedFile(file)) {
        setStatus({
          kind: 'error',
          message: `Unsupported file type. Allowed: ${ACCEPTED_EXTS.join(', ')}`,
        })
        return
      }
      if (file.size > MAX_BYTES) {
        setStatus({
          kind: 'error',
          message: `File is ${formatBytes(file.size)}; max is 10 MB.`,
        })
        return
      }

      setStatus({ kind: 'uploading', filename: file.name })
      try {
        const result = await uploadFile(file)
        setStatus({ kind: 'success', result })
        await onChanged()
      } catch (err) {
        setStatus({
          kind: 'error',
          message: err instanceof Error ? err.message : String(err),
        })
      }
    },
    [onChanged],
  )

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files?.[0]
      if (file) void handleFile(file)
    },
    [handleFile],
  )

  const onSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) void handleFile(file)
      e.target.value = ''
    },
    [handleFile],
  )

  const onClearAll = useCallback(async () => {
    if (!confirm('Remove all uploaded documents?')) return
    setIsClearing(true)
    try {
      await clearDocuments()
      setStatus({ kind: 'idle' })
      await onChanged()
    } catch (err) {
      setStatus({
        kind: 'error',
        message: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setIsClearing(false)
    }
  }, [onChanged])

  const isUploading = status.kind === 'uploading'

  const dropzoneClass = [
    'group relative flex cursor-pointer flex-col items-center justify-center',
    'rounded-lg border-2 border-dashed p-6 text-center text-sm transition',
    isUploading
      ? 'cursor-wait border-slate-300 bg-slate-100 text-slate-500'
      : isDragging
        ? 'border-slate-900 bg-slate-100 text-slate-900'
        : 'border-slate-300 bg-slate-50 text-slate-500 hover:border-slate-400 hover:bg-slate-100 hover:text-slate-700',
  ].join(' ')

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Documents
        </h2>
        {documents.length > 0 && (
          <button
            type="button"
            onClick={onClearAll}
            disabled={isClearing || isUploading}
            className="text-xs text-slate-400 underline-offset-2 hover:text-red-600 hover:underline disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isClearing ? 'Clearing…' : 'Clear all'}
          </button>
        )}
      </div>

      <div
        className={dropzoneClass}
        onClick={() => !isUploading && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          if (!isUploading) setIsDragging(true)
        }}
        onDragLeave={(e) => {
          e.preventDefault()
          setIsDragging(false)
        }}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload file"
        aria-busy={isUploading}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          onChange={onSelect}
          className="hidden"
          disabled={isUploading}
        />
        {isUploading ? (
          <>
            <Spinner />
            <p className="mt-2 truncate font-medium text-slate-700">
              Uploading {status.filename}…
            </p>
            <p className="mt-1 text-xs text-slate-400">Parsing, chunking, embedding</p>
          </>
        ) : (
          <>
            <UploadIcon />
            <p className="mt-2 font-medium text-slate-700">
              Drop a file here, or <span className="underline">browse</span>
            </p>
            <p className="mt-1 text-xs text-slate-400">
              PDF, TXT, MD · 10 MB max
            </p>
          </>
        )}
      </div>

      {status.kind === 'success' && (
        <StatusBanner kind="success" onDismiss={() => setStatus({ kind: 'idle' })}>
          Ingested <span className="font-medium">{status.result.source}</span> —{' '}
          {status.result.chunks_created} chunks
          {status.result.pages !== null ? `, ${status.result.pages} pages` : ''}.
        </StatusBanner>
      )}

      {status.kind === 'error' && (
        <StatusBanner kind="error" onDismiss={() => setStatus({ kind: 'idle' })}>
          {status.message}
        </StatusBanner>
      )}

      {documents.length === 0 ? (
        <p className="text-sm text-slate-400">No documents uploaded yet.</p>
      ) : (
        <ul className="space-y-1 text-sm">
          {documents.map((d) => (
            <li
              key={d.source}
              className="flex items-center justify-between rounded border border-slate-200 bg-white px-3 py-2"
            >
              <span className="truncate text-slate-700" title={d.source}>
                {d.source}
              </span>
              <span className="ml-2 shrink-0 text-xs text-slate-500">
                {d.chunks} chunks
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function StatusBanner({
  kind,
  children,
  onDismiss,
}: {
  kind: 'success' | 'error'
  children: React.ReactNode
  onDismiss: () => void
}) {
  const styles =
    kind === 'success'
      ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
      : 'border-red-200 bg-red-50 text-red-800'
  return (
    <div className={`flex items-start justify-between gap-2 rounded border p-2 text-sm ${styles}`}>
      <p className="flex-1">{children}</p>
      <button
        type="button"
        onClick={onDismiss}
        className="shrink-0 text-xs opacity-60 hover:opacity-100"
        aria-label="Dismiss"
      >
        ✕
      </button>
    </div>
  )
}

function UploadIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-7 w-7"
      aria-hidden
    >
      <path d="M12 16V4" />
      <path d="m6 10 6-6 6 6" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  )
}

function Spinner() {
  return (
    <svg
      className="h-6 w-6 animate-spin text-slate-500"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z"
      />
    </svg>
  )
}
