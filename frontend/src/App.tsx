import { useCallback, useEffect, useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { SourcesPanel } from './components/SourcesPanel'
import { UploadPanel } from './components/UploadPanel'
import { askQuestion, getHealth, listDocuments } from './api/client'
import {
  sourceKey,
  type ChatMessage,
  type DocumentSummary,
  type HealthResponse,
  type RetrievedChunk,
} from './types'

function makeId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeSources, setActiveSources] = useState<RetrievedChunk[]>([])
  const [highlightedKey, setHighlightedKey] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refreshAll = useCallback(async () => {
    try {
      const [h, docs] = await Promise.all([getHealth(), listDocuments()])
      setHealth(h)
      setHealthError(null)
      setDocuments(docs)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setHealthError(message)
    }
  }, [])

  useEffect(() => {
    void refreshAll()
  }, [refreshAll])

  const handleAsk = useCallback(
    async (question: string) => {
      const userMessage: ChatMessage = {
        id: makeId(),
        role: 'user',
        text: question,
      }
      setMessages((prev) => [...prev, userMessage])
      setBusy(true)
      try {
        const resp = await askQuestion(question)
        const assistantMessage: ChatMessage = {
          id: makeId(),
          role: 'assistant',
          text: resp.answer,
          sources: resp.sources,
        }
        setMessages((prev) => [...prev, assistantMessage])
        setActiveSources(resp.sources)
        setHighlightedKey(null)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: 'assistant',
            text: `Error: ${message}`,
            isError: true,
          },
        ])
      } finally {
        setBusy(false)
        void refreshAll()
      }
    },
    [refreshAll],
  )

  const handleSelectSource = useCallback(
    (source: RetrievedChunk) => {
      const key = sourceKey(source)
      setActiveSources((prev) =>
        prev.some((c) => sourceKey(c) === key) ? prev : [source, ...prev],
      )
      setHighlightedKey(null)
      window.requestAnimationFrame(() => setHighlightedKey(key))
    },
    [],
  )

  const noDocuments = documents.length === 0
  const disabledReason = noDocuments
    ? 'Drop a PDF, TXT, or MD file in the left panel.'
    : undefined

  return (
    <div className="flex h-full flex-col bg-slate-50 text-slate-900">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
        <div>
          <h1 className="text-lg font-semibold">RAG Document Chat</h1>
          <p className="text-xs text-slate-500">
            Upload documents, then ask grounded questions with source citations.
          </p>
        </div>
        <div className="text-right text-xs text-slate-500">
          {healthError ? (
            <span className="text-red-600">backend unreachable: {healthError}</span>
          ) : health ? (
            <>
              <div>
                <span className="font-medium text-slate-700">{health.model}</span>
                {' · '}
                <span>{health.chunks_indexed} chunks indexed</span>
              </div>
              <div className="text-[10px] text-slate-400">
                embeddings: {health.embedding_model}
              </div>
            </>
          ) : (
            <span>connecting…</span>
          )}
        </div>
      </header>

      <main className="grid flex-1 grid-cols-12 gap-4 overflow-hidden p-4">
        <aside className="col-span-3 overflow-y-auto">
          <UploadPanel documents={documents} onChanged={refreshAll} />
        </aside>

        <section className="col-span-6 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <ChatPanel
            messages={messages}
            busy={busy}
            disabled={noDocuments}
            disabledReason={disabledReason}
            onAsk={handleAsk}
            onSelectSource={handleSelectSource}
          />
        </section>

        <aside className="col-span-3 overflow-y-auto">
          <SourcesPanel sources={activeSources} highlightedKey={highlightedKey} />
        </aside>
      </main>
    </div>
  )
}

export default App
