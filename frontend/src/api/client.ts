import type {
  ChatResponse,
  DocumentSummary,
  HealthResponse,
  IngestResult,
} from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body?.detail) detail = body.detail
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`)
  return unwrap<HealthResponse>(res)
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const res = await fetch(`${API_BASE}/documents`)
  const body = await unwrap<{ documents: DocumentSummary[] }>(res)
  return body.documents
}

export async function clearDocuments(): Promise<void> {
  const res = await fetch(`${API_BASE}/documents`, { method: 'DELETE' })
  await unwrap<{ status: string }>(res)
}

export async function uploadFile(file: File): Promise<IngestResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: form,
  })
  return unwrap<IngestResult>(res)
}

export async function askQuestion(
  question: string,
  topK?: number,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
  return unwrap<ChatResponse>(res)
}
