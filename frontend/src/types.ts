export interface ChunkMetadata {
  source: string
  page: number | null
  chunk_index: number
}

export interface RetrievedChunk {
  text: string
  metadata: ChunkMetadata
  score: number
}

export interface IngestResult {
  source: string
  file_type: 'pdf' | 'txt'
  pages: number | null
  chunks_created: number
  total_tokens: number
}

export interface ChatResponse {
  answer: string
  sources: RetrievedChunk[]
}

export interface DocumentSummary {
  source: string
  chunks: number
}

export interface HealthResponse {
  status: string
  model: string
  embedding_model: string
  chunks_indexed: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  sources?: RetrievedChunk[]
  isError?: boolean
}

export function sourceKey(c: RetrievedChunk): string {
  return `${c.metadata.source}::${c.metadata.page ?? '-'}::${c.metadata.chunk_index}`
}
