import { useEffect, useRef } from 'react'
import { sourceKey, type RetrievedChunk } from '../types'

interface Props {
  sources: RetrievedChunk[]
  highlightedKey: string | null
}

function scoreColor(score: number): string {
  if (score >= 0.5) return 'bg-emerald-100 text-emerald-800 ring-emerald-200'
  if (score >= 0.25) return 'bg-amber-100 text-amber-800 ring-amber-200'
  return 'bg-slate-100 text-slate-600 ring-slate-200'
}

function scoreLabel(score: number): string {
  if (score >= 0.5) return 'strong match'
  if (score >= 0.25) return 'partial match'
  return 'weak match'
}

export function SourcesPanel({ sources, highlightedKey }: Props) {
  const refs = useRef<Map<string, HTMLLIElement>>(new Map())

  useEffect(() => {
    if (!highlightedKey) return
    const el = refs.current.get(highlightedKey)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [highlightedKey])

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Sources
        </h2>
        {sources.length > 0 && (
          <span className="text-xs text-slate-400">
            {sources.length} retrieved
          </span>
        )}
      </div>

      {sources.length === 0 ? (
        <div className="rounded-lg border border-dashed border-slate-200 bg-white p-4 text-center text-xs text-slate-400">
          <p className="font-medium text-slate-500">No sources yet</p>
          <p className="mt-1">
            After you ask a question, the chunks the model used to answer will
            appear here, ranked by similarity.
          </p>
        </div>
      ) : (
        <ul className="space-y-2">
          {sources.map((s, i) => {
            const key = sourceKey(s)
            const isHighlighted = key === highlightedKey
            return (
              <li
                key={key}
                ref={(el) => {
                  if (el) refs.current.set(key, el)
                  else refs.current.delete(key)
                }}
                className={[
                  'rounded border border-slate-200 bg-white p-3 transition-shadow',
                  isHighlighted ? 'animate-flash' : '',
                ].join(' ')}
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-900 text-[10px] font-semibold text-white">
                      {i + 1}
                    </span>
                    <span
                      className="truncate text-xs font-medium text-slate-700"
                      title={s.metadata.source}
                    >
                      {s.metadata.source}
                      {s.metadata.page !== null ? ` · p.${s.metadata.page}` : ''}
                    </span>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ${scoreColor(
                      s.score,
                    )}`}
                    title={scoreLabel(s.score)}
                  >
                    {s.score.toFixed(3)}
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-slate-600">{s.text}</p>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
