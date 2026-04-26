import { useEffect, useRef, useState } from 'react'
import { sourceKey, type ChatMessage, type RetrievedChunk } from '../types'

interface Props {
  messages: ChatMessage[]
  busy: boolean
  disabled: boolean
  disabledReason?: string
  onAsk: (question: string) => void
  onSelectSource?: (source: RetrievedChunk) => void
}

export function ChatPanel({
  messages,
  busy,
  disabled,
  disabledReason,
  onAsk,
  onSelectSource,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages, busy])

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && !busy ? (
          <EmptyState disabled={disabled} disabledReason={disabledReason} />
        ) : (
          <ul className="space-y-3">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} onSelectSource={onSelectSource} />
            ))}
            {busy && <TypingIndicator />}
          </ul>
        )}
      </div>

      <ChatInput
        busy={busy}
        disabled={disabled}
        disabledReason={disabledReason}
        onAsk={onAsk}
      />
    </div>
  )
}

function EmptyState({
  disabled,
  disabledReason,
}: {
  disabled: boolean
  disabledReason?: string
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center text-sm text-slate-400">
      {disabled ? (
        <>
          <p className="font-medium text-slate-500">Upload a document to begin</p>
          {disabledReason && <p className="mt-1 text-xs">{disabledReason}</p>}
        </>
      ) : (
        <>
          <p className="font-medium text-slate-500">Ask a question</p>
          <p className="mt-1 text-xs">Answers are grounded in your uploaded documents.</p>
        </>
      )}
    </div>
  )
}

function MessageBubble({
  message,
  onSelectSource,
}: {
  message: ChatMessage
  onSelectSource?: (source: RetrievedChunk) => void
}) {
  if (message.role === 'user') {
    return (
      <li className="ml-auto max-w-[85%] rounded-lg bg-slate-900 px-4 py-2 text-sm text-white">
        <span className="whitespace-pre-wrap">{message.text}</span>
      </li>
    )
  }

  const cls = message.isError
    ? 'mr-auto max-w-[85%] rounded-lg bg-red-50 px-4 py-2 text-sm text-red-800 ring-1 ring-red-200'
    : 'mr-auto max-w-[85%] rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-800'

  return (
    <li className={cls}>
      <div className="whitespace-pre-wrap">{message.text}</div>
      {message.sources && message.sources.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 border-t border-slate-200 pt-2">
          {message.sources.map((s, i) => (
            <button
              key={sourceKey(s)}
              type="button"
              onClick={() => onSelectSource?.(s)}
              className="rounded-full border border-slate-300 bg-white px-2 py-0.5 text-[11px] text-slate-600 hover:border-slate-500 hover:text-slate-900"
              title={s.text}
            >
              [{i + 1}] {s.metadata.source}
              {s.metadata.page !== null ? ` p.${s.metadata.page}` : ''}
            </button>
          ))}
        </div>
      )}
    </li>
  )
}

function TypingIndicator() {
  return (
    <li className="mr-auto rounded-lg bg-slate-100 px-4 py-2">
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
      </span>
    </li>
  )
}

function ChatInput({
  busy,
  disabled,
  disabledReason,
  onAsk,
}: {
  busy: boolean
  disabled: boolean
  disabledReason?: string
  onAsk: (question: string) => void
}) {
  const [text, setText] = useState('')
  const inputDisabled = busy || disabled
  const trimmed = text.trim()
  const canSend = !inputDisabled && trimmed.length > 0

  const submit = () => {
    if (!canSend) return
    onAsk(trimmed)
    setText('')
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form
      className="flex items-end gap-2 border-t border-slate-200 bg-white p-3"
      onSubmit={(e) => {
        e.preventDefault()
        submit()
      }}
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
        disabled={inputDisabled}
        placeholder={
          disabled
            ? (disabledReason ?? 'Upload a document first…')
            : 'Ask a question (Enter to send, Shift+Enter for newline)'
        }
        className="max-h-40 min-h-[40px] flex-1 resize-y rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-500 focus:outline-none disabled:cursor-not-allowed disabled:bg-slate-50"
      />
      <button
        type="submit"
        disabled={!canSend}
        className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {busy ? 'Sending…' : 'Send'}
      </button>
    </form>
  )
}
