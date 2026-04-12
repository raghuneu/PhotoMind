import { useState } from 'react'
import { Send, Zap, Brain, Loader2 } from 'lucide-react'
import { quickQuery, crewQuery, photoUrl } from '../utils/api'
import { gradeColor } from '../utils/chartHelpers'

const SAMPLE_QUERIES = [
  'How much did I spend at ALDI?',
  'Show me photos of pizza',
  'What type of food do I photograph most?',
  'What items did I buy at Trader Joe\'s?',
  'How many receipts do I have?',
  'Find photos of beer or drinks',
]

export default function QueryInterface() {
  const [input, setInput] = useState('')
  const [mode, setMode] = useState('quick') // 'quick' or 'deep'
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function handleSubmit(queryText) {
    const query = (queryText || input).trim()
    if (!query || loading) return

    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: query }])
    setLoading(true)

    try {
      if (mode === 'quick') {
        const data = await quickQuery(query)
        setMessages((prev) => [...prev, { role: 'system', type: 'quick', data }])
      } else {
        const data = await crewQuery(query)
        setMessages((prev) => [...prev, { role: 'system', type: 'deep', data }])
      }
    } catch (err) {
      setMessages((prev) => [...prev, {
        role: 'system', type: 'error',
        data: { error: err.message || 'Failed to get a response.' },
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="query-container">
      <div className="page-header">
        <h1>Query Your Photos</h1>
        <p>Ask anything about your photo collection</p>
      </div>

      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === 'quick' ? 'active' : ''}`}
          onClick={() => setMode('quick')}
        >
          <Zap size={14} style={{ marginRight: 4, verticalAlign: -2 }} />
          Quick Search
        </button>
        <button
          className={`mode-btn ${mode === 'deep' ? 'active' : ''}`}
          onClick={() => setMode('deep')}
        >
          <Brain size={14} style={{ marginRight: 4, verticalAlign: -2 }} />
          Deep Analysis
        </button>
      </div>

      {mode === 'deep' && (
        <div style={{
          padding: '8px 14px', marginBottom: 12, borderRadius: 8,
          background: 'rgba(232, 168, 56, 0.08)', fontSize: '0.8125rem',
          color: '#c68a20',
        }}>
          Deep Analysis uses the full CrewAI pipeline with LLM agents. It takes 15-90 seconds and uses API credits.
        </div>
      )}

      {/* Sample Queries */}
      {messages.length === 0 && (
        <div className="sample-queries">
          {SAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              className="sample-chip"
              onClick={() => handleSubmit(q)}
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Messages */}
      <div className="query-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`query-msg ${msg.role}`}>
            {msg.role === 'user' ? (
              <span style={{ fontWeight: 500 }}>{msg.content}</span>
            ) : msg.type === 'error' ? (
              <div style={{ color: 'var(--error)' }}>{msg.data.error}</div>
            ) : msg.type === 'quick' ? (
              <QuickResult data={msg.data} />
            ) : (
              <DeepResult data={msg.data} />
            )}
          </div>
        ))}

        {loading && (
          <div className="query-msg system" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Loader2 size={18} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
            <span style={{ color: 'var(--text-secondary)' }}>
              {mode === 'quick' ? 'Searching...' : 'Running agent pipeline...'}
            </span>
          </div>
        )}
      </div>

      {/* Input Bar */}
      <div className="query-input-row" style={{ marginTop: 20, position: 'sticky', bottom: 20 }}>
        <input
          className="query-input"
          type="text"
          placeholder="Ask about your photos..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          disabled={loading}
        />
        <button
          className="query-btn"
          onClick={() => handleSubmit()}
          disabled={!input.trim() || loading}
        >
          <Send size={16} style={{ verticalAlign: -2 }} />
        </button>
      </div>
    </div>
  )
}

function QuickResult({ data }) {
  const grade = data.confidence_grade
  const results = data.results || []
  return (
    <div>
      {/* Header badges */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        {grade && (
          <span style={{
            padding: '3px 10px', borderRadius: 100, fontWeight: 600,
            fontSize: '0.75rem', background: gradeColor(grade) + '18',
            color: gradeColor(grade),
          }}>
            Grade {grade}
          </span>
        )}
        {data.query_type_detected && (
          <span className="badge" style={{ background: '#F1F0ED', color: '#6B7280' }}>
            {data.query_type_detected}
          </span>
        )}
      </div>

      {/* Summary */}
      {data.answer_summary && (
        <p style={{ fontSize: '0.9375rem', lineHeight: 1.65, marginBottom: 12 }}>
          {data.answer_summary}
        </p>
      )}

      {/* Warning */}
      {data.warning && (
        <div style={{
          padding: '8px 12px', borderRadius: 8, marginBottom: 12,
          background: 'rgba(234, 179, 8, 0.08)', fontSize: '0.8125rem',
          color: '#a16207',
        }}>
          {data.warning}
        </div>
      )}

      {/* Photo results */}
      {results.length > 0 && (
        <div style={{ display: 'flex', gap: 10, overflowX: 'auto', paddingBottom: 4 }}>
          {results.map((r) => {
            const filename = r.photo_path?.split('/').pop()
            return (
              <div key={r.photo_id} style={{ minWidth: 130, flex: '0 0 auto' }}>
                <img
                  src={photoUrl(filename)}
                  alt={filename}
                  style={{
                    width: 130, height: 90, objectFit: 'cover',
                    borderRadius: 6, background: '#F1F0ED', display: 'block',
                  }}
                  loading="lazy"
                />
                <div style={{ fontSize: '0.6875rem', color: '#6B7280', marginTop: 4 }}>
                  {filename}
                </div>
                <div style={{
                  fontSize: '0.6875rem', fontFamily: "'JetBrains Mono', monospace",
                  color: gradeColor(data.confidence_grade || 'F'),
                }}>
                  {(r.relevance_score * 100).toFixed(0)}% match
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function DeepResult({ data }) {
  return (
    <div>
      <p style={{ fontSize: '0.9375rem', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
        {data.result || 'No response received.'}
      </p>
    </div>
  )
}
