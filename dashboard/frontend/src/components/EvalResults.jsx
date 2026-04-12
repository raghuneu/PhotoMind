import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { gradeColor } from '../utils/chartHelpers'
import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'

export default function EvalResults({ evalData }) {
  if (!evalData || !evalData.results) {
    return (
      <div className="page-header">
        <h1>Evaluation Results</h1>
        <p>No evaluation data found. Run <code>python -m src.main eval</code> first.</p>
      </div>
    )
  }

  const { summary, results } = evalData

  const categoryBreakdown = useMemo(() => {
    const cats = {}
    results.forEach((r) => {
      if (!cats[r.category]) cats[r.category] = { total: 0, retrieval: 0, routing: 0 }
      cats[r.category].total++
      if (r.retrieval_correct) cats[r.category].retrieval++
      if (r.routing_correct) cats[r.category].routing++
    })
    return Object.entries(cats).map(([name, data]) => ({
      name,
      retrieval: Math.round((data.retrieval / data.total) * 100),
      routing: Math.round((data.routing / data.total) * 100),
      total: data.total,
    }))
  }, [results])

  const latencyData = results
    .filter((r) => r.latency_s > 0)
    .map((r) => ({
      query: r.query.length > 35 ? r.query.slice(0, 35) + '...' : r.query,
      latency: Math.round(r.latency_s * 10) / 10,
      category: r.category,
    }))

  return (
    <div>
      <div className="page-header">
        <h1>Evaluation Results</h1>
        <p>{results.length} test queries across 4 categories</p>
      </div>

      {/* Summary Stats */}
      <div className="stat-row">
        <SummaryCard
          label="Retrieval Accuracy"
          value={summary.retrieval_accuracy}
          good={summary.retrieval_accuracy >= 0.7}
        />
        <SummaryCard
          label="Routing Accuracy"
          value={summary.routing_accuracy}
          good={summary.routing_accuracy >= 0.7}
        />
        <SummaryCard
          label="Silent Failure"
          value={summary.silent_failure_rate}
          good={summary.silent_failure_rate < 0.1}
          invert
        />
        <SummaryCard
          label="Decline Accuracy"
          value={summary.decline_accuracy}
          good={summary.decline_accuracy >= 0.8}
        />
      </div>

      {/* Category Breakdown */}
      <div className="chart-row">
        <div className="card chart-card">
          <h3>Accuracy by Category</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 8 }}>
            {categoryBreakdown.map((cat) => (
              <div key={cat.name}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.875rem', fontWeight: 500, textTransform: 'capitalize' }}>
                    {cat.name.replace('_', ' ')} ({cat.total})
                  </span>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8125rem',
                    fontWeight: 600, color: cat.retrieval >= 70 ? '#22C55E' : '#F97316',
                  }}>
                    {cat.retrieval}%
                  </span>
                </div>
                <div style={{ height: 6, background: '#F1F0ED', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 3,
                    width: `${cat.retrieval}%`,
                    background: cat.retrieval >= 70 ? '#22C55E' : cat.retrieval >= 50 ? '#EAB308' : '#EF4444',
                    transition: 'width 0.5s ease',
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Latency Chart */}
        <div className="card chart-card">
          <h3>Query Latency</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={latencyData} margin={{ left: 0, right: 10 }}>
              <XAxis dataKey="query" hide />
              <YAxis
                tick={{ fontSize: 11, fill: '#6B7280' }}
                tickFormatter={(v) => `${v}s`}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff', border: '1px solid #E8E5E0',
                  borderRadius: 8, fontSize: 12,
                }}
                formatter={(val) => [`${val}s`, 'Latency']}
              />
              <Bar dataKey="latency" radius={[4, 4, 0, 0]} barSize={14}>
                {latencyData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.latency < 20 ? '#22C55E' : entry.latency < 60 ? '#EAB308' : '#EF4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{
            display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8,
            fontSize: '0.6875rem', color: '#9CA3AF',
          }}>
            <span><span style={{ color: '#22C55E' }}>--</span> &lt;20s</span>
            <span><span style={{ color: '#EAB308' }}>--</span> 20-60s</span>
            <span><span style={{ color: '#EF4444' }}>--</span> &gt;60s</span>
          </div>
        </div>
      </div>

      {/* Per-Query Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="eval-table-wrap">
          <table className="eval-table">
            <thead>
              <tr>
                <th>Query</th>
                <th>Category</th>
                <th>Retrieval</th>
                <th>Routing</th>
                <th>Grade</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i}>
                  <td style={{ maxWidth: 300 }}>{r.query}</td>
                  <td>
                    <span className={`badge badge-${r.category === 'edge_case' ? 'other' : r.category}`} style={{ textTransform: 'capitalize' }}>
                      {r.category.replace('_', ' ')}
                    </span>
                  </td>
                  <td>
                    {r.retrieval_correct
                      ? <CheckCircle2 size={16} className="check-yes" />
                      : <XCircle size={16} className="check-no" />}
                  </td>
                  <td>
                    {r.routing_correct
                      ? <CheckCircle2 size={16} className="check-yes" />
                      : <XCircle size={16} className="check-no" />}
                  </td>
                  <td>
                    {r.confidence_grade ? (
                      <span style={{
                        fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
                        color: gradeColor(r.confidence_grade),
                      }}>
                        {r.confidence_grade}
                      </span>
                    ) : '—'}
                  </td>
                  <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8125rem' }}>
                    {r.latency_s > 0 ? `${r.latency_s.toFixed(1)}s` : (
                      <span style={{ color: 'var(--error)', display: 'flex', alignItems: 'center', gap: 4 }}>
                        <AlertTriangle size={13} /> Error
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Avg Latency */}
      <div style={{
        textAlign: 'center', marginTop: 20, fontSize: '0.8125rem', color: '#9CA3AF',
      }}>
        Average latency: {summary.avg_latency_s?.toFixed(1)}s across {summary.total_queries} queries
      </div>
    </div>
  )
}

function SummaryCard({ label, value, good, invert = false }) {
  const pct = (value * 100).toFixed(0)
  const color = good ? '#22C55E' : '#F97316'
  return (
    <div className="card stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={{ color }}>{pct}%</span>
    </div>
  )
}
