import { Camera, FileText, Clock, TrendingUp, Store } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell,
} from 'recharts'
import { photoUrl } from '../utils/api'
import {
  getTypeDistribution, getConfidenceData, getVendorDistribution, getTypeColor,
} from '../utils/chartHelpers'

export default function DashboardHome({ kb, evalData }) {
  if (!kb || !kb.photos) {
    return <div className="page-header"><h1>No data loaded</h1></div>
  }

  const photos = kb.photos
  const typeDist = getTypeDistribution(photos)
  const confData = getConfidenceData(photos)
  const vendors = getVendorDistribution(photos)
  const avgConf = photos.reduce((s, p) => s + (p.confidence || 0), 0) / photos.length
  const recentPhotos = [...photos].sort(
    (a, b) => new Date(b.indexed_at) - new Date(a.indexed_at)
  ).slice(0, 6)

  const dominantType = typeDist[0]

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Overview of your PhotoMind knowledge base</p>
      </div>

      {/* Stat Cards */}
      <div className="stat-row">
        <div className="card stat-card">
          <span className="stat-label"><Camera size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Total Photos</span>
          <span className="stat-value">{photos.length}</span>
          <span className="stat-sub">{kb.metadata?.total_photos || photos.length} indexed</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label"><FileText size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Dominant Type</span>
          <span className="stat-value" style={{ color: getTypeColor(dominantType?.name) }}>
            {dominantType?.name || '—'}
          </span>
          <span className="stat-sub">{dominantType?.value || 0} photos</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label"><TrendingUp size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Avg Confidence</span>
          <span className="stat-value">{(avgConf * 100).toFixed(1)}%</span>
          <span className="stat-sub">across all photos</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label"><Clock size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Last Updated</span>
          <span className="stat-value" style={{ fontSize: '1.25rem' }}>
            {kb.metadata?.last_updated
              ? new Date(kb.metadata.last_updated).toLocaleDateString()
              : '—'}
          </span>
          <span className="stat-sub">knowledge base</span>
        </div>
      </div>

      {/* Charts Row */}
      <div className="chart-row">
        <div className="card chart-card">
          <h3>Photo Types</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={typeDist} layout="vertical" margin={{ left: 10, right: 20 }}>
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="name"
                width={80}
                tick={{ fontSize: 12, fill: '#6B7280' }}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff', border: '1px solid #E8E5E0',
                  borderRadius: 8, fontSize: 13,
                }}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
                {typeDist.map((entry) => (
                  <Cell key={entry.name} fill={getTypeColor(entry.name)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h3>Confidence Scores</h3>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <XAxis
                dataKey="index" name="Photo" hide
              />
              <YAxis
                dataKey="confidence"
                domain={[0, 1]}
                tick={{ fontSize: 12, fill: '#6B7280' }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff', border: '1px solid #E8E5E0',
                  borderRadius: 8, fontSize: 13,
                }}
                formatter={(val) => [`${(val * 100).toFixed(1)}%`, 'Confidence']}
                labelFormatter={(i) => confData[i]?.filename || ''}
              />
              <Scatter data={confData}>
                {confData.map((entry, i) => (
                  <Cell key={i} fill={getTypeColor(entry.type)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Photos */}
      <div className="card chart-card" style={{ marginBottom: 28 }}>
        <h3>Recently Indexed</h3>
        <div style={{ display: 'flex', gap: 14, overflowX: 'auto', paddingBottom: 4 }}>
          {recentPhotos.map((p) => (
            <div key={p.id} style={{ minWidth: 160, flex: '0 0 auto' }}>
              <img
                src={photoUrl(p.filename)}
                alt={p.filename}
                style={{
                  width: 160, height: 110, objectFit: 'cover',
                  borderRadius: 8, background: '#F1F0ED', display: 'block',
                }}
                loading="lazy"
              />
              <div style={{ fontSize: '0.75rem', marginTop: 6, color: '#6B7280' }}>
                {p.filename}
              </div>
              <span className={`badge badge-${p.image_type}`} style={{ marginTop: 4 }}>
                {p.image_type}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Top Vendors */}
      <div className="card chart-card">
        <h3><Store size={15} style={{ marginRight: 6, verticalAlign: -2 }} />Top Vendors</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {vendors.slice(0, 8).map((v, i) => (
            <div key={v.name} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: '0.75rem',
                color: '#9CA3AF', width: 20, textAlign: 'right',
              }}>
                {i + 1}
              </span>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 500 }}>{v.name}</div>
                <div style={{
                  height: 4, background: '#F1F0ED', borderRadius: 2,
                  marginTop: 4, overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%', borderRadius: 2,
                    width: `${(v.value / vendors[0].value) * 100}%`,
                    background: 'var(--accent-gradient)',
                  }} />
                </div>
              </div>
              <span style={{
                fontFamily: "'JetBrains Mono', monospace", fontSize: '0.8125rem',
                fontWeight: 600, color: '#2D2D2D',
              }}>
                {v.value}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Eval Quick Stats */}
      {evalData?.summary && (
        <div className="card chart-card" style={{ marginTop: 20 }}>
          <h3>Evaluation Snapshot</h3>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <EvalMiniStat label="Retrieval" value={evalData.summary.retrieval_accuracy} />
            <EvalMiniStat label="Routing" value={evalData.summary.routing_accuracy} />
            <EvalMiniStat label="Silent Fail" value={evalData.summary.silent_failure_rate} invert />
            <EvalMiniStat label="Decline Acc" value={evalData.summary.decline_accuracy} />
          </div>
        </div>
      )}
    </div>
  )
}

function EvalMiniStat({ label, value, invert = false }) {
  const pct = (value * 100).toFixed(0)
  const good = invert ? value < 0.1 : value >= 0.7
  const color = good ? '#22C55E' : value >= 0.5 ? '#EAB308' : '#EF4444'
  return (
    <div style={{ textAlign: 'center', minWidth: 80 }}>
      <div style={{
        fontFamily: "'JetBrains Mono', monospace", fontSize: '1.5rem',
        fontWeight: 700, color,
      }}>
        {pct}%
      </div>
      <div style={{ fontSize: '0.75rem', color: '#6B7280', marginTop: 2 }}>{label}</div>
    </div>
  )
}
