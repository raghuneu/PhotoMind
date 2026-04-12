import { useState, useMemo } from 'react'
import { photoUrl } from '../utils/api'
import { getTypeColor } from '../utils/chartHelpers'
import PhotoDetail from './PhotoDetail'
import ConfidenceGauge from './ConfidenceGauge'

const TYPE_FILTERS = ['all', 'receipt', 'food', 'screenshot', 'document', 'other']
const SORT_OPTIONS = [
  { value: 'recent', label: 'Most Recent' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'filename', label: 'Filename' },
]

export default function PhotoGallery({ kb }) {
  const [typeFilter, setTypeFilter] = useState('all')
  const [sortBy, setSortBy] = useState('recent')
  const [selected, setSelected] = useState(null)

  const photos = kb?.photos || []

  const filtered = useMemo(() => {
    let list = typeFilter === 'all'
      ? photos
      : photos.filter((p) => p.image_type === typeFilter)

    if (sortBy === 'recent') {
      list = [...list].sort((a, b) => new Date(b.indexed_at) - new Date(a.indexed_at))
    } else if (sortBy === 'confidence') {
      list = [...list].sort((a, b) => b.confidence - a.confidence)
    } else {
      list = [...list].sort((a, b) => a.filename.localeCompare(b.filename))
    }
    return list
  }, [photos, typeFilter, sortBy])

  return (
    <div>
      <div className="page-header">
        <h1>Photo Gallery</h1>
        <p>{photos.length} photos in your knowledge base</p>
      </div>

      {/* Filter + Sort Bar */}
      <div className="filter-bar">
        {TYPE_FILTERS.map((t) => (
          <button
            key={t}
            className={`filter-pill ${typeFilter === t ? 'active' : ''}`}
            onClick={() => setTypeFilter(t)}
          >
            {t === 'all' ? 'All' : t.charAt(0).toUpperCase() + t.slice(1)}
            {t !== 'all' && (
              <span style={{ marginLeft: 4, opacity: 0.7 }}>
                ({photos.filter((p) => p.image_type === t).length})
              </span>
            )}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <select
          className="sort-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
        >
          {SORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      </div>

      {/* Photo Grid */}
      <div className="photo-grid">
        {filtered.map((photo) => (
          <div
            key={photo.id}
            className="card card-lift photo-card"
            onClick={() => setSelected(photo)}
          >
            <img
              src={photoUrl(photo.filename)}
              alt={photo.filename}
              className="photo-thumb"
              loading="lazy"
              onError={(e) => {
                e.target.style.background = '#F1F0ED'
                e.target.alt = 'Failed to load'
              }}
            />
            <div className="photo-info">
              <div className="photo-filename">{photo.filename}</div>
              <div className="photo-desc">
                {photo.description || 'No description'}
              </div>
              <div className="photo-meta">
                <span className={`badge badge-${photo.image_type}`}>
                  {photo.image_type}
                </span>
                <div style={{ flex: 1, maxWidth: 120 }}>
                  <ConfidenceGauge score={photo.confidence} size="sm" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div style={{
          textAlign: 'center', padding: '48px 0',
          color: 'var(--text-tertiary)', fontSize: '0.9375rem',
        }}>
          No photos match this filter.
        </div>
      )}

      {/* Detail Modal */}
      {selected && (
        <PhotoDetail photo={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}
