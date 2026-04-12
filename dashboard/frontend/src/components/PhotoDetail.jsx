import { X } from 'lucide-react'
import { photoUrl } from '../utils/api'
import ConfidenceGauge from './ConfidenceGauge'

export default function PhotoDetail({ photo, onClose }) {
  if (!photo) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{ position: 'relative' }}
      >
        <button className="modal-close" onClick={onClose}>
          <X size={20} />
        </button>

        {/* Image */}
        <img
          src={photoUrl(photo.filename)}
          alt={photo.filename}
          style={{
            width: '100%',
            maxHeight: 340,
            objectFit: 'cover',
            borderRadius: '16px 16px 0 0',
            display: 'block',
            background: '#F1F0ED',
          }}
        />

        <div style={{ padding: '24px 28px 28px' }}>
          {/* Header row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <h2 style={{ flex: 1, fontSize: '1.125rem' }}>{photo.filename}</h2>
            <span className={`badge badge-${photo.image_type}`}>{photo.image_type}</span>
          </div>

          {/* Confidence */}
          <div style={{ marginBottom: 16 }}>
            <ConfidenceGauge score={photo.confidence} />
          </div>

          {/* Description */}
          <div style={{ marginBottom: 20 }}>
            <h4 style={{ fontSize: '0.8125rem', color: '#6B7280', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Description
            </h4>
            <p style={{ fontSize: '0.9375rem', lineHeight: 1.65, color: '#2D2D2D' }}>
              {photo.description || 'No description available.'}
            </p>
          </div>

          {/* Entities */}
          {photo.entities && photo.entities.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <h4 style={{ fontSize: '0.8125rem', color: '#6B7280', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Extracted Entities
              </h4>
              <div className="entity-chips">
                {photo.entities.map((e, i) => (
                  <span key={i} className={`entity-chip ${e.type}`}>
                    <span style={{ opacity: 0.7, fontSize: '0.6875rem' }}>{e.type}</span>
                    {e.value}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* OCR Text */}
          {photo.ocr_text && photo.ocr_text.trim() && (
            <div>
              <h4 style={{ fontSize: '0.8125rem', color: '#6B7280', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                OCR Text
              </h4>
              <div className="ocr-box">{photo.ocr_text}</div>
            </div>
          )}

          {/* Metadata */}
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid #F0EEEA', display: 'flex', gap: 24, fontSize: '0.75rem', color: '#9CA3AF' }}>
            <span>ID: {photo.id?.slice(0, 8)}...</span>
            <span>Indexed: {photo.indexed_at ? new Date(photo.indexed_at).toLocaleString() : '—'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
