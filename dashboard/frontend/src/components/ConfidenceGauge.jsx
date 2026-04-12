import { gradeColor } from '../utils/chartHelpers'

export default function ConfidenceGauge({ score, grade, size = 'md' }) {
  const pct = Math.round((score || 0) * 100)
  const color = grade ? gradeColor(grade) : barColor(score)
  const isSm = size === 'sm'

  return (
    <div className="confidence-gauge">
      <div className="confidence-bar-bg" style={{ height: isSm ? 4 : 6 }}>
        <div
          className="confidence-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span
        className="confidence-label"
        style={{ color, fontSize: isSm ? '0.75rem' : '0.8125rem' }}
      >
        {grade || `${pct}%`}
      </span>
    </div>
  )
}

function barColor(score) {
  if (score >= 0.9) return '#22C55E'
  if (score >= 0.7) return '#84CC16'
  if (score >= 0.5) return '#EAB308'
  if (score >= 0.3) return '#F97316'
  return '#EF4444'
}
