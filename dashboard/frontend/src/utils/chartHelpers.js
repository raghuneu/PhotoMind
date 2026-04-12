/**
 * Transform knowledge base data for charts.
 */

export function getTypeDistribution(photos) {
  const counts = {}
  photos.forEach((p) => {
    const t = p.image_type || 'unknown'
    counts[t] = (counts[t] || 0) + 1
  })
  return Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
}

export function getVendorDistribution(photos) {
  const counts = {}
  photos.forEach((p) => {
    ;(p.entities || []).forEach((e) => {
      if (e.type === 'vendor' && e.value && e.value !== 'Receipt') {
        const vendor = e.value.trim()
        counts[vendor] = (counts[vendor] || 0) + 1
      }
    })
  })
  return Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
}

export function getSpendingByVendor(photos) {
  const spending = {}
  photos
    .filter((p) => p.image_type === 'receipt')
    .forEach((p) => {
      let vendor = 'Unknown'
      let maxAmount = 0
      ;(p.entities || []).forEach((e) => {
        if (e.type === 'vendor' && e.value && e.value !== 'Receipt') {
          vendor = e.value.trim()
        }
        if (e.type === 'amount') {
          const val = parseFloat(String(e.value).replace(/[^0-9.-]/g, ''))
          if (!isNaN(val) && val > maxAmount) maxAmount = val
        }
      })
      if (maxAmount > 0) {
        spending[vendor] = (spending[vendor] || 0) + maxAmount
      }
    })
  return Object.entries(spending)
    .map(([name, value]) => ({ name, value: Math.round(value * 100) / 100 }))
    .sort((a, b) => b.value - a.value)
}

export function getConfidenceData(photos) {
  return photos.map((p, i) => ({
    index: i,
    filename: p.filename,
    confidence: p.confidence,
    type: p.image_type,
  }))
}

export function getTopFoodItems(photos) {
  const counts = {}
  photos.forEach((p) => {
    ;(p.entities || []).forEach((e) => {
      if (e.type === 'food_item') {
        const item = e.value.trim()
        counts[item] = (counts[item] || 0) + 1
      }
    })
  })
  return Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 15)
}

export function getReceiptTimeline(photos) {
  return photos
    .filter((p) => p.image_type === 'receipt')
    .map((p) => {
      let date = null
      let amount = 0
      let vendor = 'Unknown'
      ;(p.entities || []).forEach((e) => {
        if (e.type === 'date') date = e.value
        if (e.type === 'vendor' && e.value && e.value !== 'Receipt') vendor = e.value
        if (e.type === 'amount') {
          const val = parseFloat(String(e.value).replace(/[^0-9.-]/g, ''))
          if (!isNaN(val) && val > amount) amount = val
        }
      })
      return { date: date || 'Unknown', amount, vendor, filename: p.filename }
    })
    .filter((r) => r.amount > 0)
    .sort((a, b) => {
      if (a.date === 'Unknown') return 1
      if (b.date === 'Unknown') return -1
      return new Date(a.date) - new Date(b.date)
    })
}

const TYPE_COLORS = {
  receipt: '#5B8DEF',
  food: '#E8A838',
  screenshot: '#9B6FE3',
  document: '#4ECDC4',
  other: '#95A5B8',
  unknown: '#BDC3CB',
}

export function getTypeColor(type) {
  return TYPE_COLORS[type] || TYPE_COLORS.unknown
}

export function gradeColor(grade) {
  const colors = { A: '#22C55E', B: '#84CC16', C: '#EAB308', D: '#F97316', F: '#EF4444' }
  return colors[grade] || colors.F
}
