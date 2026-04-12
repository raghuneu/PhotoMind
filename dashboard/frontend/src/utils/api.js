const API_BASE = '/api'

export async function fetchKnowledgeBase() {
  const res = await fetch(`${API_BASE}/knowledge-base`)
  if (!res.ok) throw new Error('Failed to fetch knowledge base')
  return res.json()
}

export async function fetchEvalResults() {
  const res = await fetch(`${API_BASE}/eval-results`)
  if (!res.ok) throw new Error('Failed to fetch eval results')
  return res.json()
}

export async function quickQuery(query, queryType = 'auto', topK = 5) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, query_type: queryType, top_k: topK }),
  })
  if (!res.ok) throw new Error('Query failed')
  return res.json()
}

export async function crewQuery(query) {
  const res = await fetch(`${API_BASE}/query/crew`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error('Crew query failed')
  return res.json()
}

export function photoUrl(filename) {
  return `${API_BASE}/photos/${encodeURIComponent(filename)}`
}
