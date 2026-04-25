import { useEffect, useState, useCallback } from 'react'
import {
  Paper, Stack, TextField, Button, ToggleButton, ToggleButtonGroup,
  Chip, Typography, Box, Alert, IconButton, Tooltip, CircularProgress,
} from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import BoltIcon from '@mui/icons-material/Bolt'
import PsychologyIcon from '@mui/icons-material/Psychology'
import AccessTimeIcon from '@mui/icons-material/AccessTime'
import ShieldIcon from '@mui/icons-material/Shield'
import ThumbUpIcon from '@mui/icons-material/ThumbUpAlt'
import ThumbDownIcon from '@mui/icons-material/ThumbDownAlt'
import ZoomInIcon from '@mui/icons-material/ZoomIn'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchJson, type QueryResponse } from '../lib/api'
import { NPR, GRADE_PALETTE } from '../theme'
import FilmStripLoader from './FilmStripLoader'
import PhotoLightbox from './PhotoLightbox'

const EXAMPLE_QUERIES = [
  'How much did I spend at ALDI?',
  'Show me photos of food',
  'What type of food do I eat most often?',
  'Find expensive restaurant meals',
  'What is the total across all receipts?',
]

const EVIDENCE_PREVIEW_COUNT = 5

interface ParsedAnswer {
  main: string
  highlight: string | null
  details: string | null
  bullets: { label: string; count: number }[] | null
  bulletsLabel: string | null
  totalPhotos: number | null
}

/** Parse a Python-dict-literal fragment like `{'A': 3, 'B': 2}` into entries. */
function parsePyDict(raw: string): { key: string; value: number }[] {
  const entries: { key: string; value: number }[] = []
  // Tolerant regex: 'key': number   |   "key": number
  const re = /['"]([^'"]+)['"]\s*:\s*(\d+(?:\.\d+)?)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(raw)) !== null) {
    entries.push({ key: m[1], value: Number(m[2]) })
  }
  return entries
}

function pluralizeEntityLabel(entityType: string): string {
  const readable = entityType.replace(/_/g, ' ').trim()
  if (!readable) return 'items'
  if (readable.endsWith('s')) return readable
  return `${readable}s`
}

/** Parse the technical answer_summary into user-friendly parts */
function parseAnswerSummary(summary: string, resultCount: number = 1): ParsedAnswer {
  const empty: ParsedAnswer = {
    main: 'No matching photos found.',
    highlight: null,
    details: null,
    bullets: null,
    bulletsLabel: null,
    totalPhotos: null,
  }
  // Authoritative signal: if the backend returned zero results, never claim a match.
  if (resultCount === 0) {
    return empty
  }
  if (!summary || summary === 'No matching photos found in the knowledge base.') {
    return empty
  }

  // --- Behavioral path: photo-type distribution + "Most frequent X" bullets ---
  const dist = summary.match(/Photo type distribution:\s*\{([^}]*)\}/)
  const freqTyped = summary.match(/Most frequent ([a-z_]+?)s?:\s*\{([^}]*)\}/i)
  const freqGeneric = summary.match(/Most frequent entities:\s*\{([^}]*)\}/i)

  if (dist || freqTyped || freqGeneric) {
    let totalPhotos: number | null = null
    if (dist) {
      const typeEntries = parsePyDict(`{${dist[1]}}`)
      if (typeEntries.length > 0) {
        totalPhotos = typeEntries.reduce((sum, e) => sum + e.value, 0)
      }
    }

    let bullets: { label: string; count: number }[] | null = null
    let bulletsLabel: string | null = null
    if (freqTyped) {
      const entries = parsePyDict(`{${freqTyped[2]}}`)
      if (entries.length > 0) {
        bullets = entries.map((e) => ({ label: e.key, count: e.value }))
        bulletsLabel = pluralizeEntityLabel(freqTyped[1])
      }
    } else if (freqGeneric) {
      const entries = parsePyDict(`{${freqGeneric[1]}}`)
      if (entries.length > 0) {
        bullets = entries.map((e) => ({ label: e.key, count: e.value }))
        bulletsLabel = 'entities'
      }
    }

    if (bullets && bullets.length > 0) {
      const main = `Found patterns across your photos`
      return {
        main,
        highlight: null,
        details: null,
        bullets,
        bulletsLabel,
        totalPhotos,
      }
    }
  }

  // --- Financial path: aggregated total ---
  const totalMatch = summary.match(/Aggregated total across (\d+) receipts?: \$?([\d,.]+)/)
  const vendorMatch = summary.match(/vendor: ([^;]+)/i)
  const vendor = vendorMatch ? vendorMatch[1].trim() : null
  const typeMatch = summary.match(/type: (\w+)/)
  const photoType = typeMatch ? typeMatch[1] : 'photo'

  if (totalMatch) {
    const count = totalMatch[1]
    const amount = totalMatch[2]
    const main = vendor ? `You spent at ${vendor}` : `Total spending found`
    const details = `Based on ${count} ${photoType}${parseInt(count) > 1 ? 's' : ''}`
    return {
      main,
      highlight: `$${amount}`,
      details,
      bullets: null,
      bulletsLabel: null,
      totalPhotos: null,
    }
  }

  // --- Generic single/multi match path ---
  const countMatch = summary.match(/across (\d+)/)
  const count = countMatch ? parseInt(countMatch[1]) : 1

  let evidence = ''
  const evidenceMatch = summary.match(/Evidence: (.+?)(?:\s*\||$)/)
  if (evidenceMatch) {
    evidence = evidenceMatch[1]
      .replace(/vendor: [^;]+;?\s*/gi, '')
      .replace(/OCR text match;?\s*/gi, '')
      .replace(/amounts?: [^;]+;?\s*/gi, '')
      .replace(/topic: /gi, '')
      .replace(/entities?: /gi, '')
      .trim()
  }

  const main = count > 1 ? `Found ${count} matching ${photoType}s` : `Found a matching ${photoType}`
  return {
    main,
    highlight: null,
    details: evidence || null,
    bullets: null,
    bulletsLabel: null,
    totalPhotos: null,
  }
}

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'fast' | 'full'>('fast')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)
  const [thumbs, setThumbs] = useState<Record<string, string>>({})
  const [lightboxId, setLightboxId] = useState<string | null>(null)
  const [expandEvidence, setExpandEvidence] = useState(false)

  const handleImageClick = useCallback((photoId: string) => {
    setLightboxId(photoId)
  }, [])

  useEffect(() => {
    if (!result) return
    const ids = result.results.map((r) => r.photo_id).filter((id) => id && !thumbs[id])
    if (ids.length === 0) return
    Promise.allSettled(
      ids.map((id) =>
        fetchJson<{ photo_id: string; data_url: string }>(`/photos/${id}/thumbnail`),
      ),
    ).then((res) => {
      const map: Record<string, string> = { ...thumbs }
      res.forEach((r) => {
        if (r.status === 'fulfilled') map[r.value.photo_id] = r.value.data_url
      })
      setThumbs(map)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setFeedbackSent(false)
    setFeedbackError(null)
    setExpandEvidence(false)
    try {
      const data = await fetchJson<QueryResponse>('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), mode, top_k: 5 }),
      })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  async function sendFeedback(wasCorrect: boolean) {
    if (!result) return
    setFeedbackError(null)
    try {
      await fetchJson('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: result.query,
          strategy: result.query_type_detected,
          was_correct: wasCorrect,
          confidence_score: result.confidence_score,
        }),
      })
      setFeedbackSent(true)
    } catch (err) {
      setFeedbackError(err instanceof Error ? err.message : 'Failed to send feedback')
    }
  }

  const grade = result ? (GRADE_PALETTE[result.confidence_grade] ?? GRADE_PALETTE.F) : null
  const visibleResults = result
    ? expandEvidence
      ? result.results
      : result.results.slice(0, EVIDENCE_PREVIEW_COUNT)
    : []
  const hiddenCount = result ? Math.max(0, result.results.length - EVIDENCE_PREVIEW_COUNT) : 0

  return (
    <Stack spacing={3}>
      {/* Query input card */}
      <Paper sx={{ p: 3, borderRadius: 3 }}>
        <Box component="form" onSubmit={handleSubmit}>
          <Stack direction="row" spacing={1.5}>
            <TextField
              fullWidth
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about your photos…"
              autoComplete="off"
            />
            <Button
              type="submit"
              variant="contained"
              disabled={loading || !query.trim()}
              startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <SendIcon />}
              sx={{ px: 3, minWidth: 120 }}
            >
              Ask
            </Button>
          </Stack>

          {/* Mode toggle */}
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2, flexWrap: 'wrap' }}>
            <Typography variant="body2" color="text.secondary">Mode:</Typography>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={mode}
              onChange={(_, v) => v && setMode(v)}
            >
              <Tooltip title="Keyword + RL retrieval, no LLM call (<1s)">
                <ToggleButton value="fast">
                  <BoltIcon fontSize="small" sx={{ mr: 0.5 }} />
                  Quick answer
                </ToggleButton>
              </Tooltip>
              <Tooltip title="Multi-agent reasoning via CrewAI + GPT-4o (~5s)">
                <ToggleButton value="full">
                  <PsychologyIcon fontSize="small" sx={{ mr: 0.5 }} />
                  Deep reasoning
                </ToggleButton>
              </Tooltip>
            </ToggleButtonGroup>
            {mode === 'full' && (
              <Chip
                size="small"
                variant="outlined"
                label="CrewAI · GPT-4o"
                sx={{
                  fontFamily: 'ui-monospace, monospace',
                  fontSize: 11,
                  color: NPR.jayBlue,
                  borderColor: NPR.jayBlue,
                }}
              />
            )}
          </Stack>

          {/* Example queries */}
          <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {EXAMPLE_QUERIES.map((eq) => (
              <Chip
                key={eq}
                label={eq}
                onClick={() => setQuery(eq)}
                variant="outlined"
                size="small"
                sx={{
                  transition: 'all .18s ease',
                  '&:hover': {
                    borderColor: NPR.jayBlue,
                    color: NPR.jayBlue,
                    transform: 'translateY(-1px)',
                  },
                }}
              />
            ))}
          </Box>
        </Box>
      </Paper>

      {/* Loading film-strip */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Paper sx={{ p: 3, borderRadius: 3 }}>
              <FilmStripLoader
                label={mode === 'full' ? 'Delegating to CrewAI agents…' : 'Running bandit-routed retrieval…'}
              />
            </Paper>
          </motion.div>
        )}
      </AnimatePresence>

      {error && <Alert severity="error" variant="outlined">{error}</Alert>}

      {/* Unified Insight Card */}
      <AnimatePresence>
        {result && grade && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Paper sx={{ borderRadius: 3, overflow: 'hidden' }}>
              {/* ── Header + Body ── */}
              <Box sx={{ p: 3 }}>
                <Stack direction="row" alignItems="flex-start" spacing={2}>
                  <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                    <Typography variant="overline" color="text.secondary">Answer</Typography>
                    {(() => {
                      const parsed = parseAnswerSummary(result.answer_summary, result.results.length)
                      return (
                        <Box sx={{ mt: 0.5 }}>
                          <Typography variant="body1" sx={{ lineHeight: 1.6, color: 'text.primary' }}>
                            {parsed.main}
                          </Typography>

                          {parsed.highlight && (
                            <Box sx={{
                              mt: 1.5, p: 2, borderRadius: 2,
                              bgcolor: `${NPR.jayBlue}10`,
                              border: `1px solid ${NPR.jayBlue}25`,
                            }}>
                              <Typography variant="h4" sx={{ fontWeight: 700, color: NPR.jayBlue }}>
                                {parsed.highlight}
                              </Typography>
                            </Box>
                          )}

                          {parsed.bullets && parsed.bullets.length > 0 && (
                            <Box sx={{ mt: 1.5 }}>
                              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                                {parsed.totalPhotos
                                  ? `Based on ${parsed.totalPhotos} photos, your most frequent ${parsed.bulletsLabel ?? 'items'} are:`
                                  : `Your most frequent ${parsed.bulletsLabel ?? 'items'} are:`}
                              </Typography>
                              <Box component="ul" sx={{ pl: 3, m: 0 }}>
                                {parsed.bullets.map((b, i) => (
                                  <Box
                                    component="li"
                                    key={i}
                                    sx={{ mb: 0.25, lineHeight: 1.6 }}
                                  >
                                    <Typography variant="body2" component="span" sx={{ fontWeight: 600, color: 'text.primary' }}>
                                      {b.label}
                                    </Typography>
                                    <Typography variant="body2" component="span" sx={{ color: 'text.secondary' }}>
                                      {' '}({b.count})
                                    </Typography>
                                  </Box>
                                ))}
                              </Box>
                            </Box>
                          )}

                          {parsed.details && !parsed.bullets && (
                            <Typography variant="body2" sx={{ mt: 1.5, color: 'text.secondary' }}>
                              {parsed.details}
                            </Typography>
                          )}
                        </Box>
                      )
                    })()}
                  </Box>

                  {/* Grade badge docked top-right */}
                  <Tooltip title={grade.label} placement="left">
                    <Box sx={{ position: 'relative', flexShrink: 0 }}>
                      <Box
                        sx={{
                          width: 72, height: 72, borderRadius: 3,
                          display: 'grid', placeItems: 'center',
                          background: grade.bg,
                          border: `2px solid ${grade.border}`,
                          color: grade.fg,
                          fontWeight: 800, fontSize: 40, lineHeight: 1,
                          animation: 'grade-pop 0.5s cubic-bezier(.34,1.56,.64,1) both',
                          position: 'relative',
                        }}
                      >
                        {result.confidence_grade}
                        <Box
                          sx={{
                            position: 'absolute', inset: -2, borderRadius: 3,
                            color: grade.fg, animation: 'halo 1.1s ease-out 1',
                            pointerEvents: 'none',
                          }}
                        />
                      </Box>
                    </Box>
                  </Tooltip>
                </Stack>

                {result.warning && (
                  <Alert severity="warning" icon={<ShieldIcon />} sx={{ mt: 2 }}>
                    {result.warning}
                  </Alert>
                )}

                {/* ── Embedded Evidence Grid ── */}
                {result.results.length > 0 && (
                  <Box sx={{ mt: 2.5 }}>
                    <Typography
                      variant="overline"
                      sx={{ display: 'block', mb: 1, color: NPR.jayBlue, fontWeight: 600 }}
                    >
                      Source Evidence
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1.5 }}>
                      {visibleResults.map((r, i) => (
                        <motion.div
                          key={r.photo_id || i}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.05, duration: 0.25 }}
                        >
                          <Tooltip
                            title={r.matched_entity ? `Evidence: ${r.matched_entity} — ${r.photo_path}` : r.photo_path}
                            placement="top"
                          >
                            <Box
                              onClick={() => r.photo_id && handleImageClick(r.photo_id)}
                              sx={{
                                position: 'relative',
                                width: 120, height: 120,
                                borderRadius: 2,
                                overflow: 'hidden',
                                cursor: r.photo_id ? 'zoom-in' : 'default',
                                border: `1px solid ${NPR.border}`,
                                bgcolor: NPR.surface,
                                transition: 'transform 0.15s ease, border-color 0.15s ease',
                                '&:hover': {
                                  transform: 'translateY(-2px)',
                                  borderColor: NPR.jayBlue,
                                },
                                '&:hover .zoom-overlay': { opacity: 1 },
                              }}
                            >
                              {thumbs[r.photo_id] ? (
                                <Box
                                  component="img"
                                  src={thumbs[r.photo_id]}
                                  alt={r.photo_path}
                                  loading="lazy"
                                  sx={{
                                    width: '100%', height: '100%',
                                    objectFit: 'cover', display: 'block',
                                  }}
                                />
                              ) : (
                                <Box sx={{
                                  width: '100%', height: '100%',
                                  display: 'grid', placeItems: 'center',
                                  color: 'text.secondary',
                                  fontFamily: 'ui-monospace, monospace', fontSize: 13,
                                }}>
                                  #{i + 1}
                                </Box>
                              )}

                              {/* Zoom overlay on hover */}
                              <Box
                                className="zoom-overlay"
                                sx={{
                                  position: 'absolute', inset: 0,
                                  display: 'grid', placeItems: 'center',
                                  bgcolor: 'rgba(0,0,15,0.35)',
                                  opacity: 0, transition: 'opacity 0.15s ease',
                                  pointerEvents: 'none',
                                }}
                              >
                                <ZoomInIcon sx={{ color: NPR.white, fontSize: 28 }} />
                              </Box>

                              {/* Bottom metadata overlay */}
                              <Box sx={{
                                position: 'absolute', left: 0, right: 0, bottom: 0,
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                px: 0.75, py: 0.5,
                                bgcolor: 'rgba(0,0,15,0.55)',
                                color: NPR.white,
                                fontSize: 11, fontWeight: 600,
                                fontFamily: 'ui-monospace, monospace',
                              }}>
                                <Box component="span">{(r.relevance_score * 100).toFixed(0)}%</Box>
                                <Box component="span" sx={{
                                  opacity: 0.85,
                                  textTransform: 'lowercase',
                                  maxWidth: '60%',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                }}>
                                  {r.matched_entity || r.image_type}
                                </Box>
                              </Box>
                            </Box>
                          </Tooltip>
                        </motion.div>
                      ))}

                      {hiddenCount > 0 && !expandEvidence && (
                        <Box
                          onClick={() => setExpandEvidence(true)}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') setExpandEvidence(true)
                          }}
                          sx={{
                            width: 120, height: 120,
                            borderRadius: 2,
                            border: `1px dashed ${NPR.border}`,
                            display: 'grid', placeItems: 'center',
                            cursor: 'pointer',
                            color: NPR.jayBlue, fontWeight: 600,
                            transition: 'background-color 0.15s ease, border-color 0.15s ease',
                            '&:hover': {
                              bgcolor: `${NPR.jayBlue}10`,
                              borderColor: NPR.jayBlue,
                            },
                          }}
                        >
                          +{hiddenCount} more
                        </Box>
                      )}
                    </Box>
                  </Box>
                )}
              </Box>

              {/* ── Footer: telemetry + feedback ── */}
              <Box sx={{
                px: 3, py: 1.5,
                bgcolor: NPR.surface,
                borderTop: `1px solid ${NPR.border}`,
                display: 'flex', alignItems: 'center', gap: 1,
                flexWrap: 'wrap',
              }}>
                <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', rowGap: 0.75 }}>
                  <Chip size="small" icon={<AccessTimeIcon />} label={`${result.latency_s}s`} variant="outlined" sx={{ borderColor: NPR.jayBlue, '& .MuiChip-icon': { color: NPR.jayBlue } }} />
                  <Chip size="small" label={`Strategy: ${result.query_type_detected}`} variant="outlined" sx={{ borderColor: NPR.jayBlue }} />
                  <Chip size="small" label={`Score: ${result.confidence_score}`} variant="outlined" sx={{ borderColor: NPR.jayBlue }} />
                  <Chip size="small" label={`Mode: ${result.mode}`} variant="outlined" sx={{ borderColor: NPR.jayBlue }} />
                  <Chip size="small" label={`Results: ${result.results.length}`} variant="outlined" sx={{ borderColor: NPR.jayBlue }} />
                </Stack>

                <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary">Helpful?</Typography>
                  {feedbackSent ? (
                    <Typography variant="body2" sx={{ color: 'success.main', fontWeight: 600 }}>
                      Thanks!
                    </Typography>
                  ) : (
                    <>
                      <IconButton
                        color="success"
                        size="small"
                        aria-label="Mark answer as helpful"
                        onClick={() => sendFeedback(true)}
                      >
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        color="error"
                        size="small"
                        aria-label="Mark answer as not helpful"
                        onClick={() => sendFeedback(false)}
                      >
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </>
                  )}
                </Box>
                {feedbackError && (
                  <Typography variant="caption" color="error" sx={{ ml: 'auto', display: 'block', textAlign: 'right' }}>
                    {feedbackError}
                  </Typography>
                )}
              </Box>
            </Paper>
          </motion.div>
        )}
      </AnimatePresence>

      <PhotoLightbox
        open={!!lightboxId}
        onClose={() => setLightboxId(null)}
        photoId={lightboxId}
        thumbnailUrl={lightboxId ? thumbs[lightboxId] : undefined}
      />
    </Stack>
  )
}
