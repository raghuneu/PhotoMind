import { useState } from 'react'
import {
  Paper, Stack, TextField, Button, ToggleButton, ToggleButtonGroup,
  Chip, Typography, Box, Alert, IconButton, Divider, Tooltip, CircularProgress,
} from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import BoltIcon from '@mui/icons-material/Bolt'
import PsychologyIcon from '@mui/icons-material/Psychology'
import AccessTimeIcon from '@mui/icons-material/AccessTime'
import ShieldIcon from '@mui/icons-material/Shield'
import ThumbUpIcon from '@mui/icons-material/ThumbUpAlt'
import ThumbDownIcon from '@mui/icons-material/ThumbDownAlt'
import ImageIcon from '@mui/icons-material/Image'
import { motion, AnimatePresence } from 'framer-motion'
import { fetchJson, type QueryResponse } from '../lib/api'
import { NPR, GRADE_PALETTE } from '../theme'
import FilmStripLoader from './FilmStripLoader'

const EXAMPLE_QUERIES = [
  'How much did I spend at ALDI?',
  'Show me photos of food',
  'What type of food do I eat most often?',
  'Find expensive restaurant meals',
  'What is the total across all receipts?',
]

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'fast' | 'full'>('fast')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [feedbackSent, setFeedbackSent] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    setFeedbackSent(false)
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
    try {
      await fetchJson('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: result.query,
          strategy: result.query_type_detected,
          was_correct: wasCorrect,
        }),
      })
      setFeedbackSent(true)
    } catch {
      /* silent */
    }
  }

  const grade = result ? (GRADE_PALETTE[result.confidence_grade] ?? GRADE_PALETTE.F) : null

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
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary">Mode:</Typography>
            <ToggleButtonGroup
              size="small"
              exclusive
              value={mode}
              onChange={(_, v) => v && setMode(v)}
            >
              <ToggleButton value="fast">
                <BoltIcon fontSize="small" sx={{ mr: 0.5 }} />
                Fast (free, &lt;1s)
              </ToggleButton>
              <ToggleButton value="full">
                <PsychologyIcon fontSize="small" sx={{ mr: 0.5 }} />
                Full CrewAI (GPT-4o)
              </ToggleButton>
            </ToggleButtonGroup>
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
                    borderColor: NPR.heliconia,
                    color: NPR.heliconia,
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

      {/* Result */}
      <AnimatePresence>
        {result && grade && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Paper sx={{ p: 3, borderRadius: 3 }}>
              <Stack direction="row" alignItems="flex-start" spacing={2}>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="overline" color="text.secondary">Answer</Typography>
                  <Typography variant="body1" sx={{ mt: 0.5, lineHeight: 1.6, color: 'text.primary' }}>
                    {result.answer_summary}
                  </Typography>
                </Box>

                {/* Animated grade badge with halo */}
                <Tooltip title={grade.label} placement="left">
                  <Box sx={{ position: 'relative' }}>
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

              <Stack direction="row" spacing={2} sx={{ mt: 2, flexWrap: 'wrap' }}>
                <Chip size="small" icon={<AccessTimeIcon />} label={`${result.latency_s}s`} variant="outlined" />
                <Chip size="small" label={`Strategy: ${result.query_type_detected}`} sx={{ color: NPR.heliconia, borderColor: NPR.heliconia }} variant="outlined" />
                <Chip size="small" label={`Score: ${result.confidence_score}`} variant="outlined" />
                <Chip size="small" label={`Mode: ${result.mode}`} variant="outlined" />
                <Chip size="small" label={`Results: ${result.results.length}`} variant="outlined" />
              </Stack>

              <Divider sx={{ my: 2 }} />
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Typography variant="body2" color="text.secondary">Was this helpful?</Typography>
                {feedbackSent ? (
                  <Typography variant="body2" sx={{ color: 'success.main', fontWeight: 600 }}>
                    Thanks for your feedback!
                  </Typography>
                ) : (
                  <>
                    <IconButton color="success" size="small" onClick={() => sendFeedback(true)}>
                      <ThumbUpIcon fontSize="small" />
                    </IconButton>
                    <IconButton color="error" size="small" onClick={() => sendFeedback(false)}>
                      <ThumbDownIcon fontSize="small" />
                    </IconButton>
                  </>
                )}
              </Stack>
            </Paper>

            {/* Source photos with stagger fade-up */}
            {result.results.length > 0 && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                  Source Photos
                </Typography>
                <Stack spacing={1.5}>
                  {result.results.map((r, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -16 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.07, duration: 0.3 }}
                    >
                      <Paper sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box
                          sx={{
                            width: 40, height: 40, borderRadius: 2,
                            bgcolor: NPR.surface, color: 'text.secondary',
                            display: 'grid', placeItems: 'center',
                            fontFamily: 'ui-monospace, monospace', fontSize: 13,
                          }}
                        >
                          #{i + 1}
                        </Box>
                        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.25 }}>
                            <ImageIcon fontSize="small" sx={{ color: NPR.jayBlue }} />
                            <Typography sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 13, color: NPR.heliconia }}>
                              {r.photo_path}
                            </Typography>
                            <Chip size="small" label={r.image_type} variant="outlined" />
                          </Stack>
                          <Typography variant="body2" color="text.secondary" noWrap>
                            {r.evidence}
                          </Typography>
                        </Box>
                        <Box sx={{ textAlign: 'right' }}>
                          <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 700, lineHeight: 1 }}>
                            {(r.relevance_score * 100).toFixed(0)}%
                          </Typography>
                          <Typography variant="caption" color="text.secondary">relevance</Typography>
                        </Box>
                      </Paper>
                    </motion.div>
                  ))}
                </Stack>
              </Box>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </Stack>
  )
}
