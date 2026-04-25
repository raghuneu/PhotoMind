import { useEffect, useMemo, useState } from 'react'
import { Box, Typography, Stack, Button, LinearProgress } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'
import RefreshIcon from '@mui/icons-material/Refresh'
import { NPR } from '../theme'
import ApertureLogo from './ApertureLogo'
import FilmStripLoader from './FilmStripLoader'

interface Props {
  elapsedSec: number
  onRetry: () => void
}

const FACTOIDS = [
  'PhotoMind uses four retrieval strategies — factual, semantic, behavioral, and embedding.',
  'A contextual bandit learns which retrieval strategy to pick for each query type.',
  'A DQN confidence calibrator decides when to accept, hedge, or decline an answer.',
  'Every answer comes with a grade (A–F) and the exact photos it was drawn from.',
  'The evaluation suite covers 83 hand-labeled queries across factual, semantic, behavioral, edge-case, and ambiguous intents.',
  'RL training runs offline on CPU — no API calls, ~120 s for 4000 episodes × 5 seeds.',
  'Silent failures — confident-but-wrong answers — are penalized at −1.0 during RL training.',
]

function stageCopy(elapsed: number): { title: string; subtitle: string } {
  if (elapsed < 5) {
    return {
      title: 'Warming up PhotoMind…',
      subtitle: 'Connecting to the knowledge base.',
    }
  }
  if (elapsed < 20) {
    return {
      title: 'Waking the backend from its nap.',
      subtitle: 'Render free tier spins down after inactivity — a cold start takes about 30 seconds.',
    }
  }
  if (elapsed < 45) {
    return {
      title: 'Almost there.',
      subtitle: 'Loading RL models and rehydrating the photo index.',
    }
  }
  return {
    title: 'Still waking up.',
    subtitle: 'If this lingers, the server may be redeploying. You can retry manually below.',
  }
}

export default function BackendWakeLoader({ elapsedSec, onRetry }: Props) {
  const { title, subtitle } = useMemo(() => stageCopy(elapsedSec), [elapsedSec])
  const [factoidIdx, setFactoidIdx] = useState(0)

  useEffect(() => {
    const t = setInterval(() => setFactoidIdx((i) => (i + 1) % FACTOIDS.length), 4500)
    return () => clearInterval(t)
  }, [])

  // Rough progress — asymptotic toward 95% over ~60 s, never completes
  const progress = Math.min(95, 100 * (1 - Math.exp(-elapsedSec / 22)))

  return (
    <Box
      component={motion.div}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      sx={{
        position: 'fixed',
        inset: 0,
        zIndex: 2000,
        bgcolor: NPR.white,
        background: `radial-gradient(1200px 600px at 50% -10%, ${NPR.surfaceAlt}, ${NPR.white} 60%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 3,
      }}
    >
      <Stack spacing={4} alignItems="center" sx={{ maxWidth: 560, width: '100%', textAlign: 'center' }}>
        {/* Hero aperture with halo rings */}
        <Box sx={{ position: 'relative', width: 160, height: 160, display: 'grid', placeItems: 'center' }}>
          {[0, 1, 2].map((i) => (
            <Box
              key={i}
              sx={{
                position: 'absolute',
                width: 160,
                height: 160,
                borderRadius: '50%',
                border: `1.5px solid ${NPR.heliconia}`,
                opacity: 0,
                animation: 'halo-ring 2.4s ease-out infinite',
                animationDelay: `${i * 0.8}s`,
                '@keyframes halo-ring': {
                  '0%':   { transform: 'scale(0.4)', opacity: 0.55 },
                  '80%':  { transform: 'scale(1.1)', opacity: 0 },
                  '100%': { transform: 'scale(1.1)', opacity: 0 },
                },
                '@media (prefers-reduced-motion: reduce)': { animation: 'none', opacity: 0.2 },
              }}
            />
          ))}
          <ApertureLogo size={96} />
        </Box>

        <Stack spacing={1} alignItems="center">
          <Typography variant="overline" sx={{ color: NPR.heliconia, letterSpacing: '0.18em', fontWeight: 700 }}>
            PhotoMind
          </Typography>
          <AnimatePresence mode="wait">
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.3 }}
            >
              <Typography variant="h4" sx={{ fontWeight: 700, color: NPR.midnight }}>
                {title}
              </Typography>
            </motion.div>
          </AnimatePresence>
          <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 440 }}>
            {subtitle}
          </Typography>
        </Stack>

        <Box sx={{ width: '100%' }}>
          <FilmStripLoader label="Scanning the knowledge base…" />
        </Box>

        <Box sx={{ width: '100%', maxWidth: 420 }}>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{
              height: 6,
              borderRadius: 3,
              bgcolor: NPR.surfaceAlt,
              '& .MuiLinearProgress-bar': {
                background: `linear-gradient(90deg, ${NPR.heliconia}, ${NPR.jayBlue})`,
              },
            }}
          />
          <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'ui-monospace, monospace' }}>
              elapsed {elapsedSec.toFixed(0)}s
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'ui-monospace, monospace' }}>
              {progress.toFixed(0)}%
            </Typography>
          </Stack>
        </Box>

        {/* Rotating factoids */}
        <Box sx={{ minHeight: 48, display: 'flex', alignItems: 'center' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={factoidIdx}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.4 }}
            >
              <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                Did you know? {FACTOIDS[factoidIdx]}
              </Typography>
            </motion.div>
          </AnimatePresence>
        </Box>

        {elapsedSec >= 15 && (
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={onRetry}
            sx={{
              borderColor: NPR.border,
              color: NPR.midnight,
              '&:hover': { borderColor: NPR.heliconia, color: NPR.heliconia },
            }}
          >
            Retry now
          </Button>
        )}
      </Stack>
    </Box>
  )
}
