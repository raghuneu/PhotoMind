import { Box, Paper, Stack, Typography } from '@mui/material'
import CameraAltIcon from '@mui/icons-material/CameraAlt'
import TravelExploreIcon from '@mui/icons-material/TravelExplore'
import VerifiedIcon from '@mui/icons-material/Verified'
import { motion } from 'framer-motion'
import { NPR } from '../theme'

const STEPS = [
  {
    n: '01',
    icon: <CameraAltIcon sx={{ fontSize: 22 }} />,
    title: 'Ingest',
    claim: 'GPT-4o Vision reads every photo.',
    proof:
      'Receipts, meals, screenshots, scenes — extracted into structured OCR, entities, and descriptions.',
  },
  {
    n: '02',
    icon: <TravelExploreIcon sx={{ fontSize: 22 }} />,
    title: 'Retrieve',
    claim: 'Intent-aware routing picks the right lens.',
    proof:
      'A contextual bandit learns whether your question is factual, semantic, or behavioral — and routes accordingly.',
  },
  {
    n: '03',
    icon: <VerifiedIcon sx={{ fontSize: 22 }} />,
    title: 'Calibrate',
    claim: 'A DQN decides when to speak up — or decline.',
    proof:
      'Every answer ships with a confidence grade and source photos. If confidence is low, PhotoMind hedges instead of hallucinating.',
  },
]

export default function HowItWorks() {
  return (
    <Box sx={{ mb: 6 }}>
      <Stack spacing={1} sx={{ mb: 3 }}>
        <Typography variant="overline" sx={{ color: 'text.secondary', letterSpacing: 2 }}>
          How it works
        </Typography>
        <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: '-0.01em' }}>
          Three systems, working together.
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 620 }}>
          PhotoMind is not just a search bar. It's a pipeline that sees, searches, and
          second-guesses itself — so the answer you get is the answer you can trust.
        </Typography>
      </Stack>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        {STEPS.map((s, i) => (
          <motion.div
            key={s.n}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: i * 0.08 }}
            style={{ flex: 1 }}
          >
            <Paper
              sx={{
                p: 3,
                height: '100%',
                borderRadius: 3,
                display: 'flex',
                flexDirection: 'column',
                gap: 1.5,
                transition: 'all .2s ease',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  borderColor: `${NPR.heliconia}55`,
                  boxShadow: `0 12px 32px -18px ${NPR.heliconia}55`,
                },
              }}
            >
              <Stack direction="row" alignItems="center" spacing={1.5}>
                <Box
                  sx={{
                    width: 38,
                    height: 38,
                    borderRadius: 2,
                    display: 'grid',
                    placeItems: 'center',
                    bgcolor: `${NPR.heliconia}12`,
                    color: NPR.heliconia,
                  }}
                >
                  {s.icon}
                </Box>
                <Typography
                  variant="caption"
                  sx={{
                    fontFamily: 'ui-monospace, monospace',
                    color: 'text.secondary',
                    letterSpacing: 1,
                  }}
                >
                  {s.n} · {s.title.toUpperCase()}
                </Typography>
              </Stack>
              <Typography
                variant="h6"
                sx={{ fontWeight: 700, color: NPR.midnight, lineHeight: 1.3 }}
              >
                {s.claim}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.55 }}>
                {s.proof}
              </Typography>
            </Paper>
          </motion.div>
        ))}
      </Stack>
    </Box>
  )
}
