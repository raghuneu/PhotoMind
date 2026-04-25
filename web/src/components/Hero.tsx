import { useEffect, useState } from 'react'
import { Box, Button, Stack, Typography, Chip } from '@mui/material'
import { alpha } from '@mui/material/styles'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import { motion } from 'framer-motion'
import { fetchJson } from '../lib/api'
import { NPR } from '../theme'

interface Photo {
  id: string
  image_type: string
}

interface Thumb {
  photo_id: string
  data_url: string
  image_type: string
}

/**
 * Landing hero — establishes the "why" in 3 beats:
 * headline, sub-line, and a softly-tiled mosaic of the user's real photos.
 */
export default function Hero({ onCTAClick }: { onCTAClick: () => void }) {
  const [thumbs, setThumbs] = useState<Thumb[]>([])

  useEffect(() => {
    // Pull a handful of real photo thumbnails for the mosaic background.
    // Graceful fallback: if any fail, we still render a solid hero.
    fetchJson<{ photos: Photo[] }>('/knowledge-base')
      .then(async (kb) => {
        const picks = kb.photos
          .filter((p) => p.id !== '25b9c917-eaad-4aa7-8601-e8c493c718a3')
          .slice(0, 9)
        const results = await Promise.allSettled(
          picks.map((p) =>
            fetchJson<{ photo_id: string; data_url: string }>(`/photos/${p.id}/thumbnail`).then(
              (t) => ({ ...t, image_type: p.image_type }),
            ),
          ),
        )
        setThumbs(
          results
            .filter((r): r is PromiseFulfilledResult<Thumb> => r.status === 'fulfilled')
            .map((r) => r.value),
        )
      })
      .catch(() => setThumbs([]))
  }, [])

  return (
    <Box
      sx={{
        position: 'relative',
        overflow: 'hidden',
        borderRadius: 3,
        mb: 6,
        background: `linear-gradient(180deg, ${alpha(NPR.heliconia, 0.05)} 0%, ${NPR.white} 100%)`,
        border: `1px solid ${NPR.border}`,
        px: { xs: 3, md: 6 },
        py: { xs: 6, md: 9 },
      }}
    >
      {/* Mosaic backdrop — right-weighted so text side stays clean */}
      {thumbs.length > 0 && (
        <Box
          aria-hidden
          sx={{
            position: 'absolute',
            inset: 0,
            display: { xs: 'none', md: 'grid' },
            gridTemplateColumns: 'repeat(9, 1fr)',
            gridTemplateRows: 'repeat(3, 1fr)',
            gap: 1.5,
            padding: 2,
            opacity: 0.42,
            filter: 'saturate(0.95)',
            pointerEvents: 'none',
          }}
        >
          {thumbs.slice(0, 9).map((t, i) => (
            <Box
              key={t.photo_id}
              component="img"
              src={t.data_url}
              alt=""
              loading="lazy"
              sx={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                borderRadius: 2,
                boxShadow: '0 8px 24px rgba(0,0,15,0.08)',
                transform: `rotate(${(i % 3) - 1}deg) scale(${1 + (i % 2) * 0.04})`,
              }}
            />
          ))}
        </Box>
      )}
      {/* Gradient veil — heavy on the left (text), soft on the right (mosaic peeks through) */}
      <Box
        aria-hidden
        sx={{
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(90deg, ${alpha(NPR.white, 0.98)} 0%, ${alpha(NPR.white, 0.96)} 52%, ${alpha(NPR.white, 0.35)} 72%, ${alpha(NPR.white, 0.15)} 100%)`,
          pointerEvents: 'none',
        }}
      />

      <Stack spacing={3} sx={{ position: 'relative', maxWidth: 640 }}>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Chip
            icon={<AutoAwesomeIcon sx={{ fontSize: 14 }} />}
            label="Multimodal · Retrieval · Reinforcement Learning"
            size="small"
            sx={{
              bgcolor: alpha(NPR.heliconia, 0.08),
              color: NPR.heliconia,
              border: `1px solid ${NPR.heliconia}33`,
              fontWeight: 600,
              mb: 1,
            }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05 }}
        >
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: '2.4rem', md: '3.6rem' },
              lineHeight: 1.05,
              fontWeight: 800,
              letterSpacing: '-0.03em',
              color: NPR.midnight,
            }}
          >
            Your photos <Box component="span" sx={{ color: NPR.heliconia }}>remember</Box> for you.
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12 }}
        >
          <Typography
            variant="body1"
            sx={{
              fontSize: { xs: '1.05rem', md: '1.2rem' },
              lineHeight: 1.55,
              color: 'text.secondary',
              maxWidth: 620,
            }}
          >
            Receipts, meals, and screenshots pile up — and you still can't find last month's
            total. PhotoMind reads every photo with GPT-4o Vision and answers in plain English,
            with sources and a confidence grade.
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Button
            variant="contained"
            size="large"
            endIcon={<ArrowDownwardIcon />}
            onClick={onCTAClick}
            sx={{ px: 3.5, py: 1.2, fontSize: '1rem' }}
          >
            Ask your photos
          </Button>
        </motion.div>
      </Stack>
    </Box>
  )
}


