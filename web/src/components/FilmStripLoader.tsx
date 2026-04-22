import { Box, Typography } from '@mui/material'
import { NPR } from '../theme'

/**
 * Film-strip scanner — novel loading animation thematic to PhotoMind.
 * A horizontal strip of 12 stylised "photo frames" scrolls continuously
 * while the retrieval pipeline runs, with a Heliconia scan line sweeping
 * across to evoke the camera aperture / bandit scanning photos.
 */
export default function FilmStripLoader({ label = 'Scanning photos…' }: { label?: string }) {
  const frames = Array.from({ length: 12 })
  return (
    <Box sx={{ position: 'relative', overflow: 'hidden', borderRadius: 2, py: 1 }}>
      <Box
        sx={{
          display: 'flex',
          gap: 1,
          width: 'max-content',
          animation: 'film-scroll 8s linear infinite',
        }}
      >
        {[...frames, ...frames].map((_, i) => (
          <Box
            key={i}
            sx={{
              width: 56,
              height: 40,
              borderRadius: 1,
              background: `linear-gradient(135deg, ${NPR.surface} 0%, ${NPR.surfaceAlt} 100%)`,
              border: `1px solid ${NPR.border}`,
              position: 'relative',
              flexShrink: 0,
              overflow: 'hidden',
              '&::before, &::after': {
                content: '""',
                position: 'absolute',
                left: 4, right: 4, height: 3,
                background: NPR.midnight,
                opacity: 0.15,
                borderRadius: 1,
              },
              '&::before': { top: 3 },
              '&::after':  { bottom: 3 },
            }}
          />
        ))}
      </Box>

      {/* Heliconia scan line */}
      <Box
        sx={{
          position: 'absolute',
          top: 0, bottom: 0, width: 2,
          background: `linear-gradient(180deg, transparent, ${NPR.heliconia}, transparent)`,
          boxShadow: `0 0 14px 2px ${NPR.heliconia}`,
          animation: 'scanline 1.8s ease-in-out infinite',
          '@keyframes scanline': {
            '0%':   { left: '0%' },
            '50%':  { left: '100%' },
            '100%': { left: '0%' },
          },
        }}
      />

      <Typography
        variant="caption"
        sx={{
          display: 'block', mt: 1, textAlign: 'center',
          color: 'text.secondary', fontWeight: 500,
          '&::after': {
            content: '"▋"',
            marginLeft: 2,
            color: NPR.heliconia,
            animation: 'caret-blink 1s step-end infinite',
          },
        }}
      >
        {label}
      </Typography>
    </Box>
  )
}
