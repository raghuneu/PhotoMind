import { Box } from '@mui/material'
import { NPR } from '../theme'

/**
 * Animated camera aperture logo — PhotoMind's signature mark.
 * Six Heliconia blades rotate subtly, Jay Blue inner iris pulses.
 * CSS-only animation, no runtime cost.
 */
export default function ApertureLogo({ size = 36 }: { size?: number }) {
  const blades = Array.from({ length: 6 })
  return (
    <Box
      sx={{
        width: size,
        height: size,
        position: 'relative',
        display: 'grid',
        placeItems: 'center',
        borderRadius: '50%',
        background: NPR.white,
        border: `2px solid ${NPR.midnight}`,
        animation: 'aperture-pulse 2.8s ease-in-out infinite',
      }}
    >
      <Box
        component="svg"
        viewBox="-50 -50 100 100"
        sx={{
          width: size - 8,
          height: size - 8,
          animation: 'spin 18s linear infinite',
          '@keyframes spin': {
            from: { transform: 'rotate(0deg)' },
            to: { transform: 'rotate(360deg)' },
          },
        }}
      >
        {blades.map((_, i) => {
          const angle = (i * 360) / 6
          return (
            <polygon
              key={i}
              points="0,-34 14,-4 -14,-4"
              fill={NPR.heliconia}
              opacity={0.85 - i * 0.04}
              transform={`rotate(${angle})`}
            />
          )
        })}
        <circle r="10" fill={NPR.jayBlue} />
        <circle r="4" fill={NPR.white} />
      </Box>
    </Box>
  )
}
