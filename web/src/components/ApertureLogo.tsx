import { Box } from '@mui/material'
import { NPR } from '../theme'

/**
 * PhotoMind signature mark — a Polaroid with a pulsing "mind" node.
 * The Jay Blue window is the image; the Heliconia node is the system
 * looking at it. Rocks gently like a photo held in hand.
 */
export default function ApertureLogo({ size = 36 }: { size?: number }) {
  return (
    <Box
      sx={{
        width: size,
        height: size,
        display: 'grid',
        placeItems: 'center',
      }}
    >
      <Box
        component="svg"
        viewBox="-45 -50 90 100"
        sx={{
          width: size,
          height: size,
          overflow: 'visible',
          animation: 'photo-rock 4s ease-in-out infinite',
          transformOrigin: 'center',
          '@keyframes photo-rock': {
            '0%, 100%': { transform: 'rotate(-3deg)' },
            '50%': { transform: 'rotate(3deg)' },
          },
        }}
      >
        {/* Soft drop shadow */}
        <rect
          x="-36"
          y="-42"
          width="72"
          height="86"
          rx="4"
          fill="rgba(0,0,15,0.18)"
          transform="translate(1.8, 2.4)"
        />
        {/* Polaroid body */}
        <rect
          x="-36"
          y="-42"
          width="72"
          height="86"
          rx="4"
          fill={NPR.white}
          stroke={NPR.midnight}
          strokeWidth="2"
        />
        {/* Image window */}
        <rect
          x="-28"
          y="-34"
          width="56"
          height="56"
          rx="2"
          fill={NPR.jayBlue}
        />
        {/* Pulsing "mind" node */}
        <Box
          component="circle"
          cx="0"
          cy="-6"
          r="8"
          sx={{
            fill: NPR.heliconia,
            transformBox: 'fill-box',
            transformOrigin: 'center',
            animation: 'node-pulse 1.8s ease-in-out infinite',
            '@keyframes node-pulse': {
              '0%, 100%': { opacity: 0.75, transform: 'scale(0.85)' },
              '50%': { opacity: 1, transform: 'scale(1.15)' },
            },
          }}
        />
        {/* Catch-light highlight */}
        <circle cx="-2.5" cy="-8.5" r="1.9" fill={NPR.white} opacity="0.9" />
      </Box>
    </Box>
  )
}
