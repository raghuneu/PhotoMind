import { Paper, Stack, Typography, Chip, Box } from '@mui/material'
import { NPR } from '../theme'

type Tone = 'default' | 'good' | 'warn' | 'bad'

interface MetricCardProps {
  label: string
  value: string | number
  icon?: React.ReactNode
  tone?: Tone
  chip?: string
  variant?: 'full' | 'compact'
}

const toneColor: Record<Tone, string> = {
  default: NPR.midnight,
  good: '#2E7D32',
  warn: '#ED6C02',
  bad: '#D32F2F',
}

/**
 * Shared stat/metric card used across Hero, KBBrowser, and Dashboard.
 * One component, one look, one vertical rhythm.
 */
export default function MetricCard({
  label, value, icon, tone = 'default', chip, variant = 'full',
}: MetricCardProps) {
  const displayValue =
    typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(1) : value

  if (variant === 'compact') {
    return (
      <Box>
        <Typography sx={{ fontWeight: 800, fontSize: '1.15rem', color: NPR.midnight, lineHeight: 1 }}>
          {displayValue}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.72rem' }}>
          {label}
        </Typography>
      </Box>
    )
  }

  return (
    <Paper sx={{ p: 2, borderRadius: 3, height: '100%' }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ color: 'text.secondary', mb: 0.5 }}>
        {icon}
        <Typography variant="overline">{label}</Typography>
      </Stack>
      <Typography variant="h4" sx={{ fontWeight: 800, color: toneColor[tone], lineHeight: 1.1 }}>
        {displayValue}
      </Typography>
      {chip && (
        <Chip
          size="small"
          label={chip}
          variant="outlined"
          sx={{ mt: 1, height: 20, fontSize: 11 }}
        />
      )}
    </Paper>
  )
}
