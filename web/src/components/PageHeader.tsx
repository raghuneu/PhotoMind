import { Box, Stack, Typography } from '@mui/material'
import { motion } from 'framer-motion'

interface PageHeaderProps {
  overline: string
  title: string
  subtitle?: string
  action?: React.ReactNode
}

/**
 * Shared top-of-tab header. Establishes the same narrative rhythm
 * on every tab: eyebrow → title → one-line context.
 */
export default function PageHeader({ overline, title, subtitle, action }: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
    >
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={2}
        alignItems={{ xs: 'flex-start', md: 'flex-end' }}
        sx={{ mb: 4 }}
      >
        <Stack spacing={1} sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography variant="overline" sx={{ color: 'text.secondary', letterSpacing: 2 }}>
            {overline}
          </Typography>
          <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: '-0.01em' }}>
            {title}
          </Typography>
          {subtitle && (
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 620 }}>
              {subtitle}
            </Typography>
          )}
        </Stack>
        {action && <Box sx={{ flexShrink: 0 }}>{action}</Box>}
      </Stack>
    </motion.div>
  )
}
