import { useEffect, useState } from 'react'
import {
  Paper, Stack, Typography, Box, Chip, Divider, Skeleton,
} from '@mui/material'
import Grid from '@mui/material/Grid2'
import SmartToyIcon from '@mui/icons-material/SmartToy'
import AccountTreeIcon from '@mui/icons-material/AccountTree'
import PsychologyIcon from '@mui/icons-material/Psychology'
import SearchIcon from '@mui/icons-material/Search'
import { motion } from 'framer-motion'
import { fetchJson } from '../lib/api'
import { NPR } from '../theme'
import { alpha } from '@mui/material/styles'

interface ArchData {
  agents: { name: string; role: string; description: string }[]
  search_strategies: { name: string; description: string; type: string }[]
  rl_components: { name: string; type: string; purpose: string; state: string; actions: string }[]
  pipeline_modes: { name: string; description: string; cost: string }[]
}

export default function Architecture() {
  const [data, setData] = useState<ArchData | null>(null)
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    Promise.allSettled([
      fetchJson<ArchData>('/system/architecture'),
      fetchJson<Record<string, unknown>>('/health'),
    ]).then(([archRes, hRes]) => {
      if (archRes.status === 'fulfilled') setData(archRes.value)
      if (hRes.status === 'fulfilled') setHealth(hRes.value)
    })
  }, [])

  if (!data) {
    return (
      <Stack spacing={2}>
        <Skeleton variant="rounded" height={80} animation="wave" />
        <Skeleton variant="rounded" height={180} animation="wave" />
        <Skeleton variant="rounded" height={240} animation="wave" />
      </Stack>
    )
  }

  return (
    <Stack spacing={3}>
      {health && (
        <Paper sx={{ p: 2, borderRadius: 3 }}>
          <Typography variant="overline" color="text.secondary">System Health</Typography>
          <Stack direction="row" spacing={3} flexWrap="wrap" sx={{ mt: 1 }}>
            <HealthDot label="Status" ok={health.status === 'ok'} />
            <HealthDot label="Knowledge Base" ok={(health.knowledge_base_photos as number) > 0} detail={`${health.knowledge_base_photos} photos`} />
            <HealthDot label="Eval Results" ok={!!health.has_eval_results} />
            <HealthDot label="RL Models" ok={!!health.has_rl_models} />
          </Stack>
        </Paper>
      )}

      {/* Animated Pipeline Diagram */}
      <Paper sx={{ p: 3, borderRadius: 3 }}>
        <Typography variant="overline" color="text.secondary">Query Pipeline (animated)</Typography>
        <PipelineDiagram />
      </Paper>

      <Paper sx={{ p: 2, borderRadius: 3 }}>
        <Typography variant="overline" color="text.secondary">Pipeline Modes</Typography>
        <Grid container spacing={2} sx={{ mt: 0.5 }}>
          {data.pipeline_modes.map((m) => (
            <Grid size={{ xs: 12, md: 6 }} key={m.name}>
              <Paper sx={{ p: 2, bgcolor: NPR.surface, borderRadius: 2 }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <AccountTreeIcon sx={{ color: NPR.heliconia }} fontSize="small" />
                  <Typography sx={{ fontWeight: 600, textTransform: 'capitalize' }}>{m.name} Mode</Typography>
                  <Box sx={{ flexGrow: 1 }} />
                  <Chip size="small" label={m.cost} />
                </Stack>
                <Typography variant="body2" color="text.secondary">{m.description}</Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Paper>

      <Paper sx={{ p: 2, borderRadius: 3 }}>
        <Typography variant="overline" color="text.secondary">CrewAI Agent Architecture (Hierarchical)</Typography>
        <Stack spacing={1.5} sx={{ mt: 1 }}>
          {data.agents.map((agent, i) => {
            const isManager = agent.role === 'manager'
            return (
              <motion.div
                key={agent.name}
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08 }}
              >
                <Stack direction="row" spacing={2} alignItems="flex-start">
                  <Box
                    sx={{
                      width: 42, height: 42, borderRadius: 2,
                      display: 'grid', placeItems: 'center', flexShrink: 0,
                      bgcolor: alpha(isManager ? NPR.heliconia : NPR.jayBlue, 0.12),
                      color: isManager ? NPR.heliconia : NPR.jayBlue,
                    }}
                  >
                    <SmartToyIcon />
                  </Box>
                  <Box sx={{ flexGrow: 1 }}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography sx={{ fontWeight: 600 }}>{agent.name}</Typography>
                      <Chip size="small" label={agent.role} sx={{ bgcolor: alpha(isManager ? NPR.heliconia : NPR.jayBlue, 0.1), color: isManager ? NPR.heliconia : NPR.jayBlue }} />
                    </Stack>
                    <Typography variant="body2" color="text.secondary">{agent.description}</Typography>
                  </Box>
                </Stack>
              </motion.div>
            )
          })}
        </Stack>
      </Paper>

      <Paper sx={{ p: 2, borderRadius: 3 }}>
        <Typography variant="overline" color="text.secondary">Search Strategies (4 Bandit Arms)</Typography>
        <Grid container spacing={2} sx={{ mt: 0.5 }}>
          {data.search_strategies.map((s, i) => (
            <Grid size={{ xs: 12, md: 6 }} key={s.name}>
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.06 }}
              >
                <Paper sx={{ p: 2, bgcolor: NPR.surface, borderRadius: 2 }}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                    <SearchIcon fontSize="small" sx={{ color: NPR.jayBlue }} />
                    <Typography sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 14, color: NPR.heliconia }}>
                      {s.name}
                    </Typography>
                    <Chip size="small" label={s.type} variant="outlined" />
                  </Stack>
                  <Typography variant="body2" color="text.secondary">{s.description}</Typography>
                </Paper>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Paper>

      <Paper sx={{ p: 2, borderRadius: 3 }}>
        <Typography variant="overline" color="text.secondary">Reinforcement Learning Components</Typography>
        <Stack spacing={2} sx={{ mt: 1 }}>
          {data.rl_components.map((rl, i) => (
            <motion.div
              key={rl.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
            >
              <Paper sx={{ p: 2, bgcolor: NPR.surface, borderRadius: 2 }}>
                <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                  <PsychologyIcon fontSize="small" sx={{ color: NPR.heliconia }} />
                  <Typography sx={{ fontWeight: 600 }}>{rl.name}</Typography>
                  <Chip size="small" label={rl.type} variant="outlined" />
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  {rl.purpose}
                </Typography>
                <Grid container spacing={1}>
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 1, bgcolor: NPR.white, borderRadius: 1 }}>
                      <Typography component="span" variant="caption" color="text.secondary">State: </Typography>
                      <Typography component="span" variant="body2">{rl.state}</Typography>
                    </Paper>
                  </Grid>
                  <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 1, bgcolor: NPR.white, borderRadius: 1 }}>
                      <Typography component="span" variant="caption" color="text.secondary">Actions: </Typography>
                      <Typography component="span" variant="body2">{rl.actions}</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </Paper>
            </motion.div>
          ))}
        </Stack>
      </Paper>
    </Stack>
  )
}

function HealthDot({ label, ok, detail }: { label: string; ok: boolean; detail?: string }) {
  return (
    <Stack direction="row" spacing={1} alignItems="center">
      <Box
        sx={{
          width: 10, height: 10, borderRadius: '50%',
          bgcolor: ok ? 'success.main' : 'error.main',
          boxShadow: ok
            ? `0 0 0 4px ${alpha('#2E7D32', 0.15)}`
            : `0 0 0 4px ${alpha('#D32F2F', 0.15)}`,
          animation: 'aperture-pulse 2.4s ease-in-out infinite',
        }}
      />
      <Typography variant="body2">{label}</Typography>
      {detail && <Typography variant="caption" color="text.secondary">({detail})</Typography>}
    </Stack>
  )
}

/** Animated pipeline: traveling Heliconia dot sweeps across the flow. */
function PipelineDiagram() {
  const steps = [
    { label: 'Query', color: NPR.midnight },
    { label: 'Controller', color: NPR.jayBlue },
    { label: 'Bandit Router', color: NPR.heliconia },
    { label: 'Retriever', color: NPR.jayBlue },
    { label: 'DQN Calibrator', color: NPR.heliconia },
    { label: 'Synthesizer', color: NPR.jayBlue },
  ]

  return (
    <Box sx={{ position: 'relative', mt: 1.5, py: 2 }}>
      {/* Connector line */}
      <Box
        sx={{
          position: 'absolute', left: '4%', right: '4%', top: '50%',
          height: 2, bgcolor: NPR.border,
          transform: 'translateY(-50%)', borderRadius: 2,
        }}
      />

      {/* Traveling dot */}
      <Box
        sx={{
          position: 'absolute', top: '50%', left: '4%',
          width: 10, height: 10, borderRadius: '50%',
          bgcolor: NPR.heliconia,
          boxShadow: `0 0 14px 2px ${alpha(NPR.heliconia, 0.6)}`,
          transform: 'translate(-50%, -50%)',
          '--flow-distance': '100%',
          animation: 'flow-dot 3.6s ease-in-out infinite',
        }}
      />

      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ position: 'relative', zIndex: 2 }}>
        {steps.map((s, i) => (
          <Box key={s.label} sx={{ textAlign: 'center', flex: 1 }}>
            <Box
              sx={{
                width: 34, height: 34, mx: 'auto', borderRadius: '50%',
                bgcolor: NPR.white,
                border: `2px solid ${s.color}`,
                display: 'grid', placeItems: 'center',
                fontWeight: 700, color: s.color, fontSize: 13,
                position: 'relative',
                animation: `aperture-pulse ${2.2 + i * 0.2}s ease-in-out infinite`,
                '&::after': {
                  content: `"${i + 1}"`,
                },
              }}
            />
            <Typography variant="caption" sx={{ mt: 0.5, display: 'block', color: 'text.secondary', fontWeight: 500 }}>
              {s.label}
            </Typography>
          </Box>
        ))}
      </Stack>
      <Divider sx={{ mt: 2 }} />
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        A query flows left-to-right: the Controller delegates, the bandit routes across 4 strategies,
        the retriever scores candidates, the DQN calibrates confidence, and the synthesizer returns a graded answer.
      </Typography>
    </Box>
  )
}
