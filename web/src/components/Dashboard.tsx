import { useEffect, useState } from 'react'
import {
  Paper, Stack, Typography, Box, Skeleton, Chip, Divider,
  Table, TableHead, TableBody, TableRow, TableCell, TableContainer,
} from '@mui/material'
import Grid from '@mui/material/Grid2'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, CartesianGrid,
} from 'recharts'
import { motion } from 'framer-motion'
import { fetchJson } from '../lib/api'
import { NPR } from '../theme'

const CHART_COLORS = [NPR.heliconia, NPR.jayBlue, '#2E7D32', '#ED6C02', '#9C27B0', '#00838F', '#E91E63', '#5E35B1']

interface EvalSummary {
  retrieval_accuracy: number
  routing_accuracy: number
  silent_failure_rate: number
  decline_accuracy: number
  avg_latency_s: number
  total_queries: number
}

interface EvalResult {
  query: string
  category: string
  retrieval_correct: boolean
  routing_correct: boolean
  confidence_grade: string
  silent_failure: boolean
  latency_s: number
}

export default function Dashboard() {
  const [evalData, setEvalData] = useState<{ summary: EvalSummary; results: EvalResult[] } | null>(null)
  const [ablation, setAblation] = useState<Record<string, unknown> | null>(null)
  const [figures, setFigures] = useState<{ name: string; filename: string; format: string }[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      fetchJson<{ summary: EvalSummary; results: EvalResult[] }>('/eval/results'),
      fetchJson<Record<string, unknown>>('/eval/ablation'),
      fetchJson<{ figures: { name: string; filename: string; format: string }[] }>('/figures'),
    ]).then(([evalRes, abRes, figRes]) => {
      if (evalRes.status === 'fulfilled') setEvalData(evalRes.value)
      if (abRes.status === 'fulfilled') setAblation(abRes.value)
      if (figRes.status === 'fulfilled') setFigures(figRes.value.figures)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <Stack spacing={2}>
        <Grid container spacing={2}>
          {Array.from({ length: 5 }).map((_, i) => (
            <Grid size={{ xs: 6, md: 2.4 }} key={i}>
              <Skeleton variant="rounded" height={96} animation="wave" />
            </Grid>
          ))}
        </Grid>
        <Skeleton variant="rounded" height={280} animation="wave" />
      </Stack>
    )
  }

  const summary = evalData?.summary
  const results = evalData?.results || []

  const categoryData = (() => {
    const cats: Record<string, { total: number; correct: number }> = {}
    results.forEach((r) => {
      if (!cats[r.category]) cats[r.category] = { total: 0, correct: 0 }
      cats[r.category].total++
      if (r.retrieval_correct) cats[r.category].correct++
    })
    return Object.entries(cats).map(([name, d]) => ({
      name,
      accuracy: Math.round((d.correct / d.total) * 100),
      total: d.total,
    }))
  })()

  const gradeData = (() => {
    const grades: Record<string, number> = {}
    results.forEach((r) => { grades[r.confidence_grade] = (grades[r.confidence_grade] || 0) + 1 })
    return Object.entries(grades).map(([name, value]) => ({ name, value }))
  })()

  const ablationRows: { name: string; retrieval_accuracy?: number; routing_accuracy?: number; silent_failure_rate?: number; decline_accuracy?: number }[] = ablation
    ? Object.entries(ablation).map(([name, data]) => {
        const d = data as { mean?: Record<string, number> }
        return { name, ...d.mean }
      })
    : []

  return (
    <Stack spacing={3}>
      {summary && (
        <Grid container spacing={2}>
          <Grid size={{ xs: 6, md: 2.4 }}><MetricCard label="Retrieval Accuracy" value={`${(summary.retrieval_accuracy * 100).toFixed(1)}%`} good={summary.retrieval_accuracy >= 0.9} /></Grid>
          <Grid size={{ xs: 6, md: 2.4 }}><MetricCard label="Routing Accuracy" value={`${(summary.routing_accuracy * 100).toFixed(1)}%`} good={summary.routing_accuracy >= 0.8} /></Grid>
          <Grid size={{ xs: 6, md: 2.4 }}><MetricCard label="Silent Failure Rate" value={`${(summary.silent_failure_rate * 100).toFixed(1)}%`} good={summary.silent_failure_rate <= 0.05} invert /></Grid>
          <Grid size={{ xs: 6, md: 2.4 }}><MetricCard label="Decline Accuracy" value={`${(summary.decline_accuracy * 100).toFixed(1)}%`} good={summary.decline_accuracy >= 0.9} /></Grid>
          <Grid size={{ xs: 6, md: 2.4 }}><MetricCard label="Avg Latency" value={`${summary.avg_latency_s.toFixed(1)}s`} good={summary.avg_latency_s < 30} /></Grid>
        </Grid>
      )}

      <Grid container spacing={2}>
        {categoryData.length > 0 && (
          <Grid size={{ xs: 12, md: 6 }}>
            <Paper sx={{ p: 2, borderRadius: 3 }}>
              <Typography variant="overline" color="text.secondary">Accuracy by Query Category</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={categoryData} margin={{ top: 16, right: 8, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={NPR.border} vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: NPR.midnight, fontSize: 12 }} axisLine={{ stroke: NPR.border }} tickLine={false} />
                  <YAxis tick={{ fill: NPR.midnight, fontSize: 12 }} domain={[0, 100]} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: NPR.white, border: `1px solid ${NPR.border}`, borderRadius: 8, color: NPR.midnight }}
                    formatter={(v) => `${v}%`}
                  />
                  <Bar dataKey="accuracy" fill={NPR.heliconia} radius={[6, 6, 0, 0]} animationDuration={900} />
                </BarChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}

        {gradeData.length > 0 && (
          <Grid size={{ xs: 12, md: 6 }}>
            <Paper sx={{ p: 2, borderRadius: 3 }}>
              <Typography variant="overline" color="text.secondary">Confidence Grade Distribution</Typography>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={gradeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`} animationDuration={900}>
                    {gradeData.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Legend />
                  <Tooltip contentStyle={{ background: NPR.white, border: `1px solid ${NPR.border}`, borderRadius: 8, color: NPR.midnight }} />
                </PieChart>
              </ResponsiveContainer>
            </Paper>
          </Grid>
        )}
      </Grid>

      {ablationRows.length > 0 && (
        <Paper sx={{ p: 2, borderRadius: 3 }}>
          <Typography variant="overline" color="text.secondary">Ablation Study Results (Mean across 5 seeds)</Typography>
          <TableContainer sx={{ mt: 1 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700 }}>Configuration</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Retrieval Acc</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Routing Acc</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Silent Failure</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>Decline Acc</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {ablationRows.map((row, i) => (
                  <TableRow
                    key={row.name}
                    component={motion.tr}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                    sx={{ '&:hover': { background: NPR.surface } }}
                  >
                    <TableCell sx={{ fontFamily: 'ui-monospace, monospace', color: NPR.heliconia }}>{row.name}</TableCell>
                    <TableCell align="right">{fmtPct(row.retrieval_accuracy)}</TableCell>
                    <TableCell align="right">{fmtPct(row.routing_accuracy)}</TableCell>
                    <TableCell align="right">{fmtPct(row.silent_failure_rate)}</TableCell>
                    <TableCell align="right">{fmtPct(row.decline_accuracy)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {figures.length > 0 && (
        <Paper sx={{ p: 2, borderRadius: 3 }}>
          <Typography variant="overline" color="text.secondary">Training &amp; Evaluation Figures</Typography>
          <Grid container spacing={2} sx={{ mt: 0.5 }}>
            {figures.filter((f) => f.format === 'png').map((fig, i) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={fig.filename}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <Paper sx={{ overflow: 'hidden', borderRadius: 2 }}>
                    <Box component="img" src={`/api/figures/${fig.filename}`} alt={fig.name} loading="lazy" sx={{ width: '100%', display: 'block', background: NPR.surface }} />
                    <Divider />
                    <Typography variant="caption" color="text.secondary" sx={{ px: 1.5, py: 1, display: 'block' }}>
                      {fig.name.replace(/_/g, ' ')}
                    </Typography>
                  </Paper>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}
    </Stack>
  )
}

function MetricCard({ label, value, good, invert }: { label: string; value: string; good: boolean; invert?: boolean }) {
  const color = invert
    ? good ? 'success.main' : 'error.main'
    : good ? 'success.main' : 'warning.main'
  return (
    <Paper sx={{ p: 2, borderRadius: 3, height: '100%' }}>
      <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        {label}
      </Typography>
      <Typography variant="h5" sx={{ fontWeight: 800, color }}>
        {value}
      </Typography>
      <Chip
        size="small"
        label={good ? (invert ? 'low ✓' : 'on target ✓') : 'watch'}
        sx={{ mt: 0.5, height: 20, fontSize: 11, bgcolor: good ? 'transparent' : NPR.surface }}
        variant="outlined"
      />
    </Paper>
  )
}

function fmtPct(v: unknown): string {
  if (typeof v !== 'number') return '—'
  return `${(v * 100).toFixed(1)}%`
}
