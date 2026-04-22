import { useEffect, useState } from 'react'
import {
  Paper, Stack, Typography, Box, TextField, Chip, Skeleton, Divider,
  Collapse,
} from '@mui/material'
import Grid from '@mui/material/Grid2'
import ImageIcon from '@mui/icons-material/Image'
import LabelIcon from '@mui/icons-material/LocalOffer'
import DescriptionIcon from '@mui/icons-material/Description'
import InsightsIcon from '@mui/icons-material/Insights'
import { motion } from 'framer-motion'
import { fetchJson, type Photo, type KBStats } from '../lib/api'
import { NPR } from '../theme'

export default function KBBrowser() {
  const [photos, setPhotos] = useState<Photo[]>([])
  const [stats, setStats] = useState<KBStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [selectedType, setSelectedType] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetchJson<{ photos: Photo[] }>('/knowledge-base'),
      fetchJson<KBStats>('/knowledge-base/stats'),
    ])
      .then(([kb, s]) => { setPhotos(kb.photos); setStats(s); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const filtered = photos.filter((p) => {
    const q = filter.toLowerCase()
    const matchText = !q
      || p.description.toLowerCase().includes(q)
      || p.filename.toLowerCase().includes(q)
      || p.ocr_text.toLowerCase().includes(q)
    const matchType = !selectedType || p.image_type === selectedType
    return matchText && matchType
  })

  if (loading) {
    return (
      <Stack spacing={2}>
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} variant="rounded" height={88} animation="wave" />
        ))}
      </Stack>
    )
  }

  return (
    <Stack spacing={3}>
      {stats && (
        <Grid container spacing={2}>
          <Grid size={{ xs: 6, md: 3 }}><StatCard label="Total Photos" value={stats.total_photos} icon={<ImageIcon />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><StatCard label="Total Entities" value={stats.total_entities} icon={<LabelIcon />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><StatCard label="With OCR Text" value={stats.has_ocr} icon={<DescriptionIcon />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><StatCard label="Avg Entities/Photo" value={stats.avg_entities_per_photo} icon={<InsightsIcon />} /></Grid>
        </Grid>
      )}

      {stats && (
        <Paper sx={{ p: 2, borderRadius: 3 }}>
          <Typography variant="overline" color="text.secondary">Photo Types</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            <Chip
              label={`All (${photos.length})`}
              color={!selectedType ? 'primary' : 'default'}
              variant={!selectedType ? 'filled' : 'outlined'}
              onClick={() => setSelectedType(null)}
            />
            {Object.entries(stats.type_distribution).map(([type, count]) => (
              <Chip
                key={type}
                label={`${type} (${count})`}
                color={selectedType === type ? 'primary' : 'default'}
                variant={selectedType === type ? 'filled' : 'outlined'}
                onClick={() => setSelectedType(type === selectedType ? null : type)}
              />
            ))}
          </Box>
        </Paper>
      )}

      <TextField
        fullWidth
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter photos by description, filename, or OCR text…"
      />

      <Grid container spacing={2}>
        {filtered.map((photo, idx) => {
          const isExpanded = expandedId === photo.id
          return (
            <Grid size={{ xs: 12, md: 6 }} key={photo.id}>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(idx, 12) * 0.03 }}
              >
                <Paper
                  sx={{
                    p: 2, borderRadius: 3, cursor: 'pointer',
                    transition: 'all .18s ease',
                    '&:hover': { borderColor: NPR.heliconia },
                  }}
                  onClick={() => setExpandedId(isExpanded ? null : photo.id)}
                >
                  <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 13, color: NPR.heliconia }}>
                        {photo.filename}
                      </Typography>
                      <Chip size="small" label={photo.image_type} variant="outlined" />
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      {(photo.confidence * 100).toFixed(0)}% conf
                    </Typography>
                  </Stack>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {photo.description}
                  </Typography>

                  <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                    <Divider sx={{ my: 1.5 }} />
                    <Stack spacing={1.5}>
                      {photo.ocr_text && (
                        <Box>
                          <Typography variant="overline" color="text.secondary">OCR Text</Typography>
                          <Box sx={{ fontFamily: 'ui-monospace, monospace', fontSize: 13, bgcolor: NPR.surface, p: 1, borderRadius: 1, mt: 0.5 }}>
                            {photo.ocr_text}
                          </Box>
                        </Box>
                      )}
                      {photo.entities.length > 0 && (
                        <Box>
                          <Typography variant="overline" color="text.secondary">Entities</Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                            {photo.entities.map((e, i) => (
                              <Chip
                                key={i}
                                size="small"
                                label={<><span style={{ color: 'rgba(0,0,15,0.5)' }}>{e.type}:</span> {e.value}</>}
                                variant="outlined"
                              />
                            ))}
                          </Box>
                        </Box>
                      )}
                      <Typography variant="caption" sx={{ fontFamily: 'ui-monospace, monospace', color: 'text.secondary' }}>
                        {photo.file_path}
                      </Typography>
                    </Stack>
                  </Collapse>
                </Paper>
              </motion.div>
            </Grid>
          )
        })}
      </Grid>
    </Stack>
  )
}

function StatCard({ label, value, icon }: { label: string; value: number; icon: React.ReactNode }) {
  return (
    <Paper sx={{ p: 2, borderRadius: 3, height: '100%' }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ color: 'text.secondary', mb: 0.5 }}>
        {icon}
        <Typography variant="overline">{label}</Typography>
      </Stack>
      <Typography variant="h4" sx={{ fontWeight: 700, color: NPR.midnight }}>
        {typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(1) : value}
      </Typography>
    </Paper>
  )
}
