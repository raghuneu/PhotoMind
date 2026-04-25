import { useEffect, useState, useCallback } from 'react'
import {
  Paper, Stack, Typography, Box, TextField, Chip, Skeleton, Divider,
  Collapse,
} from '@mui/material'
import Grid from '@mui/material/Grid2'
import ImageIcon from '@mui/icons-material/Image'
import LabelIcon from '@mui/icons-material/LocalOffer'
import DescriptionIcon from '@mui/icons-material/Description'
import InsightsIcon from '@mui/icons-material/Insights'
import ZoomInIcon from '@mui/icons-material/ZoomIn'
import ShieldOutlinedIcon from '@mui/icons-material/ShieldOutlined'
import { motion } from 'framer-motion'
import { fetchJson, type Photo, type KBStats } from '../lib/api'
import { NPR } from '../theme'
import MetricCard from './MetricCard'
import PhotoLightbox from './PhotoLightbox'

export default function KBBrowser() {
  const [photos, setPhotos] = useState<Photo[]>([])
  const [stats, setStats] = useState<KBStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [selectedType, setSelectedType] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [thumbs, setThumbs] = useState<Record<string, string>>({})
  const [lightboxId, setLightboxId] = useState<string | null>(null)

  const handleImageClick = useCallback((e: React.MouseEvent, photoId: string) => {
    e.stopPropagation() // Don't toggle expand when clicking image
    setLightboxId(photoId)
  }, [])

  useEffect(() => {
    Promise.all([
      fetchJson<{ photos: Photo[] }>('/knowledge-base'),
      fetchJson<KBStats>('/knowledge-base/stats'),
    ])
      .then(async ([kb, s]) => {
        setPhotos(kb.photos)
        setStats(s)
        setLoading(false)
        const results = await Promise.allSettled(
          kb.photos.map((p) =>
            fetchJson<{ photo_id: string; data_url: string }>(`/photos/${p.id}/thumbnail`),
          ),
        )
        const map: Record<string, string> = {}
        results.forEach((r) => {
          if (r.status === 'fulfilled') map[r.value.photo_id] = r.value.data_url
        })
        setThumbs(map)
      })
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
          <Grid size={{ xs: 6, md: 3 }}><MetricCard label="Total Photos" value={stats.total_photos} icon={<ImageIcon fontSize="small" />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><MetricCard label="Total Entities" value={stats.total_entities} icon={<LabelIcon fontSize="small" />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><MetricCard label="With OCR Text" value={stats.has_ocr} icon={<DescriptionIcon fontSize="small" />} /></Grid>
          <Grid size={{ xs: 6, md: 3 }}><MetricCard label="Avg Entities/Photo" value={stats.avg_entities_per_photo} icon={<InsightsIcon fontSize="small" />} /></Grid>
        </Grid>
      )}

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
        <ShieldOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        <Typography variant="caption" color="text.disabled">
          No PII was used in this demonstration corpus. Production use would require ingestion-time redaction.
        </Typography>
      </Box>

      {stats && (
        <Paper sx={{ p: 2, borderRadius: 2 }}>
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
          const thumb = thumbs[photo.id]
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
                    '&:hover': { borderColor: NPR.jayBlue },
                  }}
                  onClick={() => setExpandedId(isExpanded ? null : photo.id)}
                >
                  <Stack direction="row" spacing={1.5} alignItems="flex-start">
                    {thumb ? (
                      <Box
                        sx={{ position: 'relative', flexShrink: 0, cursor: 'zoom-in' }}
                        onClick={(e) => handleImageClick(e, photo.id)}
                      >
                        <Box
                          component="img"
                          src={thumb}
                          alt={photo.filename}
                          loading="lazy"
                          sx={{
                            width: 64, height: 64, objectFit: 'cover',
                            borderRadius: 1.5,
                            border: `1px solid ${NPR.border}`,
                            transition: 'transform 0.15s ease',
                            '&:hover': { transform: 'scale(1.05)' },
                          }}
                        />
                        <Box
                          sx={{
                            position: 'absolute', inset: 0,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            bgcolor: 'rgba(0,0,15,0.4)', borderRadius: 1.5,
                            opacity: 0, transition: 'opacity 0.15s ease',
                            '&:hover': { opacity: 1 },
                          }}
                        >
                          <ZoomInIcon sx={{ color: NPR.white, fontSize: 20 }} />
                        </Box>
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          width: 64, height: 64, borderRadius: 1.5, flexShrink: 0,
                          bgcolor: NPR.surface, border: `1px solid ${NPR.border}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          color: 'text.secondary',
                        }}
                      >
                        <ImageIcon fontSize="small" />
                      </Box>
                    )}
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={1}>
                        <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0, flex: 1 }}>
                          <Typography
                            sx={{
                              fontFamily: 'ui-monospace, monospace',
                              fontSize: 13,
                              color: NPR.jayBlue,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {photo.filename}
                          </Typography>
                          <Chip size="small" label={photo.image_type} variant="outlined" />
                        </Stack>
                        <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
                          {(photo.confidence * 100).toFixed(0)}% conf
                        </Typography>
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                        {photo.description}
                      </Typography>
                    </Box>
                  </Stack>

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

      <PhotoLightbox
        open={!!lightboxId}
        onClose={() => setLightboxId(null)}
        photoId={lightboxId}
        thumbnailUrl={lightboxId ? thumbs[lightboxId] : undefined}
      />
    </Stack>
  )
}
