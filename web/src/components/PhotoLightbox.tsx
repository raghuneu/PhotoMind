import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  IconButton,
  Box,
  CircularProgress,
  Typography,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import { alpha } from '@mui/material/styles'

const NPR = {
  heliconia: '#F15B1C',
  jayBlue: '#4C85C5',
  midnight: '#00000F',
  white: '#FFFFFF',
}

interface PhotoLightboxProps {
  open: boolean
  onClose: () => void
  photoId: string | null
  thumbnailUrl?: string // Show thumbnail while loading full image
}

// In-memory cache for full-size images
const imageCache = new Map<string, { dataUrl: string; width: number; height: number }>()

export default function PhotoLightbox({ open, onClose, photoId, thumbnailUrl }: PhotoLightboxProps) {
  const [loading, setLoading] = useState(false)
  const [imageData, setImageData] = useState<{ dataUrl: string; width: number; height: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open || !photoId) {
      setImageData(null)
      setError(null)
      return
    }

    // Check cache first
    const cached = imageCache.get(photoId)
    if (cached) {
      setImageData(cached)
      setLoading(false)
      return
    }

    // Fetch full-size image
    setLoading(true)
    setError(null)

    const apiRoot = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
    fetch(`${apiRoot}/api/photos/${photoId}/image`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => {
        const result = { dataUrl: data.data_url, width: data.width, height: data.height }
        imageCache.set(photoId, result)
        setImageData(result)
      })
      .catch(err => {
        setError(err.message || 'Failed to load image')
      })
      .finally(() => {
        setLoading(false)
      })
  }, [open, photoId])

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth={false}
      PaperProps={{
        sx: {
          bgcolor: 'transparent',
          boxShadow: 'none',
          maxWidth: '90vw',
          maxHeight: '90vh',
        },
      }}
      slotProps={{
        backdrop: {
          sx: { bgcolor: alpha(NPR.midnight, 0.92) },
        },
      }}
    >
      <IconButton
        onClick={onClose}
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          color: NPR.white,
          bgcolor: alpha(NPR.midnight, 0.5),
          '&:hover': { bgcolor: alpha(NPR.midnight, 0.7) },
          zIndex: 10,
        }}
      >
        <CloseIcon />
      </IconButton>

      <DialogContent
        sx={{
          p: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minWidth: 300,
          minHeight: 300,
          overflow: 'hidden',
        }}
      >
        {loading && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            {/* Show thumbnail as preview while loading */}
            {thumbnailUrl && (
              <Box
                component="img"
                src={thumbnailUrl}
                alt="Loading preview"
                sx={{
                  maxWidth: '80vw',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                  opacity: 0.6,
                  filter: 'blur(2px)',
                }}
              />
            )}
            <CircularProgress sx={{ color: NPR.heliconia, position: 'absolute' }} />
          </Box>
        )}

        {error && (
          <Box sx={{ textAlign: 'center', color: NPR.white, p: 4 }}>
            <Typography variant="h6">Failed to load image</Typography>
            <Typography variant="body2" sx={{ opacity: 0.7, mt: 1 }}>{error}</Typography>
          </Box>
        )}

        {imageData && !loading && (
          <Box
            component="img"
            src={imageData.dataUrl}
            alt="Full size photo"
            sx={{
              maxWidth: '90vw',
              maxHeight: '85vh',
              objectFit: 'contain',
              borderRadius: 1,
              boxShadow: `0 8px 32px ${alpha(NPR.midnight, 0.5)}`,
            }}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}
