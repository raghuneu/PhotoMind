import { useEffect, useRef, useState, type ReactNode } from 'react'
import { AnimatePresence } from 'framer-motion'
import { pingHealth, HAS_REMOTE_BACKEND } from '../lib/api'
import BackendWakeLoader from './BackendWakeLoader'

interface Props {
  children: ReactNode
}

/**
 * Gates the app behind a health probe while the backend wakes up.
 *
 * Render free-tier dynos cold-start in ~30-60s after inactivity. This
 * component polls /api/health and shows an engaging loader overlay until
 * the backend responds. For warm starts the first ping resolves <500ms
 * and the loader is never shown.
 *
 * When VITE_API_BASE_URL is unset (local dev against same host) the gate
 * is a pass-through.
 */
export default function BackendGate({ children }: Props) {
  // If no remote backend configured, skip the gate entirely.
  const [ready, setReady] = useState<boolean>(!HAS_REMOTE_BACKEND)
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef<number>(Date.now())
  const cancelledRef = useRef(false)
  const pollCountRef = useRef(0)
  const retryTriggerRef = useRef(0)
  const [, setRetryTick] = useState(0)

  useEffect(() => {
    if (!HAS_REMOTE_BACKEND) return
    cancelledRef.current = false

    const MAX_TOTAL_MS = 120_000

    const loop = async () => {
      while (!cancelledRef.current) {
        pollCountRef.current += 1
        const res = await pingHealth(pollCountRef.current === 1 ? 3500 : 5000)
        if (cancelledRef.current) return
        if (res.ok) {
          setReady(true)
          return
        }
        if (Date.now() - startRef.current > MAX_TOTAL_MS) {
          // Give up polling but let the user keep retrying manually.
          await new Promise<void>((resolve) => {
            const check = () => {
              if (cancelledRef.current || retryTriggerRef.current > 0) resolve()
              else setTimeout(check, 400)
            }
            check()
          })
          retryTriggerRef.current = 0
          continue
        }
        // Backoff: 1.5s → 2.5s → 4s → 5s cap, reset on manual retry.
        const n = pollCountRef.current
        const delay = Math.min(5000, 1500 + n * 500)
        await new Promise((r) => setTimeout(r, delay))
      }
    }

    loop()

    const tick = setInterval(() => {
      setElapsed((Date.now() - startRef.current) / 1000)
    }, 500)

    return () => {
      cancelledRef.current = true
      clearInterval(tick)
    }
  }, [])

  const handleRetry = () => {
    retryTriggerRef.current += 1
    pollCountRef.current = 0
    startRef.current = Date.now()
    setElapsed(0)
    setRetryTick((t) => t + 1)
  }

  return (
    <>
      {ready && children}
      <AnimatePresence>
        {!ready && <BackendWakeLoader key="wake" elapsedSec={elapsed} onRetry={handleRetry} />}
      </AnimatePresence>
    </>
  )
}
