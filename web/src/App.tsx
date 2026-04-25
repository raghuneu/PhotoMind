import { useRef, useState } from 'react'
import {
  AppBar, Toolbar, Tabs, Tab, Box, Container, Typography, Stack,
} from '@mui/material'
import SearchIcon from '@mui/icons-material/Search'
import StorageIcon from '@mui/icons-material/Storage'
import InsightsIcon from '@mui/icons-material/Insights'
import MemoryIcon from '@mui/icons-material/Memory'
import { motion } from 'framer-motion'
import QueryPage from './components/QueryPage'
import KBBrowser from './components/KBBrowser'
import Dashboard from './components/Dashboard'
import Architecture from './components/Architecture'
import ApertureLogo from './components/ApertureLogo'
import Hero from './components/Hero'
import HowItWorks from './components/HowItWorks'
import PageHeader from './components/PageHeader'
import { NPR } from './theme'

const tabs = [
  { id: 'query', label: 'Query', icon: <SearchIcon fontSize="small" /> },
  { id: 'kb', label: 'Knowledge Base', icon: <StorageIcon fontSize="small" /> },
  { id: 'dashboard', label: 'Performance', icon: <InsightsIcon fontSize="small" /> },
  { id: 'architecture', label: 'Architecture', icon: <MemoryIcon fontSize="small" /> },
] as const

type TabId = (typeof tabs)[number]['id']

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('query')
  const demoRef = useRef<HTMLDivElement | null>(null)

  const scrollToDemo = () => {
    demoRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: NPR.white, color: NPR.midnight }}>
      <AppBar position="sticky">
        <Toolbar sx={{ maxWidth: 1200, width: '100%', mx: 'auto', gap: 2 }}>
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ flexGrow: 0 }}>
            <ApertureLogo size={36} />
            <Box>
              <Typography variant="h6" sx={{ lineHeight: 1.1, fontWeight: 700 }}>
                PhotoMind
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                RL-Powered Photo Knowledge Retrieval
              </Typography>
            </Box>
          </Stack>

          <Box sx={{ flexGrow: 1 }} />

          <Tabs
            value={activeTab}
            onChange={(_, v) => setActiveTab(v)}
            sx={{ minHeight: 48 }}
          >
            {tabs.map((t) => (
              <Tab
                key={t.id}
                value={t.id}
                iconPosition="start"
                icon={t.icon}
                label={<Box sx={{ display: { xs: 'none', sm: 'inline' } }}>{t.label}</Box>}
              />
            ))}
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: 'easeOut' }}
        >
          {activeTab === 'query' && (
            <>
              <Hero onCTAClick={scrollToDemo} />
              <HowItWorks />
              <Box ref={demoRef} sx={{ scrollMarginTop: 80 }}>
                <PageHeader
                  overline="Live demo"
                  title="Ask your photos anything."
                  subtitle="Try a question. Every answer comes back with the exact photos it was drawn from — and a grade telling you how much to trust it."
                />
                <QueryPage />
              </Box>
            </>
          )}
          {activeTab === 'kb' && (
            <>
              <PageHeader
                overline="Knowledge base"
                title="Every photo, indexed and searchable."
                subtitle="The photos PhotoMind reads before answering. Filter by type, search text, and expand any card to see extracted OCR and entities."
              />
              <KBBrowser />
            </>
          )}
          {activeTab === 'dashboard' && (
            <>
              <PageHeader
                overline="Performance"
                title="How well does it actually work?"
                subtitle="Evaluation results across 20 test queries, ablation studies over 5 seeds, and training curves from the DQN calibrator."
              />
              <Dashboard />
            </>
          )}
          {activeTab === 'architecture' && (
            <>
              <PageHeader
                overline="Architecture"
                title="Under the hood."
                subtitle="A hierarchical CrewAI system with a contextual bandit router, a DQN confidence calibrator, and four retrieval strategies."
              />
              <Architecture />
            </>
          )}
        </motion.div>
      </Container>
    </Box>
  )
}
