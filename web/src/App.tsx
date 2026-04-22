import { useState } from 'react'
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
          {activeTab === 'query' && <QueryPage />}
          {activeTab === 'kb' && <KBBrowser />}
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'architecture' && <Architecture />}
        </motion.div>
      </Container>
    </Box>
  )
}
