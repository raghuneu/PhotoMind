import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThemeProvider, CssBaseline } from '@mui/material'
import { nprTheme } from './theme'
import './index.css'
import App from './App.tsx'
import BackendGate from './components/BackendGate'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={nprTheme}>
      <CssBaseline />
      <BackendGate>
        <App />
      </BackendGate>
    </ThemeProvider>
  </StrictMode>,
)
