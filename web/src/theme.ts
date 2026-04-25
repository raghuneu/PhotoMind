import { createTheme, alpha } from '@mui/material/styles'

// NPR Brand Colors
export const NPR = {
  heliconia: '#F15B1C',
  jayBlue: '#4C85C5',
  midnight: '#00000F',
  white: '#FFFFFF',
  surface: '#F7F7F8',
  surfaceAlt: '#EFEFF2',
  border: 'rgba(0, 0, 15, 0.12)',
} as const

export const nprTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: NPR.heliconia,
      light: '#F7845A',
      dark: '#C44710',
      contrastText: NPR.white,
    },
    secondary: {
      main: NPR.jayBlue,
      light: '#7BA7D8',
      dark: '#36649A',
      contrastText: NPR.white,
    },
    background: {
      default: NPR.white,
      paper: NPR.white,
    },
    text: {
      primary: NPR.midnight,
      secondary: alpha(NPR.midnight, 0.6),
      disabled: alpha(NPR.midnight, 0.38),
    },
    divider: NPR.border,
    success: { main: '#2E7D32' },
    warning: { main: '#ED6C02' },
    error:   { main: '#D32F2F' },
    info:    { main: NPR.jayBlue },
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    h1: { fontWeight: 700, letterSpacing: '-0.02em' },
    h2: { fontWeight: 700, letterSpacing: '-0.02em' },
    h3: { fontWeight: 700, letterSpacing: '-0.01em' },
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
    button: { textTransform: 'none', fontWeight: 600 },
  },
  components: {
    MuiPaper: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          border: `1px solid ${NPR.border}`,
        },
      },
    },
    MuiAppBar: {
      defaultProps: { elevation: 0, color: 'default' },
      styleOverrides: {
        root: {
          background: alpha(NPR.white, 0.85),
          backdropFilter: 'saturate(180%) blur(12px)',
          borderBottom: `1px solid ${NPR.border}`,
          color: NPR.midnight,
        },
      },
    },
    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: { borderRadius: 10 },
        containedPrimary: {
          background: `linear-gradient(135deg, ${NPR.heliconia} 0%, #FF7A42 100%)`,
          '&:hover': {
            background: `linear-gradient(135deg, #D94F13 0%, ${NPR.heliconia} 100%)`,
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          minHeight: 48,
          color: alpha(NPR.midnight, 0.6),
          '&.Mui-selected': { color: NPR.heliconia },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          height: 3,
          borderRadius: 3,
          background: NPR.heliconia,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontWeight: 500 },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          border: `1px solid ${NPR.border}`,
          transition: 'border-color .18s ease, box-shadow .18s ease, transform .18s ease',
          '&:hover': {
            borderColor: alpha(NPR.heliconia, 0.5),
            boxShadow: `0 8px 24px -12px ${alpha(NPR.heliconia, 0.25)}`,
          },
        },
      },
    },
    MuiTextField: {
      defaultProps: { variant: 'outlined' },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          background: NPR.white,
          '& .MuiOutlinedInput-notchedOutline': { borderColor: NPR.border },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: alpha(NPR.midnight, 0.3) },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: NPR.heliconia,
            borderWidth: 2,
          },
        },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          border: `1px solid ${NPR.border}`,
          '&.Mui-selected': {
            background: alpha(NPR.jayBlue, 0.1),
            color: NPR.jayBlue,
            borderColor: alpha(NPR.jayBlue, 0.4),
            '&:hover': { background: alpha(NPR.jayBlue, 0.15) },
          },
        },
      },
    },
  },
})

// Grade → color map (used by QueryPage)
export const GRADE_PALETTE: Record<string, { fg: string; bg: string; border: string; label: string }> = {
  A: { fg: NPR.jayBlue, bg: alpha(NPR.jayBlue, 0.1),   border: alpha(NPR.jayBlue, 0.4),  label: 'High confidence' },
  B: { fg: '#2E7D32',   bg: alpha('#2E7D32', 0.1),     border: alpha('#2E7D32', 0.4),    label: 'Good' },
  C: { fg: '#ED6C02',   bg: alpha('#ED6C02', 0.1),     border: alpha('#ED6C02', 0.4),    label: 'Moderate' },
  D: { fg: NPR.heliconia, bg: alpha(NPR.heliconia, 0.1), border: alpha(NPR.heliconia, 0.4), label: 'Low confidence' },
  F: { fg: '#D32F2F',   bg: alpha('#D32F2F', 0.1),     border: alpha('#D32F2F', 0.4),    label: 'Declined' },
}
