import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Images,
  Search,
  BarChart3,
  DollarSign,
  PanelLeftClose,
  PanelLeft,
  Brain,
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/gallery', icon: Images, label: 'Photo Gallery' },
  { to: '/query', icon: Search, label: 'Query' },
  { to: '/eval', icon: BarChart3, label: 'Evaluation' },
  { to: '/spending', icon: DollarSign, label: 'Spending' },
]

export default function Sidebar({ open, onToggle }) {
  return (
    <>
      {/* Mobile toggle */}
      {!open && (
        <button
          className="sidebar-toggle-mobile"
          onClick={onToggle}
          aria-label="Open sidebar"
        >
          <PanelLeft size={20} />
        </button>
      )}

      <aside className={`sidebar ${open ? '' : 'sidebar-hidden'}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <Brain size={22} strokeWidth={2.2} />
            <span className="sidebar-title">PhotoMind</span>
          </div>
          <button className="sidebar-collapse-btn" onClick={onToggle} aria-label="Collapse sidebar">
            <PanelLeftClose size={18} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-footer-text">
            Multimodal Knowledge Retrieval
          </div>
        </div>
      </aside>
    </>
  )
}
