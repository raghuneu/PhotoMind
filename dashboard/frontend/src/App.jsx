import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import DashboardHome from './components/DashboardHome'
import PhotoGallery from './components/PhotoGallery'
import QueryInterface from './components/QueryInterface'
import EvalResults from './components/EvalResults'
import SpendingInsights from './components/SpendingInsights'
import { fetchKnowledgeBase, fetchEvalResults } from './utils/api'
import './App.css'

export default function App() {
  const [kb, setKb] = useState(null)
  const [evalData, setEvalData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => {
    async function loadData() {
      try {
        const [kbData, evalRes] = await Promise.all([
          fetchKnowledgeBase(),
          fetchEvalResults(),
        ])
        setKb(kbData)
        setEvalData(evalRes)
      } catch (err) {
        console.error('Failed to load data:', err)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-pulse" />
        <p>Loading PhotoMind...</p>
      </div>
    )
  }

  return (
    <div className={`app-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DashboardHome kb={kb} evalData={evalData} />} />
          <Route path="/gallery" element={<PhotoGallery kb={kb} />} />
          <Route path="/query" element={<QueryInterface />} />
          <Route path="/eval" element={<EvalResults evalData={evalData} />} />
          <Route path="/spending" element={<SpendingInsights kb={kb} />} />
        </Routes>
      </main>
    </div>
  )
}
