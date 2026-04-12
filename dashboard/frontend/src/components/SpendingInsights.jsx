import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis,
} from 'recharts'
import { DollarSign, TrendingUp, ShoppingCart, Receipt } from 'lucide-react'
import { getSpendingByVendor, getReceiptTimeline, getTopFoodItems } from '../utils/chartHelpers'

const COLORS = ['#5B8DEF', '#E8A838', '#E07B5B', '#4ECDC4', '#9B6FE3', '#22C55E', '#F97316', '#95A5B8']

export default function SpendingInsights({ kb }) {
  if (!kb || !kb.photos) {
    return <div className="page-header"><h1>No data loaded</h1></div>
  }

  const photos = kb.photos
  const vendorSpending = getSpendingByVendor(photos)
  const timeline = getReceiptTimeline(photos)
  const topItems = getTopFoodItems(photos)
  const totalSpent = vendorSpending.reduce((s, v) => s + v.value, 0)
  const receiptCount = photos.filter((p) => p.image_type === 'receipt').length

  return (
    <div>
      <div className="page-header">
        <h1>Spending Insights</h1>
        <p>Analytics from your receipt and purchase photos</p>
      </div>

      {/* Summary Stats */}
      <div className="stat-row">
        <div className="card stat-card">
          <span className="stat-label">
            <DollarSign size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Total Spent
          </span>
          <span className="stat-value">${totalSpent.toFixed(2)}</span>
          <span className="stat-sub">from extracted receipts</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">
            <Receipt size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Receipts
          </span>
          <span className="stat-value">{receiptCount}</span>
          <span className="stat-sub">of {photos.length} total photos</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">
            <ShoppingCart size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Vendors
          </span>
          <span className="stat-value">{vendorSpending.length}</span>
          <span className="stat-sub">unique stores</span>
        </div>
        <div className="card stat-card">
          <span className="stat-label">
            <TrendingUp size={14} style={{ marginRight: 4, verticalAlign: -2 }} />Avg per Receipt
          </span>
          <span className="stat-value">
            ${receiptCount > 0 ? (totalSpent / vendorSpending.length).toFixed(2) : '0.00'}
          </span>
          <span className="stat-sub">per vendor visit</span>
        </div>
      </div>

      {/* Charts Row */}
      <div className="chart-row">
        {/* Vendor Donut */}
        <div className="card chart-card">
          <h3>Spending by Vendor</h3>
          {vendorSpending.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={vendorSpending}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={100}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {vendorSpending.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: '#fff', border: '1px solid #E8E5E0',
                    borderRadius: 8, fontSize: 13,
                  }}
                  formatter={(val) => [`$${val.toFixed(2)}`, 'Amount']}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#9CA3AF' }}>
              No spending data extracted
            </div>
          )}
          {/* Legend */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8, justifyContent: 'center' }}>
            {vendorSpending.map((v, i) => (
              <div key={v.name} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem' }}>
                <div style={{
                  width: 10, height: 10, borderRadius: 3,
                  background: COLORS[i % COLORS.length],
                }} />
                <span style={{ color: '#6B7280' }}>{v.name}</span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, fontSize: '0.6875rem' }}>
                  ${v.value.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Purchase Timeline */}
        <div className="card chart-card">
          <h3>Purchase Timeline</h3>
          {timeline.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 8 }}>
              {timeline.map((t, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  {/* Date node */}
                  <div style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: COLORS[i % COLORS.length], flexShrink: 0,
                  }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 500 }}>
                      {t.vendor}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#9CA3AF' }}>
                      {t.date}
                    </div>
                  </div>
                  <span style={{
                    fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
                    fontSize: '0.875rem',
                  }}>
                    ${t.amount.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: 40, color: '#9CA3AF' }}>
              No dated receipts found
            </div>
          )}
        </div>
      </div>

      {/* Top Purchased Items */}
      <div className="card chart-card">
        <h3>Most Purchased Items</h3>
        {topItems.length > 0 ? (
          <ResponsiveContainer width="100%" height={Math.max(200, topItems.length * 28)}>
            <BarChart
              data={topItems}
              layout="vertical"
              margin={{ left: 10, right: 20 }}
            >
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="name"
                width={200}
                tick={{ fontSize: 11, fill: '#6B7280' }}
              />
              <Tooltip
                contentStyle={{
                  background: '#fff', border: '1px solid #E8E5E0',
                  borderRadius: 8, fontSize: 12,
                }}
              />
              <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={16} fill="#E8A838" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ textAlign: 'center', padding: 40, color: '#9CA3AF' }}>
            No food item data extracted
          </div>
        )}
      </div>
    </div>
  )
}
