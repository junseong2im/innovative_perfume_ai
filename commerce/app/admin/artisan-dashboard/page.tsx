'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface DashboardStats {
  totalRecipes: number;
  todayRecipes: number;
  activeUsers: number;
  averageScore: number;
  popularNotes: { name: string; count: number }[];
  recentRecipes: any[];
}

export default function ArtisanDashboard() {
  const [stats, setStats] = useState<DashboardStats>({
    totalRecipes: 1247,
    todayRecipes: 42,
    activeUsers: 189,
    averageScore: 8.7,
    popularNotes: [
      { name: 'Rose', count: 234 },
      { name: 'Bergamot', count: 198 },
      { name: 'Sandalwood', count: 176 },
      { name: 'Jasmine', count: 145 },
      { name: 'Vanilla', count: 132 }
    ],
    recentRecipes: []
  });

  const [selectedPeriod, setSelectedPeriod] = useState('7days');

  // Chart Data
  const lineChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Recipes Created',
        data: [32, 45, 38, 52, 48, 61, 42],
        borderColor: 'rgb(147, 51, 234)',
        backgroundColor: 'rgba(147, 51, 234, 0.1)',
        tension: 0.4,
        fill: true
      },
      {
        label: 'User Sessions',
        data: [120, 145, 132, 168, 155, 189, 142],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true
      }
    ]
  };

  const barChartData = {
    labels: stats.popularNotes.map(n => n.name),
    datasets: [
      {
        label: 'Usage Count',
        data: stats.popularNotes.map(n => n.count),
        backgroundColor: [
          'rgba(147, 51, 234, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(16, 185, 129, 0.8)',
          'rgba(251, 146, 60, 0.8)',
          'rgba(236, 72, 153, 0.8)'
        ]
      }
    ]
  };

  const doughnutData = {
    labels: ['Floral', 'Woody', 'Citrus', 'Oriental', 'Fresh'],
    datasets: [
      {
        data: [35, 25, 20, 15, 5],
        backgroundColor: [
          'rgba(147, 51, 234, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(16, 185, 129, 0.8)',
          'rgba(251, 146, 60, 0.8)',
          'rgba(236, 72, 153, 0.8)'
        ],
        borderWidth: 0
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        cornerRadius: 8,
        titleFont: {
          size: 14,
          weight: 'normal'
        },
        bodyFont: {
          size: 13
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-display text-gray-900">Artisan Dashboard</h1>
              <p className="text-sm text-gray-500 mt-1">AI Perfume Generation Analytics</p>
            </div>
            <div className="flex items-center space-x-4">
              <select
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm focus:outline-none focus:border-purple-500"
              >
                <option value="today">Today</option>
                <option value="7days">Last 7 Days</option>
                <option value="30days">Last 30 Days</option>
                <option value="all">All Time</option>
              </select>
              <button className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 transition-colors">
                Export Report
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="p-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {[
            {
              label: 'Total Recipes',
              value: stats.totalRecipes.toLocaleString(),
              change: '+12.5%',
              trend: 'up',
              icon: '🎨'
            },
            {
              label: 'Today\'s Creations',
              value: stats.todayRecipes,
              change: '+8.2%',
              trend: 'up',
              icon: '✨'
            },
            {
              label: 'Active Users',
              value: stats.activeUsers,
              change: '+23.1%',
              trend: 'up',
              icon: '👥'
            },
            {
              label: 'Avg. Quality Score',
              value: stats.averageScore.toFixed(1),
              change: '+0.3',
              trend: 'up',
              icon: '⭐'
            }
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="text-2xl">{stat.icon}</div>
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  stat.trend === 'up'
                    ? 'bg-green-100 text-green-600'
                    : 'bg-red-100 text-red-600'
                }`}>
                  {stat.change}
                </span>
              </div>
              <h3 className="text-3xl font-display text-gray-900 mb-1">
                {stat.value}
              </h3>
              <p className="text-sm text-gray-500">{stat.label}</p>
            </motion.div>
          ))}
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Activity Chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2 bg-white rounded-xl p-6 shadow-sm"
          >
            <h3 className="text-lg font-display text-gray-900 mb-6">Activity Overview</h3>
            <div className="h-80">
              <Line data={lineChartData} options={chartOptions} />
            </div>
          </motion.div>

          {/* Family Distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-sm"
          >
            <h3 className="text-lg font-display text-gray-900 mb-6">Fragrance Families</h3>
            <div className="h-80 flex items-center justify-center">
              <div className="w-64 h-64">
                <Doughnut data={doughnutData} options={{ ...chartOptions, maintainAspectRatio: true }} />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Popular Notes */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-white rounded-xl p-6 shadow-sm"
          >
            <h3 className="text-lg font-display text-gray-900 mb-6">Most Used Notes</h3>
            <div className="h-64">
              <Bar data={barChartData} options={chartOptions} />
            </div>
          </motion.div>

          {/* Recent Recipes */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-white rounded-xl p-6 shadow-sm"
          >
            <h3 className="text-lg font-display text-gray-900 mb-6">Recent Creations</h3>
            <div className="space-y-4 max-h-64 overflow-y-auto">
              {[
                { id: 1, name: 'Midnight Rose', user: 'Sarah K.', score: 9.2, time: '5 mins ago' },
                { id: 2, name: 'Ocean Breeze', user: 'James L.', score: 8.7, time: '12 mins ago' },
                { id: 3, name: 'Golden Hour', user: 'Emma W.', score: 9.5, time: '23 mins ago' },
                { id: 4, name: 'Forest Dream', user: 'Alex M.', score: 8.3, time: '45 mins ago' },
                { id: 5, name: 'Velvet Touch', user: 'Sophia R.', score: 9.0, time: '1 hour ago' }
              ].map((recipe) => (
                <div key={recipe.id} className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 transition-colors">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{recipe.name}</h4>
                    <p className="text-sm text-gray-500">{recipe.user}</p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center space-x-1 text-sm">
                      <span className="text-yellow-500">⭐</span>
                      <span className="font-medium">{recipe.score}</span>
                    </div>
                    <p className="text-xs text-gray-400">{recipe.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}