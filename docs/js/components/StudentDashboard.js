// ABOUTME: Student dashboard with metrics, mastery distribution, and recent activity
// ABOUTME: Section 1.1 - Comprehensive student overview with bar chart and table

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';
import MetricCard from '../utils/MetricCard.js';

class StudentDashboard {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      animationDuration: 300,
      ...options
    };
    this.chart = null;
    this.metrics = {};
  }

  async render(studentId = null) {
    const data = await DataLoader.load('student_dashboard.json');

    if (!data || !data.data) {
      console.error('Invalid student dashboard data');
      return this;
    }

    const studentData = studentId
      ? data.data.find(s => s.user_id === studentId)
      : data.data[0];

    if (!studentData) {
      console.error(`Student ${studentId} not found`);
      return this;
    }

    // Render metrics
    this.renderMetrics(studentData);

    // Render mastery distribution
    await this.renderMasteryDistribution(studentData);

    // Render recent activity table
    this.renderRecentActivity(studentData);

    return this;
  }

  renderMetrics(studentData) {
    // Create metric cards if they don't exist
    if (!this.metrics.avgMastery) {
      this.metrics.avgMastery = new MetricCard('student-avg-mastery', {
        title: 'Average Mastery',
        formatter: (v) => PlotlyHelpers.formatNumber(v, 'percent')
      });
    }
    if (!this.metrics.totalSkills) {
      this.metrics.totalSkills = new MetricCard('student-total-skills', {
        title: 'Total Skills',
        formatter: (v) => v.toString()
      });
    }
    if (!this.metrics.confidence) {
      this.metrics.confidence = new MetricCard('student-confidence', {
        title: 'Confidence Score',
        formatter: (v) => PlotlyHelpers.formatNumber(v, 'percent')
      });
    }

    // Update values
    this.metrics.avgMastery.render(studentData.avg_mastery || 0);
    this.metrics.totalSkills.render(studentData.total_skills || 0);
    this.metrics.confidence.render(studentData.confidence_score || 0);
  }

  async renderMasteryDistribution(studentData) {
    const container = document.getElementById(`${this.containerId}-chart`);
    if (!container) return;

    const distribution = studentData.skill_distribution || studentData.mastery_distribution || [];

    const trace = {
      x: distribution.map((_, i) => `${(i * 0.05).toFixed(2)}-${((i + 1) * 0.05).toFixed(2)}`),
      y: distribution,
      type: 'bar',
      marker: {
        color: PlotlyHelpers.colors.primary,
        line: { color: '#2171b5', width: 1 }
      },
      hovertemplate: '<b>Mastery Range:</b> %{x}<br><b>Skill Count:</b> %{y}<extra></extra>'
    };

    const layout = PlotlyHelpers.getLayout('Skill Mastery Distribution', {
      xaxis: { title: 'Mastery Range' },
      yaxis: { title: 'Number of Skills' },
      height: 300
    });

    const config = PlotlyHelpers.getConfig({ filename: 'mastery_distribution' });

    if (this.chart) {
      await Plotly.react(container, [trace], layout, config);
    } else {
      await Plotly.newPlot(container, [trace], layout, config);
      this.chart = container;
    }
  }

  renderRecentActivity(studentData) {
    const container = document.getElementById(`${this.containerId}-activity`);
    if (!container) return;

    const activities = studentData.recent_activity || [];

    if (activities.length === 0) {
      container.innerHTML = '<p class="no-data">No recent activity</p>';
      return;
    }

    const tableHtml = `
      <table class="activity-table">
        <thead>
          <tr>
            <th>Item ID</th>
            <th>Skill</th>
            <th>Correctness</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          ${activities.map(a => `
            <tr>
              <td>${a.item_id || 'N/A'}</td>
              <td>${a.skill || a.skill_name || a.skill_id || 'N/A'}</td>
              <td class="correctness ${a.correct !== undefined ? (a.correct ? 'correct' : 'incorrect') : ''}">
                ${a.correct !== undefined ? (a.correct ? '✅' : '❌') : (a.mastery !== undefined ? a.mastery.toFixed(2) : 'N/A')}
              </td>
              <td>${a.timestamp || 'N/A'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;

    container.innerHTML = tableHtml;
  }

  async update(studentId) {
    return await this.render(studentId);
  }

  destroy() {
    if (this.chart) {
      Plotly.purge(this.chart);
      this.chart = null;
    }
    Object.values(this.metrics).forEach(m => m.destroy());
    this.metrics = {};
  }
}

export default StudentDashboard;
