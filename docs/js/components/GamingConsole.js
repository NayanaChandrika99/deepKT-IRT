// ABOUTME: Gaming detection alerts and metrics
// ABOUTME: Section 1.5 - Alert panel identifying suspicious student behavior

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class GamingConsole {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('gaming_alerts.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const alerts = data.data;

    container.innerHTML = `
      <div class="gaming-console">
        <h4>Gaming Detection Alerts</h4>
        <div class="alert-summary">
          <span>Flagged Students: ${data.metadata?.total_flagged || 0}</span>
        </div>
        <table class="gaming-table">
          <thead>
            <tr>
              <th>User ID</th>
              <th>Rapid Guess Rate</th>
              <th>Help Abuse %</th>
              <th>Severity</th>
            </tr>
          </thead>
          <tbody>
            ${alerts.map(a => `
              <tr class="severity-${a.severity}">
                <td>${a.user_id}</td>
                <td>${(a.rapid_guess_rate * 100).toFixed(1)}%</td>
                <td>${(a.help_abuse_pct * 100).toFixed(1)}%</td>
                <td><span class="badge ${a.severity}">${a.severity}</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;

    return this;
  }

  async update(studentId) {
    return await this.render(studentId);
  }

  destroy() {
    if (this.chart) {
      Plotly.purge(this.chart);
      this.chart = null;
    }
  }
}

export default GamingConsole;
