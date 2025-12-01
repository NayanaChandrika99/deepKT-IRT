// ABOUTME: RL recommendation explorer with UCB metrics
// ABOUTME: Section 2.1 - Table showing expected rewards and exploration/exploitation

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class RLRecommendations {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('rl_recommendations.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const recommendations = data.data[0]?.recommendations || data.data;

    container.innerHTML = `
      <div class="rl-recommendations">
        <h4>RL Recommendations</h4>
        <table>
          <thead>
            <tr>
              <th>Item ID</th>
              <th>Expected Reward</th>
              <th>Uncertainty</th>
              <th>Mode</th>
            </tr>
          </thead>
          <tbody>
            ${recommendations.slice(0, 10).map(r => `
              <tr>
                <td>${r.item_id}</td>
                <td>${((r.expected || r.expected_reward) * 100).toFixed(1)}%</td>
                <td>${(r.uncertainty * 100).toFixed(1)}%</td>
                <td><span class="badge ${r.mode}">${r.mode}</span></td>
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

export default RLRecommendations;
