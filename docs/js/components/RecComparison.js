// ABOUTME: RL vs rule-based recommendation comparison
// ABOUTME: Section 2.3 - Side-by-side table highlighting differences

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class RecComparison {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('rec_comparison.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const compData = data.data[0] || data.data || {};
    const rlRecs = compData.rl_recommendations || [];
    const ruleRecs = compData.rule_based_recommendations || [];

    container.innerHTML = `
      <div class="rec-comparison">
        <h4>RL vs Rule-Based Recommendations</h4>
        <div class="comparison-grid">
          <div class="column">
            <h5>RL-Based</h5>
            <ul>${rlRecs.map(r => `<li>${typeof r === 'string' ? r : r.item_id}</li>`).join('')}</ul>
          </div>
          <div class="column">
            <h5>Rule-Based</h5>
            <ul>${ruleRecs.map(r => `<li>${typeof r === 'string' ? r : r.item_id}</li>`).join('')}</ul>
          </div>
        </div>
        <p class="overlap">Overlap: ${compData.overlap_percentage || compData.overlap_pct || 0}%</p>
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

export default RecComparison;
