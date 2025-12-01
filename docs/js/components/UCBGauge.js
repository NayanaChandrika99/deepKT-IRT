// ABOUTME: UCB confidence gauge for top recommendation
// ABOUTME: Section 2.2 - Animated gauge showing confidence level

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class UCBGauge {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('rl_recommendations.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const topRec = data.data[0];
    const confidence = (topRec.expected_reward - topRec.uncertainty) * 100;

    const trace = {
      type: 'indicator',
      mode: 'gauge+number',
      value: confidence,
      title: { text: 'Top Recommendation Confidence' },
      gauge: {
        axis: { range: [0, 100] },
        bar: { color: PlotlyHelpers.colors.primary },
        steps: [
          { range: [0, 30], color: '#fee090' },
          { range: [30, 70], color: '#abd9e9' },
          { range: [70, 100], color: '#7ED321' }
        ]
      }
    };

    const layout = { height: this.options.height };

    if (this.chart) {
      await Plotly.react(container, [trace], layout, PlotlyHelpers.getConfig());
    } else {
      await Plotly.newPlot(container, [trace], layout, PlotlyHelpers.getConfig());
      this.chart = container;
    }

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

export default UCBGauge;
