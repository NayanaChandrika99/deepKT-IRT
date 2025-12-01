// ABOUTME: Attention weight heatmap across query/key positions
// ABOUTME: Section 1.6 - Interactive 2D visualization of attention patterns

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class AttentionHeatmap {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('attention_heatmap.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const matrix = data.data.matrix;
    const queries = data.data.query_positions;
    const keys = data.data.key_positions;

    const trace = {
      z: matrix,
      x: keys,
      y: queries,
      type: 'heatmap',
      colorscale: 'Viridis',
      hovertemplate: 'Query: %{y}<br>Key: %{x}<br>Weight: %{z:.3f}<extra></extra>'
    };

    const layout = PlotlyHelpers.getLayout('Attention Weights', {
      xaxis: { title: 'Key Position' },
      yaxis: { title: 'Query Position' },
      height: this.options.height
    });

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

export default AttentionHeatmap;
