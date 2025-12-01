// ABOUTME: Generic plot component for simple visualizations
// ABOUTME: Flexible component that can render various plot types from JSON data

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class GenericPlot {
  constructor(containerId, dataFile, title, options = {}) {
    this.containerId = containerId;
    this.dataFile = dataFile;
    this.title = title;
    this.options = { height: 400, plotType: 'line', ...options };
    this.chart = null;
  }

  async render() {
    const data = await DataLoader.load(this.dataFile);
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const traces = this.createTraces(data.data);
    const layout = PlotlyHelpers.getLayout(this.title, {
      height: this.options.height,
      ...this.options.layoutOptions
    });
    const config = PlotlyHelpers.getConfig({ filename: this.dataFile.replace('.json', '') });

    if (this.chart) {
      await Plotly.react(container, traces, layout, config);
    } else {
      await Plotly.newPlot(container, traces, layout, config);
      this.chart = container;
    }

    return this;
  }

  createTraces(data) {
    // Default: line chart
    if (Array.isArray(data) && data.length > 0) {
      return [{
        x: data.map((d, i) => d.x || i),
        y: data.map(d => d.y || 0),
        type: this.options.plotType,
        name: this.title
      }];
    }
    return [];
  }

  async update() {
    return await this.render();
  }

  destroy() {
    if (this.chart) {
      Plotly.purge(this.chart);
      this.chart = null;
    }
  }
}

export default GenericPlot;
