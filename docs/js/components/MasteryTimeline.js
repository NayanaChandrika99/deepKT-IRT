// ABOUTME: Animated timeline showing student mastery evolution over sequence positions
// ABOUTME: Section 1.2 - Interactive time-series with play/pause controls

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class MasteryTimeline {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      animationDuration: 300,
      height: 400,
      ...options
    };
    this.chart = null;
    this.data = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('mastery_timeline.json');

    if (!data || !data.data) {
      console.error('Invalid mastery timeline data');
      return this;
    }

    this.data = data.data;

    // Group by skill_id
    const skillGroups = {};
    this.data.forEach(row => {
      const skillId = row.skill_id || 'unknown';
      if (!skillGroups[skillId]) {
        skillGroups[skillId] = [];
      }
      skillGroups[skillId].push(row);
    });

    // Create traces for each skill
    const traces = Object.entries(skillGroups).map(([skillId, rows]) => ({
      x: rows.map(r => r.sequence_position),
      y: rows.map(r => r.mastery_score),
      type: 'scatter',
      mode: 'lines+markers',
      name: `Skill ${skillId}`,
      line: { width: 2 },
      marker: { size: 6 },
      hovertemplate: `<b>Skill:</b> ${skillId}<br><b>Position:</b> %{x}<br><b>Mastery:</b> %{y:.3f}<extra></extra>`
    }));

    const layout = PlotlyHelpers.getLayout('Student Mastery Evolution', {
      xaxis: { title: 'Sequence Position' },
      yaxis: { title: 'Mastery Score', range: [0, 1] },
      height: this.options.height,
      showlegend: true,
      legend: { x: 1.05, y: 1 }
    });

    const config = PlotlyHelpers.getConfig({ filename: 'mastery_timeline' });

    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error(`Container ${this.containerId} not found`);
      return this;
    }

    if (this.chart) {
      await Plotly.react(container, traces, layout, config);
    } else {
      await Plotly.newPlot(container, traces, layout, config);
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

export default MasteryTimeline;
