// ABOUTME: Radar chart showing student mastery across top skills
// ABOUTME: Section 1.4 - Polar visualization of skill proficiency

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class SkillRadar {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = { height: 400, ...options };
    this.chart = null;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('skill_radar.json');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const skillData = data.data;

    const trace = {
      type: 'scatterpolar',
      r: skillData.map(s => s.mastery),
      theta: skillData.map(s => s.skill_name || s.skill_id),
      fill: 'toself',
      marker: { color: PlotlyHelpers.colors.primary },
      hovertemplate: '<b>%{theta}</b><br>Mastery: %{r:.3f}<extra></extra>'
    };

    const layout = PlotlyHelpers.getLayout('Skill Proficiency', {
      polar: { radialaxis: { visible: true, range: [0, 1] } },
      height: this.options.height
    });

    const config = PlotlyHelpers.getConfig({ filename: 'skill_radar' });

    if (this.chart) {
      await Plotly.react(container, [trace], layout, config);
    } else {
      await Plotly.newPlot(container, [trace], layout, config);
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

export default SkillRadar;
