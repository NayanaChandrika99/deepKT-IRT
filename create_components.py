#!/usr/bin/env python3
"""Generate all dashboard component JS files"""

COMPONENT_TEMPLATE = """// ABOUTME: {description}
// ABOUTME: Section {section} - {detail}

import DataLoader from '../utils/DataLoader.js';
import PlotlyHelpers from '../utils/PlotlyHelpers.js';

class {class_name} {{
  constructor(containerId, options = {{}}) {{
    this.containerId = containerId;
    this.options = {{ height: 400, ...options }};
    this.chart = null;
  }}

  async render(studentId = null) {{
    const data = await DataLoader.load('{data_file}');
    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    {render_logic}

    return this;
  }}

  async update(studentId) {{
    return await this.render(studentId);
  }}

  destroy() {{
    if (this.chart) {{
      Plotly.purge(this.chart);
      this.chart = null;
    }}
  }}
}}

export default {class_name};
"""

COMPONENTS = [
    # Section 1 - remaining
    {
        'name': 'GamingConsole',
        'section': '1.5',
        'description': 'Gaming detection alerts and metrics',
        'detail': 'Alert panel identifying suspicious student behavior',
        'data_file': 'gaming_alerts.json',
        'render_logic': '''const alerts = data.data;

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
    `;'''
    },
    {
        'name': 'AttentionHeatmap',
        'section': '1.6',
        'description': 'Attention weight heatmap across query/key positions',
        'detail': 'Interactive 2D visualization of attention patterns',
        'data_file': 'attention_heatmap.json',
        'render_logic': '''const matrix = data.data.matrix;
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
    }'''
    },

    # Section 2
    {
        'name': 'RLRecommendations',
        'section': '2.1',
        'description': 'RL recommendation explorer with UCB metrics',
        'detail': 'Table showing expected rewards and exploration/exploitation',
        'data_file': 'rl_recommendations.json',
        'render_logic': '''const recommendations = data.data;

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
                <td>${(r.expected_reward * 100).toFixed(1)}%</td>
                <td>${(r.uncertainty * 100).toFixed(1)}%</td>
                <td><span class="badge ${r.mode}">${r.mode}</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;'''
    },
    {
        'name': 'UCBGauge',
        'section': '2.2',
        'description': 'UCB confidence gauge for top recommendation',
        'detail': 'Animated gauge showing confidence level',
        'data_file': 'rl_recommendations.json',
        'render_logic': '''const topRec = data.data[0];
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
    }'''
    },
    {
        'name': 'RecComparison',
        'section': '2.3',
        'description': 'RL vs rule-based recommendation comparison',
        'detail': 'Side-by-side table highlighting differences',
        'data_file': 'rec_comparison.json',
        'render_logic': '''const rlRecs = data.data.rl_recommendations || [];
    const ruleRecs = data.data.rule_based_recommendations || [];

    container.innerHTML = `
      <div class="rec-comparison">
        <h4>RL vs Rule-Based Recommendations</h4>
        <div class="comparison-grid">
          <div class="column">
            <h5>RL-Based</h5>
            <ul>${rlRecs.map(r => `<li>${r.item_id}</li>`).join('')}</ul>
          </div>
          <div class="column">
            <h5>Rule-Based</h5>
            <ul>${ruleRecs.map(r => `<li>${r.item_id}</li>`).join('')}</ul>
          </div>
        </div>
        <p class="overlap">Overlap: ${data.data.overlap_pct || 0}%</p>
      </div>
    `;'''
    }
]

import os
from pathlib import Path

output_dir = Path('docs/js/components')
output_dir.mkdir(parents=True, exist_ok=True)

for comp in COMPONENTS:
    comp['class_name'] = comp['name']
    filename = output_dir / f"{comp['name']}.js"
    content = COMPONENT_TEMPLATE.format(**comp)
    filename.write_text(content)
    print(f"✓ Created {filename}")

print(f"\n✓ Generated {len(COMPONENTS)} components")
