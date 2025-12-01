// ABOUTME: Explainability card showing influential past interactions
// ABOUTME: Section 1.3 - LLM-generated explanation with attention weights

import DataLoader from '../utils/DataLoader.js';

class ExplainabilityCard {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = options;
  }

  async render(studentId = null) {
    const data = await DataLoader.load('explainability_sample.json');

    const container = document.getElementById(this.containerId);
    if (!container || !data || !data.data) return this;

    const explanation = data.data;

    container.innerHTML = `
      <div class="explainability-card">
        <h4>Why this prediction?</h4>
        <p class="explanation-text">${explanation.explanation || 'No explanation available'}</p>

        ${explanation.influential_interactions ? `
          <h5>Top Influential Interactions:</h5>
          <ul class="influential-list">
            ${explanation.influential_interactions.slice(0, 5).map(interaction => `
              <li>
                <span class="item-id">${interaction.item_id}</span>
                <span class="correctness ${interaction.correct ? 'correct' : 'incorrect'}">
                  ${interaction.correct ? '✅' : '❌'}
                </span>
                <span class="attention-weight">${(interaction.attention_weight * 100).toFixed(1)}%</span>
                ${interaction.skill_name ? `<span class="skill-name">${interaction.skill_name}</span>` : ''}
              </li>
            `).join('')}
          </ul>
        ` : ''}
      </div>
    `;

    return this;
  }

  async update(studentId) {
    return await this.render(studentId);
  }

  destroy() {
    const container = document.getElementById(this.containerId);
    if (container) container.innerHTML = '';
  }
}

export default ExplainabilityCard;
