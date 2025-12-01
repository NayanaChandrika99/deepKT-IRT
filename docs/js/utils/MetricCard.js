// ABOUTME: Reusable metric display card with optional delta indicators
// ABOUTME: Consistent formatting for dashboard KPIs

class MetricCard {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      title: '',
      formatter: (v) => v,
      ...options
    };
  }

  render(value, delta = null) {
    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error(`Container ${this.containerId} not found`);
      return this;
    }

    container.className = 'metric-card';
    container.innerHTML = `
      <span class="metric-title">${this.options.title}</span>
      <strong class="metric-value">${this.options.formatter(value)}</strong>
      ${delta !== null ? `<small class="metric-delta ${delta >= 0 ? 'positive' : 'negative'}">${delta >= 0 ? '+' : ''}${delta}</small>` : ''}
    `;
    return this;
  }

  update(value, delta = null) {
    return this.render(value, delta);
  }

  destroy() {
    const container = document.getElementById(this.containerId);
    if (container) {
      container.innerHTML = '';
    }
  }
}

export default MetricCard;
