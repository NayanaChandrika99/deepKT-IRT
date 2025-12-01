// ABOUTME: Shared Plotly configuration and theming utilities
// ABOUTME: Ensures consistent styling and interaction patterns across all visualizations

const PlotlyHelpers = {
  // Standard layout configuration
  getLayout(title, options = {}) {
    return {
      title: {
        text: title,
        font: { size: 18, family: 'Inter, sans-serif' }
      },
      font: { family: 'Inter, sans-serif' },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#f8f9fa',
      margin: options.margin || { t: 60, r: 40, b: 60, l: 60 },
      hovermode: options.hovermode || 'closest',
      ...options
    };
  },

  // Standard config with demo-friendly features
  getConfig(options = {}) {
    return {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: options.filename || 'visualization',
        height: 800,
        width: 1200,
        scale: 2
      },
      ...options
    };
  },

  // Color palettes
  colors: {
    primary: '#4A90E2',
    secondary: '#7ED321',
    danger: '#D0021B',
    warning: '#F5A623',
    info: '#50E3C2',
    success: '#7ED321',
    sequential: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    diverging: ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'],
    categorical: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  },

  // Smooth animation configuration
  getTransition(duration = 500) {
    return {
      transition: {
        duration: duration,
        easing: 'cubic-in-out'
      },
      frame: {
        duration: duration,
        redraw: true
      }
    };
  },

  // Format numbers with appropriate precision
  formatNumber(value, type = 'auto') {
    if (value === null || value === undefined) return 'N/A';

    switch (type) {
      case 'percent':
        return `${(value * 100).toFixed(1)}%`;
      case 'decimal':
        return value.toFixed(3);
      case 'integer':
        return Math.round(value).toString();
      case 'auto':
      default:
        if (value >= 0 && value <= 1) {
          return value.toFixed(3);
        } else if (Number.isInteger(value)) {
          return value.toString();
        } else {
          return value.toFixed(2);
        }
    }
  },

  // Create hover template
  getHoverTemplate(fields) {
    return fields.map(f => `<b>${f.label}:</b> ${f.value}`).join('<br>') + '<extra></extra>';
  }
};

export default PlotlyHelpers;
