// ABOUTME: Reusable dropdown component with consistent styling
// ABOUTME: Provides onChange callbacks for interactive filtering

class Dropdown {
  constructor(containerId, options = {}) {
    this.containerId = containerId;
    this.options = {
      label: '',
      onChange: () => {},
      ...options
    };
    this.element = null;
  }

  render(items, selectedValue = null) {
    const container = document.getElementById(this.containerId);
    if (!container) {
      console.error(`Container ${this.containerId} not found`);
      return this;
    }

    container.innerHTML = `
      <label class="dropdown-label">${this.options.label}</label>
      <select class="dropdown-select">
        ${items.map(item => `
          <option value="${item.value}" ${item.value === selectedValue ? 'selected' : ''}>
            ${item.label}
          </option>
        `).join('')}
      </select>
    `;

    this.element = container.querySelector('select');
    this.element.addEventListener('change', (e) => {
      this.options.onChange(e.target.value);
    });

    return this;
  }

  getValue() {
    return this.element ? this.element.value : null;
  }

  setValue(value) {
    if (this.element) {
      this.element.value = value;
      this.options.onChange(value);
    }
    return this;
  }

  destroy() {
    if (this.element) {
      this.element.remove();
      this.element = null;
    }
  }
}

export default Dropdown;
