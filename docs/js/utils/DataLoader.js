// ABOUTME: Centralized data loading with caching and error handling
// ABOUTME: Demonstrates production-ready async patterns and graceful degradation

class DataLoader {
  constructor() {
    this.cache = new Map();
    this.baseUrl = 'data/';
  }

  async load(filename) {
    // Check cache first
    if (this.cache.has(filename)) {
      console.log(`📦 Cache hit: ${filename}`);
      return this.cache.get(filename);
    }

    try {
      console.log(`🔄 Loading: ${filename}`);
      const response = await fetch(`${this.baseUrl}${filename}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      this.cache.set(filename, data);
      console.log(`✓ Loaded: ${filename}`);
      return data;

    } catch (error) {
      console.error(`✗ Failed to load ${filename}:`, error);

      // Try mock fallback
      try {
        const mockResponse = await fetch(`${this.baseUrl}mocks/${filename}`);
        if (mockResponse.ok) {
          const mockData = await mockResponse.json();
          console.log(`⚠️  Using mock data for ${filename}`);
          this.cache.set(filename, mockData);
          return mockData;
        }
      } catch (mockError) {
        console.error(`✗ Mock fallback also failed for ${filename}`);
      }

      // Return empty data structure to prevent crashes
      return { data: [], metadata: { mock: true, error: error.message } };
    }
  }

  clearCache() {
    this.cache.clear();
    console.log('🗑️  Cache cleared');
  }

  getCacheStats() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }
}

// Singleton instance
const dataLoader = new DataLoader();
export default dataLoader;
