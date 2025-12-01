// ABOUTME: Main dashboard orchestration with modular components
// ABOUTME: Loads and initializes all 24 visualizations with lazy loading

import DataLoader from './utils/DataLoader.js';
import StudentDashboard from './components/StudentDashboard.js';
import MasteryTimeline from './components/MasteryTimeline.js';
import ExplainabilityCard from './components/ExplainabilityCard.js';
import SkillRadar from './components/SkillRadar.js';
import GamingConsole from './components/GamingConsole.js';
import AttentionHeatmap from './components/AttentionHeatmap.js';
import RLRecommendations from './components/RLRecommendations.js';
import UCBGauge from './components/UCBGauge.js';
import RecComparison from './components/RecComparison.js';
import GenericPlot from './components/GenericPlot.js';

class DashboardApp {
  constructor() {
    this.components = {};
    this.currentStudent = null;
  }

  async init() {
    console.log('🚀 Initializing dashboard...');

    try {
      // Initialize Section 1: Student Insights
      await this.initSection1();

      // Initialize Section 2: Recommendations
      await this.initSection2();

      // Initialize Section 3: Model Performance
      await this.initSection3();

      // Initialize Section 4: Data Quality
      await this.initSection4();

      // Initialize Section 5: Pipeline Health
      await this.initSection5();

      console.log('✅ Dashboard initialized successfully');
      this.logCacheStats();

    } catch (error) {
      console.error('❌ Dashboard initialization failed:', error);
    }
  }

  async initSection1() {
    console.log('📊 Section 1: Student Insights');

    // 1.1 Student Dashboard
    this.components.studentDashboard = new StudentDashboard('student-dashboard');
    await this.components.studentDashboard.render();

    // 1.2 Mastery Timeline
    this.components.masteryTimeline = new MasteryTimeline('mastery-timeline');
    await this.components.masteryTimeline.render();

    // 1.3 Explainability Card
    this.components.explainabilityCard = new ExplainabilityCard('explainability-card');
    await this.components.explainabilityCard.render();

    // 1.4 Skill Radar
    this.components.skillRadar = new SkillRadar('skill-radar');
    await this.components.skillRadar.render();

    // 1.5 Gaming Console
    this.components.gamingConsole = new GamingConsole('gaming-console');
    await this.components.gamingConsole.render();

    // 1.6 Attention Heatmap
    this.components.attentionHeatmap = new AttentionHeatmap('attention-heatmap');
    await this.components.attentionHeatmap.render();
  }

  async initSection2() {
    console.log('🎯 Section 2: Recommendations');

    // 2.1 RL Recommendations
    this.components.rlRecommendations = new RLRecommendations('rl-recommendations');
    await this.components.rlRecommendations.render();

    // 2.2 UCB Gauge
    this.components.ucbGauge = new UCBGauge('ucb-gauge');
    await this.components.ucbGauge.render();

    // 2.3 Recommendation Comparison
    this.components.recComparison = new RecComparison('rec-comparison');
    await this.components.recComparison.render();
  }

  async initSection3() {
    console.log('📈 Section 3: Model Performance');

    // 3.1 Training Dashboard
    this.components.trainingMetrics = new GenericPlot(
      'training-metrics',
      'training_metrics.json',
      'Training Metrics'
    );
    await this.components.trainingMetrics.render();

    // 3.2 Attention Network
    this.components.attentionNetwork = new GenericPlot(
      'attention-network',
      'attention_network.json',
      'Attention Network'
    );
    await this.components.attentionNetwork.render();

    // 3.3 Item Health
    this.components.itemHealth = new GenericPlot(
      'item-health',
      'item_health.json',
      'Item Health Dashboard'
    );
    await this.components.itemHealth.render();

    // 3.4 Training Curves
    this.components.trainingCurves = new GenericPlot(
      'training-curves',
      'training_metrics.json',
      'Training Curves (Deep Dive)'
    );
    await this.components.trainingCurves.render();

    // 3.5 Feature Importance
    this.components.featureImportance = new GenericPlot(
      'feature-importance',
      'feature_importance.json',
      'Feature Importance (WD-IRT)'
    );
    await this.components.featureImportance.render();
  }

  async initSection4() {
    console.log('🔍 Section 4: Data Quality');

    // 4.1 Pipeline Flow
    this.components.pipelineFlow = new GenericPlot(
      'pipeline-flow',
      'pipeline_flow.json',
      'Data Pipeline Flow'
    );
    await this.components.pipelineFlow.render();

    // 4.2 Coverage Heatmap
    this.components.coverageHeatmap = new GenericPlot(
      'coverage-heatmap',
      'coverage_heatmap.json',
      'User×Skill Coverage'
    );
    await this.components.coverageHeatmap.render();

    // 4.3 Sequence Quality
    this.components.sequenceQuality = new GenericPlot(
      'sequence-quality',
      'sequence_quality.json',
      'Sequence Quality Metrics'
    );
    await this.components.sequenceQuality.render();

    // 4.4 Split Integrity
    this.components.splitIntegrity = new GenericPlot(
      'split-integrity',
      'split_integrity.json',
      'Train/Val/Test Split Integrity'
    );
    await this.components.splitIntegrity.render();

    // 4.5 Schema Validation
    this.components.schemaValidation = new GenericPlot(
      'schema-validation',
      'schema_validation.json',
      'Schema Validation Results'
    );
    await this.components.schemaValidation.render();

    // 4.6 Joinability Gauge
    this.components.joinabilityGauge = new GenericPlot(
      'joinability-gauge',
      'joinability_gauge.json',
      'Joinability Score'
    );
    await this.components.joinabilityGauge.render();
  }

  async initSection5() {
    console.log('⚙️ Section 5: Pipeline Health');

    // 5.1 Lineage Map
    this.components.lineageMap = new GenericPlot(
      'lineage-map',
      'lineage_map.json',
      'Data Lineage Map'
    );
    await this.components.lineageMap.render();

    // 5.2 Throughput Monitoring
    this.components.throughputMonitoring = new GenericPlot(
      'throughput-monitoring',
      'throughput_monitoring.json',
      'Pipeline Throughput'
    );
    await this.components.throughputMonitoring.render();

    // 5.3 Join Overview
    this.components.joinOverview = new GenericPlot(
      'join-overview',
      'join_overview.json',
      'Join Overlap Analysis'
    );
    await this.components.joinOverview.render();

    // 5.4 Drift Alerts
    this.components.driftAlerts = new GenericPlot(
      'drift-alerts',
      'drift_alerts.json',
      'Item Drift Alerts'
    );
    await this.components.driftAlerts.render();
  }

  logCacheStats() {
    const stats = DataLoader.getCacheStats();
    console.log(`📦 Cache stats: ${stats.size} files cached`);
    console.log('   Files:', stats.keys.join(', '));
  }

  destroy() {
    Object.values(this.components).forEach(component => {
      if (component && component.destroy) {
        component.destroy();
      }
    });
    this.components = {};
    DataLoader.clearCache();
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', async () => {
  const app = new DashboardApp();
  await app.init();

  // Make app globally available for debugging
  window.dashboardApp = app;
});
