// ABOUTME: Placeholder for GitHub Pages interactivity (Plotly/D3 hooks).

document.addEventListener("DOMContentLoaded", async () => {
  const loadJSON = async (path) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}`);
    return response.json();
  };

  try {
    const [pipelineData, skillData, attentionData, recData, modelData, opsData] = await Promise.all([
      loadJSON("data/pipeline_sample.json"),
      loadJSON("data/skill_sample.json"),
      loadJSON("data/attention_sample.json"),
      loadJSON("data/recommendation_sample.json"),
      loadJSON("data/model_metrics.json"),
      loadJSON("data/ops_summary.json"),
    ]);

    renderPipelineSankey(pipelineData);
    renderStudentInsights(skillData, attentionData);
    renderRecommendationSection(recData);
    renderModelSection(modelData);
    renderOpsSection(pipelineData, opsData);
  } catch (error) {
    console.warn("Failed to load dashboard data", error);
  }
});

function renderPipelineSankey(pipelineData) {
  if (!window.Plotly || !pipelineData.length) return;
  const total = pipelineData.length;
  const half = Math.round(total * 0.5);

  Plotly.newPlot(
    "pipeline-viz",
    [
      {
        type: "sankey",
        node: {
          pad: 15,
          thickness: 20,
          label: ["Raw CSV", "Canonical", "SAKT Prep", "WD-IRT Prep"],
          color: ["#3a7bd5", "#58a5f0", "#7fd1ae", "#f8c291"],
        },
        link: {
          source: [0, 1, 1],
          target: [1, 2, 3],
          value: [total, half, total - half],
        },
      },
    ],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#f5f5f5" },
    }
  );
}

function renderStudentInsights(skillData, attentionData) {
  if (!window.Plotly) return;
  const mastery = skillData.map((row) => row.mastery_mean).filter((v) => typeof v === "number");
  const skillNames = skillData.slice(0, 6).map((row) => row.skill);
  const skillValues = skillData.slice(0, 6).map((row) => row.mastery_mean);

  const histTrace = {
    x: mastery,
    type: "histogram",
    marker: { color: "#7fd1ae" },
    opacity: 0.75,
  };

  const radarTrace = {
    type: "scatterpolar",
    r: skillValues,
    theta: skillNames,
    fill: "toself",
    name: "Mastery",
    marker: { color: "#58a5f0" },
    subplot: "polar",
  };

  const flattenedInfluences = attentionData
    .flatMap((row) => {
      try {
        if (Array.isArray(row.top_influences)) return row.top_influences;
        if (typeof row.top_influences === "string") return JSON.parse(row.top_influences);
      } catch (err) {
        return [];
      }
      return [];
    })
    .slice(0, 10);

  if (flattenedInfluences.length) {
    const barTrace = {
      x: flattenedInfluences.map((inf) => inf.weight),
      y: flattenedInfluences.map((inf) => inf.item_id),
      type: "bar",
      orientation: "h",
      marker: { color: "#f8c291" },
    };
    Plotly.newPlot(
      "student-viz",
      [histTrace, radarTrace, barTrace],
      {
        grid: { rows: 3, columns: 1, pattern: "independent" },
        polar: { radialaxis: { visible: true, range: [0, 1] } },
        paper_bgcolor: "transparent",
        plot_bgcolor: "#1b1f2a",
        font: { color: "#f5f5f5" },
        title: "Mastery distribution, radar snapshot, attention weights",
      }
    );
  } else {
    Plotly.newPlot(
      "student-viz",
      [histTrace, radarTrace],
      {
        polar: { radialaxis: { visible: true, range: [0, 1] } },
        paper_bgcolor: "transparent",
        plot_bgcolor: "#1b1f2a",
        font: { color: "#f5f5f5" },
        title: "Mastery distribution + radar snapshot",
      }
    );
  }
}

function renderRecommendationSection(recData) {
  if (!window.Plotly) return;
  const container = document.getElementById("recommendation-viz");
  if (!container || !recData.length) return;

  container.innerHTML = `
    <div id="rec-scatter"></div>
    <div class="metric-grid" id="rec-metrics"></div>
    <div id="rec-gauge"></div>
  `;

  const scatter = {
    x: recData.map((d) => d.expected),
    y: recData.map((d) => d.uncertainty),
    text: recData.map((d) => d.item),
    mode: "markers",
    marker: { size: 14, color: recData.map((d) => (d.mode === "explore" ? "#f8c291" : "#7fd1ae")) },
  };
  Plotly.newPlot("rec-scatter", [scatter], {
    title: "Expected reward vs. uncertainty (LinUCB sample)",
    xaxis: { title: "Expected reward" },
    yaxis: { title: "Uncertainty" },
    paper_bgcolor: "transparent",
    plot_bgcolor: "#1b1f2a",
    font: { color: "#f5f5f5" },
  });

  const grid = document.getElementById("rec-metrics");
  grid.innerHTML = recData
    .map(
      (d) => `
        <div class="metric">
          <span>${d.mode.toUpperCase()}</span>
          <strong>${(d.expected * 100).toFixed(0)}%</strong>
          <small>uncertainty ${(d.uncertainty * 100).toFixed(0)}%</small>
        </div>
      `
    )
    .join("");

  const first = recData[0];
  Plotly.newPlot("rec-gauge", [
    {
      type: "indicator",
      mode: "gauge+number",
      value: first.expected - first.uncertainty,
      title: { text: "Confidence margin (Top rec)" },
      gauge: { axis: { range: [0, 1] } },
    },
  ], {
    paper_bgcolor: "transparent",
    font: { color: "#f5f5f5" },
  });
}

function renderModelSection(modelData) {
  if (!window.Plotly || !modelData.length) return;
  const epochs = modelData.map((d) => d.epoch);
  const sakt_auc = modelData.map((d) => d.sakt_auc);
  const wd_auc = modelData.map((d) => d.wd_auc);

  Plotly.newPlot(
    "model-viz",
    [
      { x: epochs, y: sakt_auc, type: "scatter", mode: "lines+markers", name: "SAKT AUC" },
      { x: epochs, y: wd_auc, type: "scatter", mode: "lines+markers", name: "WD-IRT AUC" },
    ],
    {
      title: "Training performance (sample)",
      xaxis: { title: "Epoch" },
      yaxis: { title: "AUC", range: [0.5, 0.8] },
      paper_bgcolor: "transparent",
      plot_bgcolor: "#1b1f2a",
      font: { color: "#f5f5f5" },
    }
  );
}

function renderOpsSection(pipelineData, opsData) {
  const container = document.getElementById("ops-viz");
  if (!container) return;
  const total = pipelineData.length;
  container.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><span>Total canonical events</span><strong>${total.toLocaleString()}</strong></div>
      <div class="metric"><span>SAKT prep (sample)</span><strong>${Math.round(total * 0.5).toLocaleString()}</strong></div>
      <div class="metric"><span>WD-IRT prep (sample)</span><strong>${Math.round(total * 0.5).toLocaleString()}</strong></div>
      <div class="metric"><span>Joinable predictions</span><strong>${(opsData?.joinable_predictions || 0).toLocaleString()}</strong></div>
      <div class="metric"><span>Attention rows</span><strong>${(opsData?.attention_rows || 0).toLocaleString()}</strong></div>
    </div>
    <p>Sample counts sourced from <code>data/interim/edm_cup_2023_42_events.parquet</code>. Update by re-exporting JSONs via <code>docs/scripts/export_docs_assets.py</code> once implemented.</p>
  `;
}
