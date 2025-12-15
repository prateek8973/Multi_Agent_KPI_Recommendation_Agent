import React, { useState, useEffect, useRef } from "react";
import { BarChart3, GitBranch, ThumbsUp, ThumbsDown } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./App.css";

export default function App() {
  // ===================== STATE VARIABLES =====================
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("Increase sales");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [collapsedSteps, setCollapsedSteps] = useState({});
  const [quality, setQuality] = useState(null);
  const [impacts, setImpacts] = useState(null);
  const [feedbackMsg, setFeedbackMsg] = useState("");
  const [dependencyGraph, setDependencyGraph] = useState(null);
  const canvasRef = useRef(null);

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#A855F7", "#F87171"];

  // ===================== HANDLERS =====================
  const toggleStep = (idx) => {
    setCollapsedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Upload CSV");

    const form = new FormData();
    form.append("file", file);
    form.append("goal_text", goal);

    setLoading(true);
    setFeedbackMsg("");
    setResult(null);
    setDependencyGraph(null);

    try {
      const res = await fetch("http://localhost:8000/orchestrate", {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setResult(data);
      setQuality(data.data_quality || null);
      setImpacts(data.impact_map || null);

      if (data.recommendations) {
        const graph = generateDependencyGraph(data.recommendations, data.profile);
        setDependencyGraph(graph);
      }
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===================== DEPENDENCY GRAPH GENERATION =====================
  const generateDependencyGraph = (kpis, profile) => {
    const nodes = [];
    const edges = [];
    const columns = profile?.columns || [];

    // Column nodes
    columns.forEach((col, idx) => {
      nodes.push({ id: `col_${col}`, label: col, type: "column", x: 100, y: 50 + idx * 60 });
    });

    // KPI nodes + edges from columns
    kpis.forEach((kpi, idx) => {
      const kpiId = `kpi_${idx}`;
      nodes.push({
        id: kpiId,
        label: kpi.name,
        type: "kpi",
        x: 400,
        y: 50 + idx * 80,
        kpiType: kpi.type,
      });

      const formula = kpi.formula || "";
      columns.forEach((col) => {
        if (formula.includes(`'${col}'`) || formula.includes(`"${col}"`)) {
          edges.push({ from: `col_${col}`, to: kpiId, label: "uses" });
        }
      });
    });

    // KPI-to-KPI dependency edges
    kpis.forEach((kpi1, i) => {
      kpis.forEach((kpi2, j) => {
        if (i !== j) {
          const formula = kpi2.formula || "";
          if (formula.toLowerCase().includes(kpi1.name.toLowerCase())) {
            edges.push({ from: `kpi_${i}`, to: `kpi_${j}`, label: "derives", isDerived: true });
          }
        }
      });
    });

    return { nodes, edges };
  };

  // ===================== CANVAS DRAWING =====================
  useEffect(() => {
    if (!dependencyGraph || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const { nodes, edges } = dependencyGraph;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw edges
    edges.forEach((edge) => {
      const fromNode = nodes.find((n) => n.id === edge.from);
      const toNode = nodes.find((n) => n.id === edge.to);
      if (fromNode && toNode) {
        ctx.beginPath();
        ctx.strokeStyle = edge.isDerived ? "#8b5cf6" : "#94a3b8";
        ctx.setLineDash(edge.isDerived ? [5, 5] : []);
        ctx.moveTo(fromNode.x + 80, fromNode.y + 20);
        ctx.lineTo(toNode.x, toNode.y + 20);
        ctx.stroke();

        // Draw arrow
        const angle = Math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x);
        ctx.beginPath();
        ctx.moveTo(toNode.x, toNode.y + 20);
        ctx.lineTo(toNode.x - 10 * Math.cos(angle - Math.PI / 6), toNode.y + 20 - 10 * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(toNode.x, toNode.y + 20);
        ctx.lineTo(toNode.x - 10 * Math.cos(angle + Math.PI / 6), toNode.y + 20 - 10 * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw nodes
    nodes.forEach((node) => {
      const nodeWidth = 160;
      const nodeHeight = 40;
      ctx.fillStyle = node.type === "column" ? "#dbeafe" : "#dcfce7";
      ctx.strokeStyle = node.type === "column" ? "#3b82f6" : "#22c55e";
      ctx.fillRect(node.x, node.y, nodeWidth, nodeHeight);
      ctx.strokeRect(node.x, node.y, nodeWidth, nodeHeight);

      ctx.fillStyle = "#1e293b";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const maxWidth = nodeWidth - 20;
      let displayText = node.label;
      while (ctx.measureText(displayText + "...").width > maxWidth && displayText.length > 0) {
        displayText = displayText.slice(0, -1);
      }
      ctx.fillText(displayText, node.x + 10, node.y + nodeHeight / 2);

      ctx.font = "9px sans-serif";
      ctx.fillStyle = "#64748b";
      ctx.fillText(node.type === "column" ? "DATA" : node.kpiType?.toUpperCase() || "KPI", node.x + 10, node.y + nodeHeight - 8);
    });
  }, [dependencyGraph]);

  // ===================== FEEDBACK HANDLERS =====================
  async function sendKpiFeedback(kpi_name, action) {
    setFeedbackMsg("Sending feedback...");
    try {
      const res = await fetch("http://localhost:8000/feedback/kpi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kpi_name, action }),
      });
      if (res.ok) setFeedbackMsg(`Feedback recorded for KPI: ${kpi_name}`);
      else setFeedbackMsg(`Failed to record feedback for KPI: ${kpi_name}`);
    } catch (err) {
      console.error(err);
      setFeedbackMsg(`Error sending feedback for KPI: ${kpi_name}`);
    }
    setTimeout(() => setFeedbackMsg(""), 3000);
  }

  async function sendImpactFeedback(item) {
    try {
      const res = await fetch("http://localhost:8000/feedback/issue", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ issue: item.issue, action: "acknowledged" }),
      });
      if (res.ok) setFeedbackMsg(`Acknowledged: ${item.issue}`);
    } catch (err) {
      console.error(err);
    }
  }

  // ===================== JSX RETURN =====================
  return (
    <div className="main-wrapper">
      {/* ===================== HEADER ===================== */}
      <header className="header">
        <div className="header-logo">
          <BarChart3 className="icon-large" />
          <h1 className="header-title">Crew AI KPI Dashboard</h1>
        </div>
        {feedbackMsg && <div className="feedback-msg">{feedbackMsg}</div>}
      </header>

      {/* ===================== UPLOAD SECTION ===================== */}
      <section className="card upload-card">
        <div className="form-group">
          <label>Business Goal</label>
          <input value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="e.g., Increase sales" />
        </div>
        <div className="form-group">
          <label>CSV File</label>
          <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
        </div>
        <button onClick={submit} disabled={loading} className="btn-primary">
          {loading ? "Running agents… please wait." : "Run Crew AI Agents"}
        </button>
      </section>
    {/* ===================== DATASET PREVIEW ===================== */}
{result?.preview && result.preview.length > 0 && (
  <div className="card">
    <h2>Dataset Preview (First 5 Rows)</h2>

    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            {Object.keys(result.preview[0]).map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {result.preview.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {Object.values(row).map((val, colIdx) => (
                <td key={colIdx}>
                  {val !== null && val !== undefined ? val.toString() : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
)}

      {/* ===================== RESULTS SECTION ===================== */}
      {result && (
        <section className="results">
          <>
  {/* ===================== DEPENDENCY GRAPH ===================== */}
  <div className="card">
    <h2>Dependency Graph</h2>
    <canvas
      ref={canvasRef}
      width={900}
      height={600}
      style={{
        width: "100%",
        border: "1px solid #e5e7eb",
        borderRadius: "8px",
        background: "white",
        marginTop: "10px",
      }}
    ></canvas>
  </div>
{/* Computed KPIs */}
<h3 className="section-title">Computed KPIs</h3>

{result?.metadata?.computed_kpis &&
 result.metadata.computed_kpis.length > 0 ? (
  <div className="kpi-list">
    {result.metadata.computed_kpis.map((kpi, index) => (
      <div key={index} className="kpi-card">
        <h4>{kpi.name}</h4>

        {kpi.value !== undefined && (
          <p className="kpi-value">
            <strong>Value:</strong> {kpi.value}
          </p>
        )}

        {kpi.formula && (
          <p className="kpi-formula">
            <strong>Formula:</strong> {kpi.formula}
          </p>
        )}

        {kpi.description && (
          <p className="kpi-desc">{kpi.description}</p>
        )}
      </div>
    ))}
  </div>
) : (
  <p>No computed KPIs returned.</p>
)}

  {/* ===================== KPI EVALUATION METRICS ===================== */}
  {result?.metadata?.evaluation && (
    <div className="card chart-card">
      <h2>KPI Evaluation Metrics</h2>

      <div className="metrics-grid">
        {/* KG-SAS */}
        <div className="metric-box">
          <h3>KG-SAS (Semantic Alignment)</h3>
          <p className="metric-value">
            {result?.metadata?.evaluation?.KG_SAS !== undefined
              ? result.metadata.evaluation.KG_SAS.toFixed(3)
              : "-"}
          </p>
          <p className="metric-desc">How well KPIs align with the business goal.</p>
        </div>

        {/* KFFR */}
        <div className="metric-box">
          <h3>KFFR (Formula Feasibility)</h3>
          <p className="metric-value">
            {result?.metadata?.evaluation?.KFFR !== undefined
              ? (result.metadata.evaluation.KFFR * 100).toFixed(1) + "%"
              : "-"}
          </p>
          <p className="metric-desc">Executability of KPI formulas.</p>
        </div>

        {/* DRI */}
        <div className="metric-box">
          <h3>DRI (Dataset Relevance)</h3>
          <p className="metric-value">
            {result?.metadata?.evaluation?.DRI !== undefined
              ? result.metadata.evaluation.DRI.toFixed(3)
              : "-"}
          </p>
          <p className="metric-desc">Are KPIs using relevant dataset dimensions?</p>
        </div>

        {/* MADD */}
        <div className="metric-box">
          <h3>MADD (Multi-Agent Drift)</h3>
          <p className="metric-value">
            {result?.metadata?.evaluation?.MADD?.madd_score !== undefined
              ? result.metadata.evaluation.MADD.madd_score.toFixed(3)
              : "-"}
          </p>
          <p className="metric-desc">Higher = more semantic drift across agents.</p>
        </div>

        {/* KSDI */}
        <div className="metric-box">
          <h3>KSDI (Structural Diversity)</h3>
          <p className="metric-value">
            {result?.metadata?.evaluation?.KSDI !== undefined
              ? result.metadata.evaluation.KSDI.toFixed(3)
              : "-"}
          </p>
          <p className="metric-desc">Variety of KPI analytic structures.</p>
        </div>
      </div>

      {/* Failure Breakdown */}
      <h3 style={{ marginTop: "20px" }}>KFFR Failure Breakdown</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={
            result?.metadata?.evaluation?.KFFR_failures
              ? Object.entries(result.metadata.evaluation.KFFR_failures).map(([k, v]) => ({
                  type: k,
                  value: v,
                }))
              : []
          }
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="type" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#ef4444" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )}

  {/* ===================== CSV PREVIEW ===================== */}
  {result?.preview && (
    <div className="card">
      <h2>CSV Preview</h2>
      <table className="csv-table">
        <thead>
          <tr>
            {Object.keys(result.preview[0] || {}).map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.preview.map((row, idx) => (
            <tr key={idx}>
              {Object.values(row).map((value, i) => (
                <td key={i}>{String(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )}

{/* ===================== KPI RECOMMENDATIONS ===================== */}
{result.recommendations && result.recommendations.length > 0 && (
  <div className="card table-card">
    <h2>Recommended KPIs</h2>

    <ul className="recommendations-list">
      {result.recommendations.map((kpi, idx) => (
        <li key={idx} className="recommendation-item">

          <strong>{kpi.name ?? `KPI ${idx + 1}`}</strong>

          {kpi.reasoning && (
            <p className="reasoning">{kpi.reasoning}</p>
          )}

          <p><b>Formula:</b> {kpi.formula ?? "N/A"}</p>
          <p><b>Type:</b> {kpi.type ?? "N/A"}</p>

          <div className="kpi-feedback">
            <button
              onClick={() => sendKpiFeedback(kpi.name, "up")}
              className="thumb-btn"
            >
              <ThumbsUp size={18} /> Good
            </button>

            <button
              onClick={() => sendKpiFeedback(kpi.name, "down")}
              className="thumb-btn red"
            >
              <ThumbsDown size={18} /> Bad
            </button>
          </div>

        </li>
      ))}
    </ul>
  </div>
)}


  {/* ===================== INSIGHTS SECTION ===================== */}
  {/* ===================== INSIGHTS ===================== */}
{result.insights && result.insights.length > 0 && (
  <div className="card insights-card">
    <h2>Insights</h2>

    {result.insights.map((ins, idx) => (
      <div key={idx} className={`insight-item ${ins.severity}`}>
        <h3>{ins.title}</h3>
        <p>{ins.description}</p>
        <p><b>Recommendation:</b> {ins.recommendation}</p>
      </div>
    ))}
  </div>
)}

  {/* ===================== DATA QUALITY SECTION ===================== */}
  {quality && (
    <div className="card quality-card">
      <h2>Data Quality Summary</h2>

      <p><b>Missing Values:</b> {quality.missing_values}</p>
      <p><b>Duplicate Rows:</b> {quality.duplicates}</p>
      <p><b>Outliers Detected:</b> {quality.outliers}</p>
    </div>
  )}
</>

        </section>
      )}
    </div>
  );
}
