import React, { useState } from "react";
import "./App.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("Increase sales");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [collapsedSteps, setCollapsedSteps] = useState({});

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
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/orchestrate", {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Fix: use data directly (endpoint does not return "output")
      setResult(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Crew AI KPI Dashboard</h1>

      {/* Form */}
      <form onSubmit={submit} className="form">
        <div className="form-group">
          <label>Business Goal</label>
          <input
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g., Increase sales, Reduce churn"
          />
        </div>

        <div className="form-group">
          <label>CSV File</label>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>

        <button type="submit">Run Crew AI Agents</button>
      </form>

      {/* Loading */}
      {loading && <p className="loading">Running agents… please wait.</p>}

      {/* Results */}
      {result && (
        <div className="results">

          {/* CSV Preview */}
          {result.profile?.sample_rows && (
            <div className="card">
              <h2>CSV Preview (first 5 rows)</h2>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      {Object.keys(result.profile.sample_rows[0]).map((col, idx) => (
                        <th key={idx}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.profile.sample_rows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {Object.values(row).map((val, cIdx) => (
                          <td key={cIdx}>{val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* KPI Recommendations */}
          {result.recommendations && (
            <div className="card">
              <h2>KPI Recommendations</h2>
              <table>
                <thead>
                  <tr><th>Name</th><th>Description</th><th>Formula</th></tr>
                </thead>
                <tbody>
                  {result.recommendations.map((r, idx) => (
                    <tr key={idx}>
                      <td>{r.name}</td>
                      <td>{r.description}</td>
                      <td><pre>{r.formula || "-"}</pre></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Computed KPIs */}
          {result.computed && (
            <div className="card">
              <h2>Computed KPIs</h2>
              <table>
                <thead>
                  <tr><th>KPI</th><th>Value</th></tr>
                </thead>
                <tbody>
                  {Object.entries(result.computed).map(([k, v], idx) => (
                    <tr key={idx}>
                      <td>{k}</td>
                      <td style={{ color: v === null ? "red" : "black" }}>
                        {v === null ? "Error" : v}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Insights */}
          {result.insights && (
            <div className="card">
              <h2>Insights</h2>
              <ul>
                {result.insights.map((i, idx) => (
                  <li key={idx}>
                    <strong>{i.title}:</strong> {i.description}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Agent Trace */}
          {result.trace && (
            <div className="card">
              <h2>Agent Discussion / Trace</h2>
              {result.trace.map((step, idx) => (
                <div key={idx} className="trace-step">
                  <div className="trace-header" onClick={() => toggleStep(idx)}>
                    <strong>{step.agent}</strong> {collapsedSteps[idx] ? "[+]" : "[-]"}
                  </div>
                  {!collapsedSteps[idx] && (
                    <div className="trace-content">
                      <div>
                        <strong>Input:</strong>
                        <pre>{JSON.stringify(step.input, null, 2)}</pre>
                      </div>
                      <div>
                        <strong>Output:</strong>
                        <pre>{JSON.stringify(step.output, null, 2)}</pre>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

        </div>
      )}
    </div>
  );
}
