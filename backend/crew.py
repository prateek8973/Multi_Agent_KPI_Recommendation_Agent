# backend/crew.py
import csv
import json
from io import StringIO
from crewai import Crew, Task, Process
from crew_agents import DataAgent, BusinessGoalAgent, KPIRecommendationAgent, KPIComputationAgent, InsightAgent

def build_kpi_crew():
    """Build the KPI analysis crew with proper data injection."""

    # Task 1: Data profiling - Data embedded in description
    data_task = Task(
        description=(
            "Analyze the CSV dataset provided below.\n\n"
            "Dataset Details:\n"
            "- Rows: {n_rows}\n"
            "- Columns: {columns}\n\n"
            "Sample Data (first 5 rows):\n"
            "{sample_data}\n\n"
            "Statistical Summary:\n"
            "{stats_summary}\n\n"
            "Task: Create a comprehensive data profile as JSON with:\n"
            "- 'n_rows': row count\n"
            "- 'n_columns': column count  \n"
            "- 'columns': list of column names\n"
            "- 'data_types': dict mapping columns to their data types\n"
            "- 'numeric_columns': list of numeric column names\n"
            "- 'key_observations': list of important findings\n\n"
            "Output ONLY valid JSON, no markdown."
        ),
        expected_output="Valid JSON object with complete data profile",
        agent=DataAgent,
    )

    # Task 2: Goal interpretation
    goal_task = Task(
        description=(
            "Business Goal: {goal_text}\n\n"
            "Based on the data profile from the previous task and this business goal, "
            "identify:\n"
            "- Primary objective\n"
            "- Key focus areas\n"
            "- Relevant data dimensions\n"
            "- Suggested KPI categories\n\n"
            "Output JSON: {{'objective': str, 'focus_areas': [str], "
            "'relevant_dimensions': [str], 'kpi_categories': [str]}}\n\n"
            "Output ONLY valid JSON, no markdown."
        ),
        expected_output="JSON object with structured goal analysis",
        agent=BusinessGoalAgent,
        context=[data_task],
    )

    # Task 3: KPI generation
    kpi_task = Task(
        description=(
            "Available columns: {columns}\n"
            "Numeric columns: {numeric_columns}\n\n"
            "Generate 4-6 KPIs based on the data profile and business goal.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY use columns from the list above\n"
            "2. Formulas must be valid Python that works with a list of dicts\n"
            "3. Use this exact format: sum(float(row.get('column', 0)) for row in data)\n"
            "4. For ratios: sum(float(row.get('a', 0)) for row in data) / sum(float(row.get('b', 1)) for row in data)\n\n"
            "Output JSON array:\n"
            "[{{\n"
            "  \"name\": \"Total Revenue\",\n"
            "  \"description\": \"Sum of all revenue\",\n"
            "  \"formula\": \"sum(float(row.get('revenue', 0)) for row in data)\",\n"
            "  \"type\": \"sum\"\n"
            "}}]\n\n"
            "Output ONLY valid JSON array, no markdown."
        ),
        expected_output="JSON array of 4-6 KPI objects with valid formulas",
        agent=KPIRecommendationAgent,
        context=[data_task, goal_task],
    )

    # Task 4: KPI computation - Full dataset embedded
    compute_task = Task(
        description=(
            "Complete Dataset:\n"
            "{full_data}\n\n"
            "KPIs to compute (from previous task):\n"
            "{{kpis_from_previous_task}}\n\n"
            "For each KPI:\n"
            "1. Take the formula\n"
            "2. Replace 'data' with the dataset above\n"
            "3. Execute the formula\n"
            "4. Return the computed value\n\n"
            "Example:\n"
            "If formula is: sum(float(row.get('price', 0)) for row in data)\n"
            "And data is: [{{'price': 10}}, {{'price': 20}}]\n"
            "Result: 30.0\n\n"
            "Output JSON object mapping KPI names to values:\n"
            "{{\n"
            "  \"Total Revenue\": 150000.50,\n"
            "  \"Average Order Value\": 75.25\n"
            "}}\n\n"
            "Output ONLY valid JSON object, no markdown."
        ),
        expected_output="JSON object with computed KPI values",
        agent=KPIComputationAgent,
        context=[data_task, kpi_task],
    )

    # Task 5: Insight generation
    insight_task = Task(
        description=(
            "Computed KPIs:\n"
            "{{computed_kpis}}\n\n"
            "Business Goal:\n"
            "{goal_text}\n\n"
            "Generate 3-5 actionable insights analyzing these KPI values.\n"
            "Each insight should have:\n"
            "- title: brief headline\n"
            "- description: detailed analysis\n"
            "- recommendation: specific action to take\n\n"
            "Output JSON array:\n"
            "[{{\n"
            "  \"title\": \"Strong Revenue Performance\",\n"
            "  \"description\": \"Total revenue of $150K exceeds typical benchmarks...\",\n"
            "  \"recommendation\": \"Focus on customer retention to maintain growth\"\n"
            "}}]\n\n"
            "Output ONLY valid JSON array, no markdown."
        ),
        expected_output="JSON array of insight objects",
        agent=InsightAgent,
        context=[goal_task, compute_task],
    )

    crew = Crew(
        agents=[DataAgent, BusinessGoalAgent, KPIRecommendationAgent, 
                KPIComputationAgent, InsightAgent],
        tasks=[data_task, goal_task, kpi_task, compute_task, insight_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def parse_csv_bytes(csv_bytes: bytes):
    """Parse CSV bytes into list of dicts and column names."""
    content = csv_bytes.decode("utf-8")
    reader = csv.DictReader(StringIO(content))
    data = list(reader)
    schema = reader.fieldnames or []
    return data, schema


def prepare_crew_inputs(data: list, schema: list, profile: dict, goal_text: str) -> dict:
    """
    Prepare inputs by embedding actual data into task descriptions.
    This is the KEY function that solves the data passing problem.
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Format sample data nicely
    sample_data = json.dumps(data[:5], indent=2)
    
    # Format full data (limit if too large)
    if len(data) > 100:
        full_data_subset = data[:100]
        full_data = json.dumps(full_data_subset, indent=2)
        full_data += f"\n\n... (showing first 100 of {len(data)} rows)"
    else:
        full_data = json.dumps(data, indent=2)
    
    # Create statistical summary
    stats_summary = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats_summary[col] = {
                "mean": float(col_data.mean()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "count": int(col_data.count())
            }
    
    stats_str = json.dumps(stats_summary, indent=2)
    
    # Return formatted strings that will be injected into task descriptions
    return {
        "n_rows": len(data),
        "columns": json.dumps(schema),
        "numeric_columns": json.dumps(numeric_cols),
        "sample_data": sample_data,
        "full_data": full_data,
        "stats_summary": stats_str,
        "goal_text": goal_text,
    }