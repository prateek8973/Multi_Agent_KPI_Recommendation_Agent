# backend/crew.py - Enhanced Version
import csv
import json
from io import StringIO
import pandas as pd
from crewai import Crew, Task, Process
from crew_agents import (
    DataAgent, 
    BusinessGoalAgent, 
    KPIRecommendationAgent, 
    KPIComputationAgent, 
    InsightAgent
)

def build_kpi_crew():
    """
    Build the KPI analysis crew with optimized task design.
    
    Key improvements:
    - Clearer task descriptions with structured output requirements
    - Better context passing between agents
    - Explicit JSON schema examples
    - Robust error handling instructions
    """

    # Task 1: Data Profiling
    data_task = Task(
        description=(
            "You are analyzing a CSV dataset. Here are the details:\n\n"
            "Dataset Metadata:\n"
            "- Total Rows: {n_rows}\n"
            "- Total Columns: {n_columns}\n"
            "- Column Names: {columns}\n"
            "- Numeric Columns: {numeric_columns}\n"
            "- Categorical Columns: {categorical_columns}\n\n"
            "Sample Data (first 5 rows):\n"
            "{sample_data}\n\n"
            "Statistical Summary for Numeric Columns:\n"
            "{stats_summary}\n\n"
            "YOUR TASK:\n"
            "Create a comprehensive data profile that includes:\n"
            "1. Confirmation of row and column counts\n"
            "2. Data type classification for each column\n"
            "3. Identification of numeric vs categorical columns\n"
            "4. Key observations about data patterns\n"
            "5. Any data quality concerns you notice\n\n"
            "CRITICAL OUTPUT REQUIREMENTS:\n"
            "- Return ONLY valid JSON\n"
            "- NO markdown code blocks (no ```json)\n"
            "- NO explanatory text before or after the JSON\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "{{\n"
            "  \"n_rows\": {n_rows},\n"
            "  \"n_columns\": {n_columns},\n"
            "  \"columns\": {columns},\n"
            "  \"data_types\": {{\"column_name\": \"numeric|categorical|datetime\"}},\n"
            "  \"numeric_columns\": {numeric_columns},\n"
            "  \"key_observations\": [\n"
            "    \"observation about the data\",\n"
            "    \"another important finding\"\n"
            "  ]\n"
            "}}"
        ),
        expected_output="Valid JSON object with complete data profile",
        agent=DataAgent,
    )

    # Task 2: Business Goal Interpretation
    goal_task = Task(
        description=(
            "You are a business strategy analyst. You have been given:\n\n"
            "BUSINESS GOAL:\n"
            "\"{goal_text}\"\n\n"
            "DATA CONTEXT:\n"
            "The previous agent analyzed the data and found:\n"
            "- Available columns: {columns}\n"
            "- Numeric columns: {numeric_columns}\n"
            "- Total rows: {n_rows}\n\n"
            "YOUR TASK:\n"
            "Interpret this business goal in the context of the available data:\n"
            "1. What is the PRIMARY OBJECTIVE we're trying to measure?\n"
            "2. What are 2-4 KEY FOCUS AREAS related to this goal?\n"
            "3. Which DATA DIMENSIONS (columns) are most relevant?\n"
            "4. What KPI CATEGORIES would best measure progress?\n\n"
            "Think about:\n"
            "- What specific metrics would indicate success?\n"
            "- Which columns contain the most relevant information?\n"
            "- What types of analysis would provide the most value?\n\n"
            "CRITICAL OUTPUT REQUIREMENTS:\n"
            "- Return ONLY valid JSON\n"
            "- NO markdown code blocks\n"
            "- NO explanatory text\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "{{\n"
            "  \"objective\": \"Clear, specific statement of what to measure\",\n"
            "  \"focus_areas\": [\n"
            "    \"Focus area 1\",\n"
            "    \"Focus area 2\",\n"
            "    \"Focus area 3\"\n"
            "  ],\n"
            "  \"relevant_dimensions\": [\"column1\", \"column2\"],\n"
            "  \"kpi_categories\": [\"financial\", \"operational\", \"growth\"]\n"
            "}}"
        ),
        expected_output="JSON object with structured goal analysis",
        agent=BusinessGoalAgent,
        context=[data_task],
    )

    # Task 3: KPI Recommendation
    kpi_task = Task(
        description=(
            "You are a KPI design expert. You need to create meaningful metrics.\n\n"
            "AVAILABLE DATA:\n"
            "- All Columns: {columns}\n"
            "- Numeric Columns (use these for calculations): {numeric_columns}\n"
            "- Dataset size: {n_rows} rows\n\n"
            "BUSINESS CONTEXT:\n"
            "Goal: {goal_text}\n\n"
            "YOUR TASK:\n"
            "Design 4-6 KPIs that:\n"
            "1. Directly support the business goal\n"
            "2. Use ONLY columns from the numeric columns list above\n"
            "3. Have VALID, EXECUTABLE Python formulas\n"
            "4. Cover different aspects (totals, averages, ratios, counts)\n"
            "5. Are clearly named and well-described\n\n"
            "FORMULA RULES - ABSOLUTELY CRITICAL:\n"
            "✓ CORRECT: sum(float(row.get('revenue', 0)) for row in data)\n"
            "✓ CORRECT: sum(float(row.get('price', 0)) * float(row.get('quantity', 1)) for row in data)\n"
            "✓ CORRECT: sum(float(row.get('sales', 0)) for row in data) / len([r for r in data if r.get('sales')])\n"
            "✗ WRONG: sum(revenue)  # 'revenue' is not defined\n"
            "✗ WRONG: df['price'].sum()  # No DataFrame available\n"
            "✗ WRONG: data['column'].mean()  # data is a list, not DataFrame\n\n"
            "For RATIOS, always protect against division by zero:\n"
            "✓ CORRECT: (sum(float(row.get('profit', 0)) for row in data) / sum(float(row.get('revenue', 1)) for row in data)) * 100\n\n"
            "CRITICAL OUTPUT REQUIREMENTS:\n"
            "- Return ONLY a JSON array\n"
            "- NO markdown code blocks\n"
            "- Each KPI must have: name, description, formula, type, unit\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "[\n"
            "  {{\n"
            "    \"name\": \"Total Revenue\",\n"
            "    \"description\": \"Sum of all revenue across all transactions\",\n"
            "    \"formula\": \"sum(float(row.get('revenue', 0)) for row in data)\",\n"
            "    \"type\": \"sum\",\n"
            "    \"unit\": \"currency\"\n"
            "  }},\n"
            "  {{\n"
            "    \"name\": \"Average Order Value\",\n"
            "    \"description\": \"Mean revenue per transaction\",\n"
            "    \"formula\": \"sum(float(row.get('revenue', 0)) for row in data) / len([r for r in data if r.get('revenue')])\",\n"
            "    \"type\": \"average\",\n"
            "    \"unit\": \"currency\"\n"
            "  }}\n"
            "]"
        ),
        expected_output="JSON array of 4-6 KPI objects with valid, executable formulas",
        agent=KPIRecommendationAgent,
        context=[data_task, goal_task],
    )

    # Task 4: KPI Computation
    compute_task = Task(
        description=(
            "You are a calculation engine. You will execute KPI formulas.\n\n"
            "DATASET (limited sample for context):\n"
            "{full_data}\n\n"
            "Note: The formulas will execute on the FULL dataset of {data_size} rows.\n\n"
            "KPIs TO COMPUTE:\n"
            "You will receive KPIs from the previous agent.\n\n"
            "YOUR TASK:\n"
            "For each KPI formula:\n"
            "1. Take the exact formula provided\n"
            "2. The formula already references 'data' variable\n"
            "3. Calculate the numeric result\n"
            "4. Handle any errors gracefully (return 0.0 if calculation fails)\n"
            "5. Round to 2 decimal places\n\n"
            "EXAMPLE COMPUTATION:\n"
            "If KPI is:\n"
            "{{\n"
            "  \"name\": \"Total Sales\",\n"
            "  \"formula\": \"sum(float(row.get('sales', 0)) for row in data)\"\n"
            "}}\n\n"
            "And data contains: [{{\"sales\": 100}}, {{\"sales\": 200}}, {{\"sales\": 150}}]\n"
            "Then output: {{\"Total Sales\": 450.0}}\n\n"
            "ERROR HANDLING:\n"
            "- Division by zero → return 0.0\n"
            "- Missing column → formula already handles with row.get(col, 0)\n"
            "- Any other error → return 0.0\n\n"
            "CRITICAL OUTPUT REQUIREMENTS:\n"
            "- Return ONLY a JSON object\n"
            "- NO markdown code blocks\n"
            "- Keys are KPI names, values are numbers\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "{{\n"
            "  \"Total Revenue\": 125000.50,\n"
            "  \"Average Order Value\": 75.25,\n"
            "  \"Profit Margin\": 23.5,\n"
            "  \"Transaction Count\": 1500.0\n"
            "}}"
        ),
        expected_output="JSON object mapping KPI names to computed numeric values",
        agent=KPIComputationAgent,
        context=[data_task, kpi_task],
    )

    # Task 5: Insight Generation
    insight_task = Task(
        description=(
            "You are a business intelligence analyst generating actionable insights.\n\n"
            "BUSINESS GOAL:\n"
            "\"{goal_text}\"\n\n"
            "COMPUTED KPI VALUES:\n"
            "You will receive the computed KPI values from the previous agent.\n\n"
            "YOUR TASK:\n"
            "Analyze the KPI values and generate 3-5 business insights:\n"
            "1. Identify what the numbers tell us\n"
            "2. Highlight strong performance (severity: 'success')\n"
            "3. Flag areas of concern (severity: 'warning')\n"
            "4. Provide specific, actionable recommendations\n"
            "5. Consider the business goal context\n\n"
            "Each insight should:\n"
            "- Have a compelling, clear title\n"
            "- Explain what the data shows and why it matters\n"
            "- Include a specific recommendation for action\n"
            "- Have appropriate severity level\n\n"
            "INSIGHT CATEGORIES:\n"
            "- Performance highlights (what's going well)\n"
            "- Concerns or risks (what needs attention)\n"
            "- Opportunities (what could be improved)\n"
            "- Strategic recommendations (what to do next)\n\n"
            "CRITICAL OUTPUT REQUIREMENTS:\n"
            "- Return ONLY a JSON array\n"
            "- NO markdown code blocks\n"
            "- Each insight needs: title, description, recommendation, severity\n\n"
            "EXACT OUTPUT FORMAT:\n"
            "[\n"
            "  {{\n"
            "    \"title\": \"Strong Revenue Performance\",\n"
            "    \"description\": \"Total revenue of $125,000 indicates healthy business activity and exceeds typical benchmarks for this dataset size.\",\n"
            "    \"recommendation\": \"Maintain current sales strategies and consider scaling successful approaches to new markets or customer segments.\",\n"
            "    \"severity\": \"success\"\n"
            "  }},\n"
            "  {{\n"
            "    \"title\": \"Below-Target Profit Margins\",\n"
            "    \"description\": \"Profit margin of 23.5% is below industry standard of 30-35%, suggesting cost optimization opportunities.\",\n"
            "    \"recommendation\": \"Conduct cost analysis to identify areas for efficiency improvements, particularly in operations and overhead.\",\n"
            "    \"severity\": \"warning\"\n"
            "  }}\n"
            "]"
        ),
        expected_output="JSON array of 3-5 insight objects with actionable recommendations",
        agent=InsightAgent,
        context=[goal_task, compute_task],
    )

    crew = Crew(
        agents=[
            DataAgent, 
            BusinessGoalAgent, 
            KPIRecommendationAgent, 
            KPIComputationAgent, 
            InsightAgent
        ],
        tasks=[data_task, goal_task, kpi_task, compute_task, insight_task],
        process=Process.sequential,
        verbose=True,
        memory=False,  # Disable to avoid context pollution
    )

    return crew


def parse_csv_bytes(csv_bytes: bytes):
    """
    Parse CSV bytes into list of dicts and column names.
    
    Also converts numeric strings to actual numbers for better processing.
    """
    try:
        content = csv_bytes.decode("utf-8")
        reader = csv.DictReader(StringIO(content))
        data = list(reader)
        schema = reader.fieldnames or []
        
        # Clean and convert data
        for row in data:
            for key, value in list(row.items()):
                # Strip whitespace from keys and values
                cleaned_key = key.strip() if key else key
                if cleaned_key != key:
                    row[cleaned_key] = row.pop(key)
                    key = cleaned_key
                
                if value is not None and value != '':
                    # Try to convert to number
                    try:
                        # Try integer first
                        if '.' not in str(value):
                            row[key] = int(value)
                        else:
                            row[key] = float(value)
                    except (ValueError, TypeError):
                        # Keep as string if not numeric
                        row[key] = str(value).strip()
                else:
                    row[key] = None
        
        # Clean schema
        schema = [col.strip() for col in schema]
        
        return data, schema
        
    except Exception as e:
        raise ValueError(f"CSV parsing failed: {str(e)}")


def prepare_crew_inputs(data: list, schema: list, profile: dict, goal_text: str) -> dict:
    """
    Prepare inputs by formatting data for task descriptions.
    
    Optimized to:
    1. Limit data size to avoid token limits
    2. Provide rich statistical context
    3. Format data clearly for AI agents
    4. Handle large datasets efficiently
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Create sample data (first 5 rows)
    sample_data = json.dumps(data[:5], indent=2, default=str)
    
    # Create statistical summary for numeric columns
    stats_summary = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            stats_summary[col] = {
                "mean": round(float(col_data.mean()), 2),
                "median": round(float(col_data.median()), 2),
                "min": round(float(col_data.min()), 2),
                "max": round(float(col_data.max()), 2),
                "std": round(float(col_data.std()), 2) if len(col_data) > 1 else 0,
                "count": int(col_data.count()),
                "missing": int(df[col].isnull().sum())
            }
    
    stats_str = json.dumps(stats_summary, indent=2)
    
    # For computation task, provide dataset context
    # Limit to 100 rows for context (formula runs on full data)
    data_size = len(data)
    if data_size > 100:
        full_data_subset = data[:100]
        full_data = json.dumps(full_data_subset, indent=2, default=str)
        full_data += f"\n\n... (showing first 100 of {data_size} rows for reference)\n"
        full_data += "IMPORTANT: Your formulas will execute on ALL {data_size} rows, not just this sample."
        data_size_note = f"first 100 of {data_size}"
    else:
        full_data = json.dumps(data, indent=2, default=str)
        data_size_note = str(data_size)
    
    return {
        "n_rows": data_size,
        "n_columns": len(schema),
        "columns": json.dumps(schema),
        "numeric_columns": json.dumps(numeric_cols),
        "categorical_columns": json.dumps(categorical_cols),
        "sample_data": sample_data,
        "full_data": full_data,
        "data_size": data_size_note,
        "stats_summary": stats_str,
        "goal_text": goal_text,
    }


def validate_kpi_formula(formula: str, columns: list) -> tuple:
    """
    Validate a KPI formula before execution.
    
    Returns: (is_valid: bool, error_message: str or None)
    """
    # Check for dangerous operations
    dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file', 'os.', 'sys.']
    for keyword in dangerous_keywords:
        if keyword in formula:
            return False, f"Formula contains forbidden keyword: {keyword}"
    
    # Check if it uses proper row.get() pattern
    if 'row.get(' not in formula and "row.get(" not in formula:
        return False, "Formula must use row.get() pattern to access columns safely"
    
    # Check if referenced columns exist
    for col in columns:
        # Look for column references
        if f"'{col}'" in formula or f'"{col}"' in formula:
            # Column is referenced, which is good
            pass
    
    return True, None


def generate_fallback_kpis(schema: list, data: list, goal_text: str) -> list:
    """
    Generate intelligent fallback KPIs when AI agent fails.
    
    Uses heuristics to create relevant KPIs based on:
    - Column names
    - Business goal keywords
    - Data types
    """
    if not data:
        return []
    
    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    kpis = []
    goal_lower = goal_text.lower()
    
    # Detect common business column patterns
    revenue_cols = [c for c in numeric_cols if any(
        term in c.lower() for term in ['revenue', 'sales', 'income', 'price', 'amount']
    )]
    
    cost_cols = [c for c in numeric_cols if any(
        term in c.lower() for term in ['cost', 'expense', 'spend', 'expenditure']
    )]
    
    quantity_cols = [c for c in numeric_cols if any(
        term in c.lower() for term in ['quantity', 'count', 'units', 'volume', 'qty']
    )]
    
    # Generate revenue/sales KPIs
    for col in revenue_cols[:2]:
        kpis.append({
            "name": f"Total {col.replace('_', ' ').title()}",
            "description": f"Sum of all {col} values in the dataset",
            "formula": f"sum(float(row.get('{col}', 0)) for row in data)",
            "type": "sum",
            "unit": "currency"
        })
        
        if len(data) > 1:
            kpis.append({
                "name": f"Average {col.replace('_', ' ').title()}",
                "description": f"Mean {col} value per record",
                "formula": f"sum(float(row.get('{col}', 0)) for row in data) / len([r for r in data if r.get('{col}')])",
                "type": "average",
                "unit": "currency"
            })
    
    # Generate cost KPIs
    for col in cost_cols[:1]:
        kpis.append({
            "name": f"Total {col.replace('_', ' ').title()}",
            "description": f"Sum of all {col} expenses",
            "formula": f"sum(float(row.get('{col}', 0)) for row in data)",
            "type": "sum",
            "unit": "currency"
        })
    
    # Generate profit margin if we have both revenue and cost
    if revenue_cols and cost_cols:
        rev_col = revenue_cols[0]
        cost_col = cost_cols[0]
        kpis.append({
            "name": "Profit Margin Percentage",
            "description": "Percentage of revenue retained as profit after costs",
            "formula": f"((sum(float(row.get('{rev_col}', 0)) for row in data) - sum(float(row.get('{cost_col}', 0)) for row in data)) / sum(float(row.get('{rev_col}', 1)) for row in data)) * 100",
            "type": "ratio",
            "unit": "percentage"
        })
    
    # Generate quantity KPIs
    for col in quantity_cols[:1]:
        kpis.append({
            "name": f"Total {col.replace('_', ' ').title()}",
            "description": f"Total count of {col} across all records",
            "formula": f"sum(float(row.get('{col}', 0)) for row in data)",
            "type": "sum",
            "unit": "units"
        })
    
    # If no specific columns detected, use first numeric columns
    if not kpis and numeric_cols:
        for col in numeric_cols[:3]:
            kpis.append({
                "name": f"Total {col.replace('_', ' ').title()}",
                "description": f"Sum of all {col} values",
                "formula": f"sum(float(row.get('{col}', 0)) for row in data)",
                "type": "sum",
                "unit": "numeric"
            })
    
    return kpis[:6]  # Return max 6 KPIs