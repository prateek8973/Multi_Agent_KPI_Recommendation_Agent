# backend/app.py
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from crew import build_kpi_crew, parse_csv_bytes, prepare_crew_inputs
import re

app = FastAPI(title="Crew KPI Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=3)

# Response models
class KPIRecommendation(BaseModel):
    name: str
    description: str
    formula: str
    type: str

class Insight(BaseModel):
    title: str
    description: str

class OrchestrateResponse(BaseModel):
    profile: Dict[str, Any]
    goal: Dict[str, Any]
    recommendations: List[KPIRecommendation]
    computed: Dict[str, float]
    insights: List[Insight]


def extract_json_from_text(text: str):
    """Extract JSON from text that might have markdown or extra content."""
    # Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try again
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Find JSON in text
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None


def run_crew_sync(csv_bytes: bytes, goal_text: str) -> dict:
    """Run the crew with actual data embedded in task descriptions."""
    data, schema = parse_csv_bytes(csv_bytes)
    
    if not data:
        raise ValueError("Uploaded CSV is empty.")

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("CSV has no data rows.")

    # Create profile
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "numeric_columns": numeric_cols,
        "sample_rows": data[:5],
    }

    # Build crew
    crew = build_kpi_crew()
    
    # CRITICAL: Prepare inputs with data embedded in descriptions
    crew_inputs = prepare_crew_inputs(data, schema, profile, goal_text)
    
    # Format task descriptions with actual data
    # This replaces placeholders in task descriptions with real data
    for task in crew.tasks:
        task.description = task.description.format(**crew_inputs)
    
    # Kickoff with formatted tasks
    result = crew.kickoff(inputs={})  # Empty inputs, data is in descriptions
    
    return parse_crew_results(result, data, schema, goal_text)


def parse_crew_results(result: Any, data: list, schema: list, goal_text: str) -> Dict[str, Any]:
    """Parse crew results with better JSON extraction."""
    tasks_output = result.tasks_output if hasattr(result, 'tasks_output') else []

    parsed = {
        "profile": {
            "n_rows": len(data),
            "n_columns": len(schema),
            "columns": schema,
            "sample_rows": data[:5] if data else []
        },
        "goal": {
            "goal_text": goal_text,
            "objective": "Analyze business metrics",
            "focus_areas": []
        },
        "recommendations": [],
        "computed": {},
        "insights": []
    }

    for i, task_output in enumerate(tasks_output):
        output_text = str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output)
        
        print(f"\n=== Task {i} Raw Output ===")
        print(output_text[:500])
        print("=" * 50)
        
        output_json = extract_json_from_text(output_text)
        
        if not output_json:
            print(f"WARNING: Task {i} did not return valid JSON")
            continue

        # Task 0: Data profiling
        if i == 0 and isinstance(output_json, dict):
            parsed["profile"].update(output_json)

        # Task 1: Goal interpretation
        elif i == 1 and isinstance(output_json, dict):
            parsed["goal"] = {**parsed["goal"], **output_json}

        # Task 2: KPI recommendations
        elif i == 2:
            if isinstance(output_json, list):
                parsed["recommendations"] = output_json
            elif isinstance(output_json, dict):
                parsed["recommendations"] = output_json.get("kpis", output_json.get("recommendations", []))
            
            # Fallback if no KPIs generated
            if not parsed["recommendations"]:
                print("WARNING: No KPIs generated, using fallback")
                parsed["recommendations"] = generate_fallback_kpis(schema, data)

        # Task 3: KPI computation
        elif i == 3:
            computed_dict = {}
            
            if isinstance(output_json, dict):
                for k, v in output_json.items():
                    try:
                        computed_dict[str(k)] = float(v)
                    except (ValueError, TypeError) as e:
                        print(f"Could not convert {k}={v} to float: {e}")
            elif isinstance(output_json, list):
                for item in output_json:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            try:
                                computed_dict[str(k)] = float(v)
                            except (ValueError, TypeError):
                                pass
            
            parsed["computed"] = computed_dict
            
            # Fallback computation if agent failed
            if not computed_dict and parsed["recommendations"]:
                print("WARNING: KPI computation failed, using fallback")
                parsed["computed"] = compute_fallback_kpis(data, parsed["recommendations"])

        # Task 4: Insights
        elif i == 4:
            if isinstance(output_json, list):
                parsed["insights"] = output_json
            elif isinstance(output_json, dict):
                if "insights" in output_json:
                    parsed["insights"] = output_json["insights"]
                else:
                    parsed["insights"] = [output_json]

    return parsed


def generate_fallback_kpis(schema: list, data: list) -> list:
    """Generate basic KPIs when agent fails."""
    if not data:
        return []
    
    kpis = []
    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    for col in numeric_cols[:5]:  # Max 5 KPIs
        kpis.append({
            "name": f"Total {col}",
            "description": f"Sum of all {col} values",
            "formula": f"sum(float(row.get('{col}', 0)) for row in data)",
            "type": "sum"
        })
    
    return kpis


def compute_fallback_kpis(data: list, kpis: list) -> dict:
    """Compute KPIs when agent fails."""
    computed = {}
    
    for kpi in kpis:
        try:
            name = kpi["name"]
            formula = kpi["formula"]
            
            # Execute formula safely
            result = eval(formula, {"__builtins__": {}}, {"data": data, "sum": sum, "float": float, "row": None})
            computed[name] = float(result)
            
        except Exception as e:
            print(f"Error computing {kpi.get('name')}: {e}")
            computed[kpi.get('name', 'Unknown')] = 0.0
    
    return computed


@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(goal_text: str = Form(...), file: UploadFile = File(...)):
    """Main endpoint that processes CSV and returns KPI analysis."""
    try:
        csv_bytes = await file.read()
        
        if not csv_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Run crew in thread pool (CrewAI is synchronous)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_crew_sync, csv_bytes, goal_text)
        
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in orchestrate endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "KPI Recommender System"}


@app.get("/")
async def root():
    return {
        "service": "Multi-Agent KPI Recommender System",
        "version": "2.0.0",
        "endpoints": {
            "orchestrate": "/orchestrate (POST)",
            "health": "/health (GET)"
        }
    }