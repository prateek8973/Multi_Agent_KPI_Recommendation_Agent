# backend/app.py - Enhanced Version with TraceStep Fix
import json
import pandas as pd
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KPI Recommender System",
    description="Multi-agent AI system for intelligent KPI recommendations with dependency analysis",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=3)

# ============================================================================

# RESPONSE MODELS
class KPIRecommendation(BaseModel):
    name: str
    description: str
    formula: str
    type: str
    unit: Optional[str] = None

class Insight(BaseModel):
    title: str
    description: str
    recommendation: Optional[str] = None
    severity: Optional[str] = "info"

class DataProfile(BaseModel):
    n_rows: int
    n_columns: int
    columns: List[str]
    numeric_columns: List[str]
    sample_rows: List[Dict[str, Any]]

class GoalAnalysis(BaseModel):
    goal_text: str
    objective: str
    focus_areas: List[str]

class DataQuality(BaseModel):
    missing_values: int
    outliers: int
    schema_issues: int
    quality_score: float

class ImpactItem(BaseModel):
    issue: str
    severity: str
    impacts: List[str]

class DependencyNode(BaseModel):
    id: str
    label: str
    type: str  # 'column' or 'kpi'
    kpi_type: Optional[str] = None

class DependencyEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    label: str
    is_derived: bool = False

class DependencyGraph(BaseModel):
    nodes: List[DependencyNode]
    edges: List[DependencyEdge]

# Fix: TraceStep.output can now accept dict or list of dicts
class TraceStep(BaseModel):
    agent: str
    input: Dict[str, Any]
    output: Union[Dict[str, Any], List[Dict[str, Any]]]

class OrchestrateResponse(BaseModel):
    profile: DataProfile
    goal: GoalAnalysis
    recommendations: List[KPIRecommendation]
    computed: Dict[str, float]
    insights: List[Insight]
    dependency_graph: Optional[DependencyGraph] = None
    data_quality: Optional[DataQuality] = None
    impact_map: Optional[List[ImpactItem]] = None
    trace: Optional[List[TraceStep]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FeedbackRequest(BaseModel):
    kpi_name: Optional[str] = None
    action: str
    issue: Optional[str] = None

# ============================================================================

# DEPENDENCY GRAPH GENERATOR
class DependencyGraphBuilder:
    @staticmethod
    def extract_column_dependencies(formula: str, columns: List[str]) -> List[str]:
        dependencies = []
        for col in columns:
            patterns = [f"'{col}'", f'"{col}"', f"get('{col}'", f'get("{col}"']
            if any(pattern in formula for pattern in patterns):
                dependencies.append(col)
        return dependencies

    @staticmethod
    def detect_kpi_dependencies(kpis: List[KPIRecommendation]) -> Dict[str, List[str]]:
        kpi_deps = {}
        for i, kpi in enumerate(kpis):
            deps = []
            for j, other_kpi in enumerate(kpis):
                if i != j and other_kpi.name.lower() in kpi.formula.lower():
                    deps.append(other_kpi.name)
            if deps:
                kpi_deps[kpi.name] = deps
        return kpi_deps

    @staticmethod
    def build_graph(kpis: List[KPIRecommendation], columns: List[str]) -> DependencyGraph:
        nodes = [DependencyNode(id=f"col_{col}", label=col, type="column") for col in columns]
        edges = []

        for idx, kpi in enumerate(kpis):
            kpi_id = f"kpi_{idx}"
            nodes.append(DependencyNode(id=kpi_id, label=kpi.name, type="kpi", kpi_type=kpi.type))
            for col in DependencyGraphBuilder.extract_column_dependencies(kpi.formula, columns):
                edges.append(DependencyEdge(**{"from": f"col_{col}", "to": kpi_id, "label": "uses", "is_derived": False}))

        kpi_deps = DependencyGraphBuilder.detect_kpi_dependencies(kpis)
        for idx, kpi in enumerate(kpis):
            if kpi.name in kpi_deps:
                for dep_name in kpi_deps[kpi.name]:
                    dep_idx = next((i for i, k in enumerate(kpis) if k.name == dep_name), None)
                    if dep_idx is not None:
                        edges.append(DependencyEdge(**{"from": f"kpi_{dep_idx}", "to": f"kpi_{idx}", "label": "derives", "is_derived": True}))
        return DependencyGraph(nodes=nodes, edges=edges)

# ============================================================================

# DATA QUALITY ANALYZER
class DataQualityAnalyzer:
    @staticmethod
    def analyze(df: pd.DataFrame) -> DataQuality:
        missing_count = df.isnull().sum().sum()
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outlier_count += ((df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))).sum()
        schema_issues = sum(
            0 < pd.to_numeric(df[col], errors='coerce').notna().sum() < len(df) 
            for col in df.columns if df[col].dtype == 'object'
        )
        total_cells = len(df) * len(df.columns)
        missing_ratio = missing_count / total_cells if total_cells > 0 else 0
        outlier_ratio = outlier_count / (len(df) * len(numeric_cols)) if len(numeric_cols) > 0 else 0
        schema_ratio = schema_issues / len(df.columns) if len(df.columns) > 0 else 0
        quality_score = max(0, 100 - (missing_ratio*40 + outlier_ratio*40 + schema_ratio*20))
        return DataQuality(
            missing_values=int(missing_count),
            outliers=int(outlier_count),
            schema_issues=schema_issues,
            quality_score=round(quality_score, 2)
        )

    @staticmethod
    def generate_impact_map(quality: DataQuality, kpis: List[KPIRecommendation], df: pd.DataFrame) -> List[ImpactItem]:
        impacts = []
        if quality.missing_values > 0:
            affected_kpis = []
            missing_cols = df.columns[df.isnull().any()].tolist()
            for kpi in kpis:
                if any(col in kpi.formula for col in missing_cols):
                    affected_kpis.append(kpi.name)
            if affected_kpis:
                severity = "high" if quality.missing_values > len(df)*0.1 else "medium"
                impacts.append(ImpactItem(issue=f"{quality.missing_values} missing values detected", severity=severity, impacts=affected_kpis))
        if quality.outliers > 0:
            affected_kpis = []
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for kpi in kpis:
                if any(kpi_type in kpi.type.lower() for kpi_type in ['sum','average','mean']):
                    if any(col in kpi.formula for col in numeric_cols):
                        affected_kpis.append(kpi.name)
            if affected_kpis:
                impacts.append(ImpactItem(issue=f"{quality.outliers} outliers may skew calculations", severity="medium", impacts=affected_kpis))
        if quality.schema_issues > 0:
            impacts.append(ImpactItem(issue=f"{quality.schema_issues} columns have mixed data types", severity="high", impacts=[kpi.name for kpi in kpis if kpi.type in ['sum','average']]))
        return impacts

# ============================================================================

# SAFE KPI EXECUTOR
class SafeKPIExecutor:
    ALLOWED_BUILTINS = {'sum': sum, 'len': len, 'float': float, 'int': int, 'min': min, 'max': max, 'abs': abs, 'round': round}

    @staticmethod
    def execute(formula: str, data: List[Dict[str, Any]]) -> float:
        try:
            namespace = {'__builtins__': {}, 'data': data, **SafeKPIExecutor.ALLOWED_BUILTINS}
            result = eval(formula, namespace, {})
            return float(result) if result is not None else 0.0
        except Exception as e:
            logger.warning(f"Error executing formula '{formula}': {e}")
            return 0.0

# ============================================================================

# CREW INTEGRATION
def run_crew_sync(csv_bytes: bytes, goal_text: str) -> dict:
    try:
        from crew import build_kpi_crew, parse_csv_bytes, prepare_crew_inputs
        data, schema = parse_csv_bytes(csv_bytes)
        if not data:
            raise ValueError("Uploaded CSV is empty.")
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("CSV has no data rows.")
        profile = {"n_rows": len(df), "n_columns": len(df.columns), "columns": list(df.columns),
                   "numeric_columns": df.select_dtypes(include="number").columns.tolist(), "sample_rows": data[:5]}
        crew = build_kpi_crew()
        crew_inputs = prepare_crew_inputs(data, schema, profile, goal_text)
        for task in crew.tasks:
            task.description = task.description.format(**crew_inputs)
        result = crew.kickoff(inputs={})
        return {"success": True, "result": result, "data": data, "schema": schema, "df": df}
    except Exception as e:
        logger.error(f"Crew execution failed: {e}")
        return {"success": False, "error": str(e)}

def extract_json_from_text(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
    return None

def parse_crew_results(result: Any, data: list, schema: list, goal_text: str) -> Dict[str, Any]:
    tasks_output = getattr(result, 'tasks_output', [])
    parsed = {
        "profile": {"n_rows": len(data), "n_columns": len(schema), "columns": schema, "numeric_columns": [], "sample_rows": data[:5] if data else []},
        "goal": {"goal_text": goal_text, "objective": "Analyze business metrics", "focus_areas": []},
        "recommendations": [],
        "computed": {},
        "insights": [],
        "trace": []
    }
    for i, task_output in enumerate(tasks_output):
        output_text = str(getattr(task_output, 'raw', task_output))
        logger.info(f"Task {i} output preview: {output_text[:200]}")
        output_json = extract_json_from_text(output_text)
        # Normalize list outputs into dict for trace
        if isinstance(output_json, list):
            trace_output = output_json
        else:
            trace_output = output_json or {"raw": output_text[:500]}
        parsed["trace"].append({"agent": f"Agent_{i}", "input": {"task_id": i}, "output": trace_output})
        if not output_json:
            continue
        if i == 0 and isinstance(output_json, dict):
            parsed["profile"].update(output_json)
        elif i == 1 and isinstance(output_json, dict):
            parsed["goal"] = {**parsed["goal"], **output_json}
        elif i == 2:
            if isinstance(output_json, list):
                parsed["recommendations"] = output_json
            elif isinstance(output_json, dict):
                parsed["recommendations"] = output_json.get("kpis", output_json.get("recommendations", []))
        elif i == 3:
            if isinstance(output_json, dict):
                parsed["computed"] = {k: float(v) for k, v in output_json.items() if v is not None}
        elif i == 4:
            if isinstance(output_json, list):
                parsed["insights"] = output_json
            elif isinstance(output_json, dict):
                parsed["insights"] = output_json.get("insights", [output_json])
    return parsed

# ============================================================================
# --- Add these imports near the top of app.py ---
from collections import Counter
import math
import re

# Try importing sklearn for TF-IDF embeddings and cosine similarity. If not available, fallback to simple token counts.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --- Utility embedding + similarity functions (pluggable) ---
def get_embeddings(texts):
    """
    Returns dense vectors for a list of texts.
    Uses sklearn.TfidfVectorizer when available as a lightweight fallback.
    Replace this function to call a proper embedding API (OpenAI, Vertex, etc.) if desired.
    """
    if not texts:
        return []
    if SKLEARN_AVAILABLE:
        vec = TfidfVectorizer(stop_words='english', max_features=2048)
        X = vec.fit_transform(texts).toarray()
        return X
    # Fallback simple token-count vectors (very crude, but usable)
    # Build vocabulary
    vocab = {}
    docs_tokens = []
    for t in texts:
        toks = re.findall(r"\w+", t.lower())
        docs_tokens.append(toks)
        for tok in toks:
            vocab.setdefault(tok, len(vocab))
    vectors = []
    for toks in docs_tokens:
        v = [0.0] * len(vocab)
        for tok in toks:
            v[vocab[tok]] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        v = [x / norm for x in v]
        vectors.append(v)
    return vectors

def cosine_sim_from_vectors(a, b):
    """Cosine similarity between 1D vectors (lists or numpy arrays)."""
    if not a or not b:
        return 0.0
    # NumPy not required; implement dot/L2
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x*x for x in b)) or 1.0
    return float(dot / (norm_a * norm_b))

# If sklearn available, prefer using its pairwise cosine for speed/robustness
def pairwise_cosine_matrix(vectors):
    if SKLEARN_AVAILABLE:
        return cosine_similarity(vectors)
    # fallback compute NxN
    N = len(vectors)
    M = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            M[i][j] = cosine_sim_from_vectors(vectors[i], vectors[j])
    return M

# --- Formula column extractor ---
COLUMN_REF_REGEX = re.compile(r"""row\.get\(\s*['"]([^'"]+)['"]""")

def extract_columns_from_formula(formula: str) -> List[str]:
    """Return list of column names explicitly referenced in formula via row.get('col'...)."""
    if not formula:
        return []
    return list({m.group(1) for m in COLUMN_REF_REGEX.finditer(formula)})

# --- detect KPI type (KSDI helper) ---
def detect_kpi_type(formula: str, name: str) -> str:
    f = (formula or "").lower()
    n = (name or "").lower()
    # Heuristics
    if "count(" in f or "len(" in f or "count" in n:
        return "COUNT"
    if "sum(" in f or "total" in n or re.search(r"\bsum\b", f):
        return "SUM"
    if any(tok in f for tok in ["mean(", "median(", "avg(", "average"]):
        return "AVG"
    if "/" in f and not re.search(r"\bcount\b", f):
        return "RATIO"
    if any(tok in f for tok in ["rolling", "shift(", "lag(", "diff("]):
        return "TEMPORAL"
    # Derived if it references other KPIs' names (we'll treat as DERIVED in the caller)
    if "*" in f or "+" in f or "-" in f:
        return "DERIVED"
    return "OTHER"

# --- Safe execution helper using existing SafeKPIExecutor ---
def try_execute_formula(formula: str, data_list: List[Dict[str,Any]]):
    """
    Wrap SafeKPIExecutor and surface typed errors for classification.
    Returns (ok: bool, value/None, failure_reason/None)
    """
    # Quick column presence check
    cols = extract_columns_from_formula(formula)
    # If columns referenced not present, return missing columns
    # The caller passes DataFrame 'df' to verify columns exist; but we'll check only presence in row keys if any
    # Execution attempt
    try:
        value = SafeKPIExecutor.execute(formula, data_list)
        # Sometimes executor returns 0.0 for errors silently; attempt to detect divide-by-zero by inspecting formula text
        return True, value, None
    except ZeroDivisionError:
        return False, None, "DivideByZero"
    except Exception as e:
        # Classify some common error classes by message (best-effort)
        msg = str(e).lower()
        if "keyerror" in msg or "name" in msg or "column" in msg:
            return False, None, "MissingColumns"
        if "could not convert" in msg or "valueerror" in msg or "typeerror" in msg:
            return False, None, "TypeMismatch"
        return False, None, "UnknownError"

# --- Full metrics computation function ---
def compute_evaluation_metrics(goal_text: str, data_profile: dict, kpi_list: List[Dict[str,Any]], df: pd.DataFrame, data_list: List[Dict[str,Any]], alpha: float = 0.4):
    """
    Returns a dict with all metrics:
      - kgsas
      - kffr (ratio + failure list)
      - dri (value + columns used + goal_relevant_columns)
      - madd (data->goal drift, goal->kpi drift, final)
      - ksdi (value + types_detected)
    """
    # --- KG-SAS: embed goal + KPI descriptions ---
    kpi_texts = [ (k.get("description") or k.get("name") or "") for k in kpi_list ]
    all_texts = [goal_text] + kpi_texts
    vecs = get_embeddings(all_texts)
    if len(vecs) < 1:
        kgsas_value = 0.0
    else:
        g_vec = vecs[0]
        k_vecs = vecs[1:]
        sims = []
        for v in k_vecs:
            # If sklearn used, can compute with pairwise; but we use pairwise_cosine_matrix when needed
            sims.append(cosine_sim_from_vectors(g_vec, v))
        kgsas_value = float(sum(sims)/len(sims)) if sims else 0.0

    # --- KFFR: attempt to execute formulas and categorize failures ---
    total = max(1, len(kpi_list))
    ok_count = 0
    failures = []
    for k in kpi_list:
        formula = k.get("formula", "")
        # Simple pre-check for missing columns using formula references vs df columns
        referenced_cols = extract_columns_from_formula(formula)
        missing_cols = [c for c in referenced_cols if c not in list(df.columns)]
        if missing_cols:
            failures.append({"kpi": k.get("name"), "reason": "MissingColumns", "missing_columns": missing_cols})
            continue
        ok, value, reason = try_execute_formula(formula, data_list)
        if ok:
            ok_count += 1
        else:
            failures.append({"kpi": k.get("name"), "reason": reason})
    kffr_ratio = round(ok_count / total, 3)

    # --- DRI: distinct columns used vs relevant_dimensions (from data_profile or business agent) ---
    used_cols = set()
    for k in kpi_list:
        used_cols.update(extract_columns_from_formula(k.get("formula", "")))
    # relevant columns: try to get from data_profile['relevant_dimensions'] else fallback to profile columns if present
    goal_relevant = data_profile.get("relevant_dimensions") or data_profile.get("columns") or []
    # If goal_relevant is JSON string (crew.prepare_crew_inputs uses json.dumps), attempt to load
    if isinstance(goal_relevant, str):
        try:
            import json as _json
            parsed = _json.loads(goal_relevant)
            goal_relevant = parsed if isinstance(parsed, list) else goal_relevant
        except Exception:
            # leave as-is
            pass
    goal_relevant = list(goal_relevant) if goal_relevant else []
    dri_value = round((len(used_cols) / max(1, len(goal_relevant))) if goal_relevant else (len(used_cols) / max(1, len(df.columns))), 3)

    # --- MADD: embeddings of data_profile summary, goal, and concatenated KPI texts ---
    # Build data_profile_text from data_profile dictionary (concise)
    data_profile_text = " ".join([
        f"columns: {','.join(data_profile.get('columns', [])[:10])}",
        f"n_rows: {data_profile.get('n_rows', 0)}",
        f"numeric_columns: {','.join(data_profile.get('numeric_columns', [])[:10])}"
    ])
    madd_texts = [data_profile_text, goal_text, " ".join(kpi_texts)]
    madd_vecs = get_embeddings(madd_texts)
    if len(madd_vecs) == 3:
        E_d, E_g, E_k = madd_vecs
        drift_dg = round(1.0 - cosine_sim_from_vectors(E_d, E_g), 3)
        drift_gk = round(1.0 - cosine_sim_from_vectors(E_g, E_k), 3)
        madd_final = round(alpha * drift_dg + (1.0 - alpha) * drift_gk, 3)
    else:
        drift_dg = drift_gk = madd_final = 0.0

    # --- KSDI: detect KPI types and compute diversity ---
    POSSIBLE_TYPES = ["SUM","AVG","RATIO","COUNT","TEMPORAL","DERIVED","RISK","OTHER"]
    types_found = set()
    for k in kpi_list:
        t = detect_kpi_type(k.get("formula",""), k.get("name",""))
        # treat formulas referencing KPI names as DERIVED
        # (also check if any kpi name appears in another's formula)
        if any(other.get("name","").lower() in (k.get("formula") or "").lower() for other in kpi_list if other is not k):
            t = "DERIVED"
        types_found.add(t)
    types_detected = sorted(list(types_found))
    ksdi_value = round(len(types_found) / len(POSSIBLE_TYPES), 3)

    # --- Prepare metric structure ---
    metrics = {
    "KG_SAS": round(kgsas_value, 3),
    "KFFR": round(kffr_ratio, 3),
    "KFFR_failures": failures,
    "DRI": round(dri_value, 3),
    "MADD": {"madd_score": madd_final},
    "KSDI": round(ksdi_value, 3)
}

    return metrics

# MAIN ENDPOINT
@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate(goal_text: str = Form(...), file: UploadFile = File(...)):
    start_time = datetime.now()
    try:
        csv_bytes = await file.read()
        if not csv_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        loop = asyncio.get_event_loop()
        crew_result = await loop.run_in_executor(executor, run_crew_sync, csv_bytes, goal_text)
        if not crew_result.get('success'):
            raise HTTPException(status_code=500, detail=crew_result.get('error', 'Unknown error'))

        parsed = parse_crew_results(crew_result['result'], crew_result['data'], crew_result['schema'], goal_text)

        # Fallback KPIs if agent failed to produce any
        if not parsed['recommendations']:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:3]
            parsed['recommendations'] = [
                {
                    "name": f"Total {col}",
                    "description": f"Sum of all {col} values",
                    "formula": f"sum(float(row.get('{col}',0)) for row in data)",
                    "type": "sum",
                    "unit": "numeric"
                } for col in numeric_cols
            ]

        data_list = crew_result['data']

        # Compute KPIs using safe executor
        computed = {}
        for kpi in parsed['recommendations']:
            try:
                value = SafeKPIExecutor.execute(kpi.get('formula', '0'), data_list)
                computed[kpi['name']] = round(value, 2)
            except Exception as e:
                logger.error(f"KPI computation error for {kpi.get('name')}: {e}")
                computed[kpi['name']] = 0.0

        # Create typed KPI objects for response and for dependency graph
        kpi_objects = [KPIRecommendation(**kpi) for kpi in parsed['recommendations']]

        dependency_graph = DependencyGraphBuilder.build_graph(kpi_objects, parsed['profile']['columns'])
        data_quality = DataQualityAnalyzer.analyze(df)
        impact_map = DataQualityAnalyzer.generate_impact_map(data_quality, kpi_objects, df)

        # compute evaluation metrics and attach into metadata
        try:
            # Pass the parsed['recommendations'] (list of dicts) as kpi_list
            evaluation = compute_evaluation_metrics(
                goal_text=goal_text,
                data_profile=parsed.get('goal') or parsed.get('profile') or {},
                kpi_list=parsed['recommendations'],
                df=df,
                data_list=data_list
            )
            metadata_evaluation = evaluation
        except Exception as e:
            logger.warning(f"Evaluation metric computation failed: {e}", exc_info=True)
            metadata_evaluation = {}

        # Build metadata (attach evaluation under metadata["evaluation"])
        metadata = {
            "processing_time_ms": int((datetime.now() - start_time).total_seconds()*1000),
            "ai_analysis_used": True,
            "kpi_count": len(kpi_objects),
            "file_name": file.filename,
            "evaluation": metadata_evaluation
        }

        # Build and return response (use the metadata dict created above)
        response = OrchestrateResponse(
            profile=DataProfile(**parsed['profile']),
            goal=GoalAnalysis(**parsed['goal']),
            recommendations=kpi_objects,
            computed=computed,
            insights=[Insight(**ins) for ins in parsed.get('insights', [])],
            dependency_graph=dependency_graph,
            data_quality=data_quality,
            impact_map=impact_map,
            trace=[TraceStep(**t) for t in parsed.get('trace', [])],
            metadata=metadata
        )
        return response

    except Exception as e:
        logger.error(f"Orchestration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================

# FEEDBACK ENDPOINTS
@app.post("/feedback/kpi")
async def kpi_feedback(request: FeedbackRequest):
    logger.info(f"KPI Feedback: {request.kpi_name} - {request.action}")
    return {"status": "success", "message": f"Feedback recorded for {request.kpi_name}"}

@app.post("/feedback/issue")
async def issue_feedback(request: FeedbackRequest):
    logger.info(f"Issue Feedback: {request.issue} - {request.action}")
    return {"status": "success", "message": f"Issue acknowledged: {request.issue}"}

# ============================================================================

# UTILITY ENDPOINTS
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "KPI Recommender System", "version": "2.5.0", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    return {
        "service": "KPI Recommender System",
        "version": "2.5.0",
        "description": "AI-powered KPI recommendations with dependency analysis",
        "features": [
            "Multi-agent KPI recommendation",
            "Dependency graph generation",
            "Data quality analysis",
            "Impact assessment",
            "Safe formula execution"
        ],
        "endpoints": {
            "orchestrate": "/orchestrate (POST)",
            "kpi_feedback": "/feedback/kpi (POST)",
            "issue_feedback": "/feedback/issue (POST)",
            "health": "/health (GET)"
        }
    }
