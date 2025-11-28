# backend/tools.py
from crewai_tools import CodeInterpreterTool, tool
from typing import Dict, List, Any
import json

# Code interpreter tool for executing KPI formulas safely
code_interpreter = CodeInterpreterTool(
    name="code_interpreter",
    description="Execute Python code safely for KPI computation and data analysis"
)