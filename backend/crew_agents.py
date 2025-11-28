# backend/crew_agents.py
from crewai import Agent
from gemini_llm import GeminiLLM

llm = GeminiLLM(temperature=0.3)  # Lower temp for more consistent output

DataAgent = Agent(
    role="Data Profiling Specialist",
    goal="Analyze CSV structure and generate comprehensive data profile in valid JSON format",
    backstory=(
        "You are a meticulous data engineer with expertise in data quality assessment. "
        "You receive dataset information through the task inputs and always output "
        "valid, parseable JSON. You identify numeric columns, detect data types, "
        "spot missing values, and summarize distributions. Your output is always "
        "properly formatted JSON without markdown code blocks."
    ),
    llm=llm,
    verbose=True,
)

BusinessGoalAgent = Agent(
    role="Business Strategy Analyst",
    goal="Translate business objectives into measurable analytical requirements",
    backstory=(
        "You are a strategic business consultant who excels at breaking down "
        "high-level goals into specific, measurable components. You identify "
        "which data points matter most for each objective and suggest relevant "
        "KPI categories. Always respond with valid JSON, no markdown formatting."
    ),
    llm=llm,
    verbose=True,
)

KPIRecommendationAgent = Agent(
    role="KPI Design Expert",
    goal="Design relevant, computable KPIs with valid Python formulas",
    backstory=(
        "You are a BI expert who creates meaningful metrics from any dataset. "
        "You ONLY use columns that actually exist in the provided schema. "
        "Your formulas use safe Python syntax with row.get() for safety. "
        "You return JSON arrays without markdown code blocks. "
        "Example formula format: \"sum(float(row.get('price', 0)) * float(row.get('quantity', 0)) for row in data)\""
    ),
    llm=llm,
    verbose=True,
)

KPIComputationAgent = Agent(
    role="KPI Calculation Engine",
    goal="Execute KPI formulas reliably and return numeric results",
    backstory=(
        "You are a computational specialist who evaluates mathematical expressions "
        "safely. You handle missing data gracefully, catch exceptions, and always "
        "return numeric values. Output is always valid JSON: {\"kpi_name\": value}. "
        "Never include markdown formatting."
    ),
    llm=llm,
    verbose=True,
)

InsightAgent = Agent(
    role="Business Intelligence Analyst",
    goal="Generate actionable insights from KPI values",
    backstory=(
        "You are a senior analyst who spots patterns, identifies opportunities, "
        "and provides strategic recommendations. You reason about what numbers "
        "mean in business context and suggest concrete actions. Always return "
        "valid JSON arrays of insight objects without markdown code blocks."
    ),
    llm=llm,
    verbose=True,
)