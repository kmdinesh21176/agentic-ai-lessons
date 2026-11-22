# CONCEPT: Subagent Delegation via SubAgentMiddleware (`subagents=[...]`)
# 1. Context Isolation: Subagents maintain independent message histories, returning concise summaries to prevent supervisor context overflow.
# 2. Scoped Tooling: Each specialist subagent is provisioned with a minimal, targeted set of tools.

from pathlib import Path  # Cross-platform path management
from typing import Optional  # Optional type hints
from dotenv import load_dotenv  # Environment variable loader
from pydantic import BaseModel, Field  # Data validation and schema modeling
from deepagents import create_deep_agent  # DeepAgents framework builder
from deepagents.backends import FilesystemBackend  # Local filesystem abstraction
from langchain.agents.middleware import ToolCallLimitMiddleware  # Tool execution limits
from langchain.chat_models import init_chat_model  # Model initialization
from langchain.tools import tool  # Tool definition decorator
from langgraph.checkpoint.memory import InMemorySaver  # In-memory thread checkpointing

load_dotenv()
WORKSPACE_DIR = Path("./workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize sample workspace files for subagent inspection
(WORKSPACE_DIR / "security_policy.md").write_text(
    "# Security Policy\n- API keys must rotate every 90 days.\n- Violation: db-stage-01 key is 142 days old.\n",
    encoding="utf-8",
)
(WORKSPACE_DIR / "q4_budget.md").write_text(
    "# Q4 Cloud Budget\n- Approved Budget: 120000\n- Actual Spend: 138600\n",
    encoding="utf-8",
)
# Define the final structured JSON schema that the Supervisor will populate
class ExecutiveBrief(BaseModel):
    overdue_server: Optional[str] = Field(default=None, description="Server violating the 90-day key rotation policy.")
    budget_overrun_usd: Optional[float] = Field(default=None, description="Dollar amount over the approved Q4 budget.")
    overrun_percentage: Optional[float] = Field(default=None, description="Percentage over the approved Q4 budget.")
    summary: str = Field(description="Two-sentence executive summary combining security and finance findings.")

# Define a domain-specific tool scoped ONLY to the finance subagent
@tool
def calculate_budget_variance(budget: float, actual: float) -> dict:
    """Compute exact dollar overrun and percentage variance between budget and actual spend."""
    overrun = round(actual - budget, 2)  # Exact dollar difference
    pct = round((overrun / budget) * 100, 2)  # Percentage overrun
    return {"overrun_usd": overrun, "overrun_pct": pct}

# Configure Specialist Subagent 1: Security Auditor (uses built-in read_file/grep only)
security_subagent = {
    "name": "security-auditor",
    "description": "Reads /security_policy.md and identifies any server violating rotation rules.",
    "system_prompt": "Call read_file(file_path='/security_policy.md', offset=0, limit=30) once and report the exact violation.",
}

# Configure Specialist Subagent 2: Finance Analyst (given scoped `calculate_budget_variance` tool)
finance_subagent = {
    "name": "finance-analyst",
    "description": "Reads /q4_budget.md and calculates exact budget overrun figures.",
    "system_prompt": "Call read_file(file_path='/q4_budget.md', offset=0, limit=30), then call calculate_budget_variance once.",
    "tools": [calculate_budget_variance],  # Only this subagent can invoke calculate_budget_variance
}
# Initialize the local Ollama chat model (temperature=0 for deterministic delegation)
model = init_chat_model("ollama:qwen2.5:7b", temperature=0, num_ctx=4096, client_kwargs={"timeout": 60.0})
structured_extractor = model.with_structured_output(ExecutiveBrief)  # Post-run JSON schema parser

# Build the Supervisor Deep Agent: SubAgentMiddleware automatically adds the `task` tool
supervisor = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),  # Shared sandboxed root
    subagents=[security_subagent, finance_subagent],  # Registers both specialist workers
    system_prompt=(
    "Delegate ONLY the /security_policy.md check to `security-auditor`, and delegate "
    "ONLY the /q4_budget.md variance calculation to `finance-analyst` via `task`. Then synthesize."),
    middleware=[ToolCallLimitMiddleware(run_limit=8, exit_behavior="end")],  # Guardrail against loops
    checkpointer=InMemorySaver(),  # Persists LangGraph state across steps
)

# Invoke the Supervisor agent on a cross-domain audit task
res = supervisor.invoke(
    {"messages": [{"role": "user", "content": "Audit /security_policy.md and /q4_budget.md using your subagents."}]},
    config={"configurable": {"thread_id": "subagent-audit-1"}, "recursion_limit": 50},
)

# Print the Supervisor's delegation trace (`task` calls and subagent summaries)
for msg in res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"-> Supervisor called {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"<- Subagent returned: {str(msg.content)[:180].replace(chr(10), ' ')}...")

# Convert the Supervisor's final synthesis into a validated Pydantic JSON object
brief: ExecutiveBrief = structured_extractor.invoke(f"Populate ExecutiveBrief from:\n{res['messages'][-1].content}")
print("\nStructured Executive Brief:\n" + brief.model_dump_json(indent=2))
