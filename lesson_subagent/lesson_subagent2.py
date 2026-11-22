# CONCEPT: Multi-Agent Software Development & Verification (SDV) Pipeline
# 1. Role Separation: `feature-developer` writes code; `qa-verifier` writes tests & runs pytest.
# 2. Shared Workspace: Both subagents collaborate on `/src/` and `/tests/` via FilesystemBackend.
# 3. Scoped Execution: Only `qa-verifier` has `run_pytest`, enforcing clean separation of duties.
import sys  # Ensures subprocess uses the active venv Python interpreter

from pathlib import Path  # Cross-platform filesystem paths
import subprocess  # Executes real pytest test suites safely
from typing import Optional  # Typed optional fields
from dotenv import load_dotenv  # Loads environment settings
from pydantic import BaseModel, Field  # Enforces structured release report JSON
from deepagents import create_deep_agent  # DeepAgents orchestrator builder
from deepagents.backends import FilesystemBackend  # Sandboxed virtual filesystem
from langchain.agents.middleware import ToolCallLimitMiddleware  # Prevents infinite tool loops
from langchain.chat_models import init_chat_model  # Initializes local Ollama model
from langchain.tools import tool  # Decorator to expose Python functions as agent tools
from langgraph.checkpoint.memory import InMemorySaver  # Persists LangGraph conversation state

load_dotenv()  # Load variables from .env
WORKSPACE_DIR = Path("./workspace_sdv").resolve()  # Dedicated repo sandbox (hides .venv)
(WORKSPACE_DIR / "src").mkdir(parents=True, exist_ok=True)  # Source code folder
(WORKSPACE_DIR / "tests").mkdir(parents=True, exist_ok=True)  # Automated tests folder

# Seed a realistic Jira/product specification file in the workspace
(WORKSPACE_DIR / "spec.md").write_text(
    "# Feature Spec: Checkout Discount Calculator\n"
    "- File to create: `/src/discount.py`\n"
    "- Function signature: `apply_discount(price: float, tier: str) -> float`\n"
    "- Rules: 'GOLD' gets 20% off, 'SILVER' gets 10% off, any other tier gets 0% off.\n"
    "- Validation: Raise `ValueError` if `price < 0`. Round result to 2 decimal places.\n",
    encoding="utf-8",
)

# Define the final structured Release Verification Report schema
class ReleaseReport(BaseModel):
    implemented_file: str = Field(description="Path of the source module created by the developer.")
    test_file: str = Field(description="Path of the pytest suite created by QA.")
    tests_passed: bool = Field(description="True if pytest exited with return code 0, else False.")
    summary: str = Field(description="Two-sentence release readiness summary grounded in pytest output.")

# Define a real pytest runner tool scoped ONLY to `qa-verifier`
@tool
def run_pytest(test_path: str = "tests/test_discount.py") -> dict:
    """Run pytest inside workspace_sdv and return exit code and test output."""
    clean_rel = test_path.lstrip("/")  # Normalize virtual path '/tests/...' to relative 'tests/...'
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", clean_rel, "-v"],  # Run pytest via active venv Python
            cwd=str(WORKSPACE_DIR),  # Run inside isolated workspace_sdv directory
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as UTF-8 strings
            timeout=30,  # Prevent hanging test runs
        )
        return {
            "exit_code": proc.returncode,  # 0 means all tests passed
            "passed": proc.returncode == 0,  # Boolean pass/fail flag
            "output": (proc.stdout + "\n" + proc.stderr).strip()[-1000:],  # Test log tail
        }
    except Exception as e:
        return {"exit_code": -1, "passed": False, "output": f"Pytest execution error: {e}"}

# Specialist Subagent 1: Software Developer (reads spec.md, writes /src/discount.py)
developer_subagent = {
    "name": "feature-developer",
    "description": "Reads /spec.md and implements clean production code in /src/discount.py.",
    "system_prompt": (
        "1. Call read_file(file_path='/spec.md', offset=0, limit=30) to read requirements.\n"
        "2. Call write_file(file_path='/src/discount.py', content=...) with clean, typed Python code.\n"
        "3. Report the implemented function signature and stop."
    ),
}

# Specialist Subagent 2: QA Verification Engineer (writes /tests/test_discount.py, runs pytest)
qa_subagent = {
    "name": "qa-verifier",
    "description": "Writes pytest unit tests in /tests/test_discount.py and runs run_pytest to verify code.",
    "system_prompt": (
        "1. Call read_file(file_path='/spec.md', offset=0, limit=30) and read_file(file_path='/src/discount.py', offset=0, limit=50).\n"
        "2. Call write_file(file_path='/tests/test_discount.py', content=...) importing `apply_discount` from `src.discount` and testing GOLD, SILVER, REGULAR, and negative price ValueError.\n"
        "3. Call run_pytest(test_path='tests/test_discount.py') and report exact pass/fail results."
    ),
    "tools": [run_pytest],  # Scoped exclusively to the QA verification subagent
}

# Initialize the local Ollama chat model (temperature=0 for deterministic code generation)
model = init_chat_model("ollama:qwen2.5:7b", temperature=0, num_ctx=4096, client_kwargs={"timeout": 60.0})
structured_extractor = model.with_structured_output(ReleaseReport)  # Converts final report to Pydantic JSON

# Build the Tech Lead (Supervisor) Deep Agent orchestrating sequential Dev -> QA stages
tech_lead = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),  # Shared repo sandbox
    subagents=[developer_subagent, qa_subagent],  # Registers both Development & QA workers
    system_prompt=(
        "You are a Tech Lead. Execute sequentially:\n"
        "1. Delegate implementation of /spec.md to `feature-developer` via `task` and wait for completion.\n"
        "2. Delegate test creation and pytest execution to `qa-verifier` via `task`.\n"
        "3. Summarize the final release readiness."
    ),
    middleware=[ToolCallLimitMiddleware(run_limit=8, exit_behavior="end")],  # Prevents runaway tool loops
    checkpointer=InMemorySaver(),  # Stores LangGraph execution state
)

# Run the end-to-end Software Development & Verification workflow
res = tech_lead.invoke(
    {"messages": [{"role": "user", "content": "Implement /spec.md using feature-developer, then verify it using qa-verifier."}]},
    config={"configurable": {"thread_id": "sdv-pipeline-1"}, "recursion_limit": 60},
)

# Print the Tech Lead's delegation trace showing Dev -> QA handoff
for msg in res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"-> Tech Lead delegated {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"<- Subagent result: {str(msg.content)[:180].replace(chr(10), ' ')}...")

# Extract the final structured ReleaseReport JSON
report: ReleaseReport = structured_extractor.invoke(f"Populate ReleaseReport from:\n{res['messages'][-1].content}")
print("\nStructured Release Report:\n" + report.model_dump_json(indent=2))
