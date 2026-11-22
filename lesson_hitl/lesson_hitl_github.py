# CONCEPT: Stage 4 - HITL Credential Injection & Pure-Git Backup for C:\Dinesh\langchain-learning
# 1. Zero-Leak Auth: Agent passes username/token="PENDING_HITL"; human enters real credentials at __interrupt__.
# 2. Proxy-Safe Git Push: Uses native `git.exe` (Windows Schannel TLS) instead of Python `urllib` to avoid HTTP 406.
# 3. Clean Scope: Auto-generates `.gitignore` so `venv/`, `workspace*/`, and `.env` are never committed.

from pathlib import Path  # Cross-platform filesystem paths
import subprocess  # Executes native git CLI commands
from dotenv import load_dotenv  # Loads .env settings
from langchain.tools import tool  # Registers Python functions as LangChain tools

load_dotenv()  # Load environment variables
PROJECT_ROOT = Path(r"C:\Dinesh\langchain-learning").resolve()  # Root folder of your practiced lessons
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__"}  # Directories excluded from lesson scan

@tool
def list_practiced_lessons() -> list[str]:
    """Return relative paths of all practiced Python lesson scripts in C:\\Dinesh\\langchain-learning."""
    return sorted(
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in PROJECT_ROOT.rglob("*.py")
        if not any(part in SKIP_DIRS or part.startswith("workspace") for part in p.parts)
    )

import base64  # Encodes GitHub PAT safely into HTTP Basic Authorization header

@tool
def push_lessons_to_github(repo_name: str, username: str, token: str) -> str:
    """Configure .gitignore and push all practiced lessons in C:\\Dinesh\\langchain-learning to GitHub via git."""
    if not token or token == "PENDING_HITL" or not username or username == "PENDING_HITL":
        return "AUTH_REJECTED: Missing GitHub username or Personal Access Token."

    (PROJECT_ROOT / ".gitignore").write_text("venv/\n.venv/\n__pycache__/\n*.pyc\n.env\nworkspace*/\n", encoding="utf-8")
    clean_url = f"https://github.com{username}/{repo_name}.git"  # Standard HTTPS Git URL (no embedded token)
    b64_auth = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")  # Safe HTTP Basic header

    cmds = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", username],
        ["git", "config", "user.email", f"{username}@users.noreply.github.com"],
        ["git", "add", "."],
        ["git", "commit", "-m", "Backup practiced Agentic AI lessons"],
        ["git", "remote", "remove", "origin"],
        ["git", "remote", "add", "origin", clean_url],
        ["git", "-c", f"http.https://github.com.extraheader=AUTHORIZATION: basic {b64_auth}", "push", "-u", "origin", "main", "--force"],
    ]
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if proc.returncode != 0 and cmd[1] not in ("commit", "remote"):
            safe_err = proc.stderr.replace(token, "***").replace(b64_auth, "***").strip()[:300]
            return f"GIT_ERROR ({' '.join(cmd[:2])}): {safe_err}"
    return f"BACKUP_SUCCESS: Pushed practiced lessons to https://github.com{username}/{repo_name}"



import uuid  # Generates unique thread_id per execution
from deepagents import create_deep_agent  # DeepAgents orchestrator builder
from langchain.agents.middleware import ToolCallLimitMiddleware  # Caps tool calls to prevent loops
from langchain.chat_models import init_chat_model  # Initializes local Ollama chat model
from langgraph.checkpoint.memory import InMemorySaver  # Persists paused LangGraph state
from langgraph.types import Command  # Resumes paused interrupt thread with human decision

# Initialize local Ollama model (temperature=0 for deterministic tool calling)
model = init_chat_model("ollama:qwen2.5:7b", temperature=0, num_ctx=4096, client_kwargs={"timeout": 60.0})

# Build the GitHub Backup DeepAgent with HITL approval enforced on `push_lessons_to_github`
backup_agent = create_deep_agent(
    model=model,
    tools=[list_practiced_lessons, push_lessons_to_github],  # Custom scanner + pure-git push tools
    system_prompt=(
        "You are a DevOps Backup Assistant. Follow these exact steps:\n"
        "1. Call `list_practiced_lessons()` first to discover practiced scripts.\n"
        "2. Call `push_lessons_to_github(repo_name='agentic-ai-lessons', username='PENDING_HITL', token='PENDING_HITL')`.\n"
        "3. Never ask for passwords in chat text; rely on the tool interrupt."
    ),
    interrupt_on={"push_lessons_to_github": True},  # Pauses before executing git push
    middleware=[ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")],  # Loop guardrail
    checkpointer=InMemorySaver(),  # Required for interrupt_on + Command(resume=...)
)

# Step 1: Start the agent; it scans your lessons and pauses right before `push_lessons_to_github`
thread_config = {"configurable": {"thread_id": f"gh-hitl-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 50}
paused_state = backup_agent.invoke(
    {"messages": [{"role": "user", "content": "Scan my practiced lessons and back them up to GitHub."}]},
    config=thread_config,
)

# Print which practiced lesson files the agent found before pausing
for msg in paused_state["messages"]:
    if msg.type == "tool" and msg.name == "list_practiced_lessons":
        print(f"\n[SCANNED LESSONS] Found practiced files:\n{msg.content}")

# Step 2: Handle the HITL interrupt — collect GitHub repo name, username, and PAT visibly in PowerShell
if "__interrupt__" in paused_state:
    interrupt_info = paused_state["__interrupt__"][0].value  # Extract interrupt payload
    action_req = interrupt_info["action_requests"][0]  # Pending push_lessons_to_github call
    print("\n[HITL INTERRUPT] GitHub Login & Backup Approval Required")
    print(f"  Proposed Repo : {action_req['args'].get('repo_name')}")

    custom_repo = input("Repository name [press Enter for 'agentic-ai-lessons']: ").strip() or "agentic-ai-lessons"
    gh_user = input("Enter GitHub username: ").strip()
    pat = input("Enter GitHub Personal Access Token (ghp_... or github_pat_...): ").strip()

    if not gh_user or not pat:  # Reject if username or token is left blank
        decision = {"type": "reject", "message": "User cancelled GitHub push (missing username or token)."}
    else:  # Inject real credentials & repo name via 'edit' decision directly into tool args
        edited_args = {**action_req["args"], "repo_name": custom_repo, "username": gh_user, "token": pat}
        decision = {"type": "edit", "edited_action": {"name": action_req["name"], "args": edited_args}}

    # Step 3: Resume the paused LangGraph thread with the human decision
    final_state = backup_agent.invoke(Command(resume={"decisions": [decision]}), config=thread_config)
else:
    final_state = paused_state  # Fallback if no interrupt fired

# Print the raw git tool result before the LLM summary
for msg in final_state["messages"]:
    if msg.type == "tool" and msg.name == "push_lessons_to_github":
        print(f"\n[TOOL RESULT] {msg.content}")

print("\nFinal Agent Response:\n" + str(final_state["messages"][-1].content))
