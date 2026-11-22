"""
================================================================================
CONCEPT & ARCHITECTURE OVERVIEW
================================================================================

1. Core Concept: Standard Tool-Calling Agent vs. Filesystem-Backed Deep Agent
   When an LLM must analyze a large external document (such as the 10,035-line
   RFC 2616 specification, ~430 KB), stuffing the entire file into the prompt
   exceeds local context limits (`num_ctx=4096`) and degrades reasoning accuracy.
   Instead, the agent must offload the document to disk and inspect it via tools:

   - Standard Agent (`create_agent`):
     Executes a basic ReAct loop equipped only with the tools you explicitly
     register (`fetch_text_from_url`). It can download the file to disk, but
     because it has no filesystem inspection tools (`read_file`, `grep`), it
     cannot verify line counts or line numbers and must honestly abstain by
     returning `null` rather than fabricating data.

   - Deep Agent (`create_deep_agent`):
     Wraps the LangGraph agent loop with built-in middleware (`FilesystemMiddleware`,
     `TodoListMiddleware`, `SubAgentMiddleware`) and a sandboxed storage backend
     (`FilesystemBackend`). This automatically grants the LLM `ls`, `read_file`,
     `write_file`, `edit_file`, `glob`, and `grep` so it can selectively read
     sections and search massive files without overflowing its context window.

2. How It Is Implemented in Detail:
   - Workspace Sandboxing (`FilesystemBackend(..., virtual_mode=True)`):
     Isolates the agent's virtual root `/` to `./workspace`. Without isolation,
     `root_dir="."` exposes `.venv` (thousands of site-packages files), causing
     `grep` to scan dependencies and time out.
   - Two-Phase Tool Execution + Structured Extraction (`with_structured_output`):
     Ollama's constrained JSON `format` mode suppresses tool calls if enabled
     during the agent loop. Running the agent with unconstrained tool calling
     first—and then pairing each `tool_call_id` with its result in a final
     extraction step—guarantees reliable tool execution AND exact JSON output.
   - Local LLM Stability (`init_chat_model` + `recursion_limit`):
     Configures `temperature=0` for deterministic tool arguments, `num_ctx=4096`
     to prevent local VRAM/RAM stalls, `timeout=60.0` to prevent infinite HTTP
     hangs, and `recursion_limit=100` so multi-step read/search flows finish cleanly.

================================================================================
"""

from pathlib import Path
import re
from typing import Optional
import urllib.request

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# ==============================================================================
# STEP 1: Configure an Isolated Workspace Directory
# ==============================================================================
WORKSPACE_DIR = Path("./workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# STEP 2: Define the Structured Output Schema (Pydantic)
# ==============================================================================
class DocumentAnalysis(BaseModel):
    transfer_encoding_line_count: Optional[int] = Field(
        default=None,
        description="Exact number of lines containing 'Transfer-Encoding' (case-sensitive), or null if unverified.",
    )
    cache_control_first_line: Optional[int] = Field(
        default=None,
        description="1-based line number of the first line containing 'Cache-Control', or null if unverified.",
    )
    synopsis: str = Field(
        description="A two-sentence neutral synopsis grounded in the document text."
    )


# ==============================================================================
# STEP 3: Define Role-Specific System Prompts
# ==============================================================================
LC_SYSTEM_PROMPT = """You are a technical document assistant.
- Call `fetch_text_from_url` once to download 'rfc2616.txt'.
- You do NOT have `read_file` or `grep` tools, so you CANNOT inspect the saved file.
- Immediately after fetching, write your final answer stating `null` for items 1 and 2, plus a two-sentence neutral synopsis for item 3."""

DA_SYSTEM_PROMPT = """You are a technical document assistant with filesystem tools (`read_file`, `grep`, `ls`).
Follow these exact steps once:
1. Call `fetch_text_from_url` to download 'rfc2616.txt'.
2. Call `read_file(file_path="/rfc2616.txt", offset=1, limit=60)` to read the title and abstract before writing your synopsis.
3. Call `grep(pattern="Transfer-Encoding", glob="rfc2616.txt", output_mode="count")` to get the exact matching line count.
4. Call `grep(pattern="Cache-Control", glob="rfc2616.txt", output_mode="content")` to get the 1-based line number of the first match.
5. Never repeat the same tool call. Ground every number strictly in tool outputs."""


# ==============================================================================
# STEP 4: Implement the Custom Document-Fetching Tool
# ==============================================================================
@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the plain-text document from a URL and save it to 'rfc2616.txt' in the workspace."""
    fallback_url = "https://www.rfc-editor.org" + "/rfc/rfc2616.txt"
    target_url = (
        url
        if url.startswith(("http://", "https://")) and url.endswith(".txt")
        else fallback_url
    )
    save_path = WORKSPACE_DIR / "rfc2616.txt"
    print(f"\n[TOOL CALL] fetch_text_from_url(using={target_url!r})", flush=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(target_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            save_path.write_text(text, encoding="utf-8")
            total_lines = len(text.splitlines())
            return (
                f"Saved {total_lines} lines to '/rfc2616.txt'. "
                "Use `read_file` and `grep` if available; otherwise answer with null for counts."
            )
    except Exception as e:
        return f"Fetch failed: {e}"


# ==============================================================================
# STEP 5: Initialize the Local Chat Model & Memory Checkpointer
# ==============================================================================
model = init_chat_model(
    "ollama:qwen2.5:7b",
    temperature=0,
    num_ctx=4096,
    client_kwargs={"timeout": 60.0},
)

# Dedicated structured-output extractor run AFTER tool calls complete
structured_extractor = model.with_structured_output(DocumentAnalysis)

checkpointer = InMemorySaver()


# ==============================================================================
# STEP 6: Build Both Agents (Unconstrained Tool Loop)
# ==============================================================================
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=LC_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
    system_prompt=DA_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


# ==============================================================================
# STEP 7: Helper to Extract Validated Pydantic JSON from Agent Trace
# ==============================================================================
def extract_structured_json(res: dict, allow_counts: bool) -> str:
    """Pair tool calls with their outputs and return a validated DocumentAnalysis JSON object."""
    call_lookup = {}
    for m in res["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                call_lookup[tc["id"]] = f"{tc['name']}({tc['args']})"

    paired_traces = []
    verified_count = None
    verified_first_line = None

    for m in res["messages"]:
        if m.type == "tool":
            call_sig = call_lookup.get(getattr(m, "tool_call_id", ""), m.name)
            content_str = str(m.content)
            paired_traces.append(f"CALL: {call_sig}\nRESULT: {content_str[:1200]}")

            # Deterministically parse built-in grep outputs so 7B models never garble digits
            if "Transfer-Encoding" in call_sig and "count" in call_sig:
                match = re.search(r":\s*(\d+)\s*$", content_str.strip())
                if match:
                    verified_count = int(match.group(1))
            elif "Cache-Control" in call_sig and "content" in call_sig:
                match = re.search(r":\s*(\d+):", content_str)
                if match:
                    verified_first_line = int(match.group(1))

    final_answer = str(res["messages"][-1].content).strip()

    extraction_prompt = (
        "Write a two-sentence neutral synopsis grounded in the read_file/tool trace below.\n\n"
        f"Tool Trace:\n{chr(10).join(paired_traces) or 'None'}\n\n"
        f"Agent Final Response:\n{final_answer or 'None'}"
    )
    parsed: DocumentAnalysis = structured_extractor.invoke(extraction_prompt)

    if allow_counts:
        parsed.transfer_encoding_line_count = verified_count
        parsed.cache_control_first_line = verified_first_line
    else:
        parsed.transfer_encoding_line_count = None
        parsed.cache_control_first_line = None

    return parsed.model_dump_json(indent=2)

# ==============================================================================
# STEP 8: Define the User Task & Execute Both Agents
# ==============================================================================
prompt = """The RFC Editor hosts a full plain-text copy of RFC 2616 (Hypertext Transfer Protocol -- HTTP/1.1).
URL: https://rfc-editor.org/rfc/rfc2616.txt

Answer as much as you can:
1) Exact number of lines containing `Transfer-Encoding` (case-sensitive).
2) 1-based line number of the first line containing `Cache-Control`.
3) A two-sentence neutral synopsis grounded in the opening lines of the file."""

print("Running create_agent...", flush=True)
agent_res = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": "lc-structured"}, "recursion_limit": 10},
)
print("\ncreate_agent structured_response:")
print(extract_structured_json(agent_res, allow_counts=False))
print("\nRunning create_deep_agent (built-in read_file + grep)...", flush=True)
da_res = deep_agent.invoke(
{"messages": [{"role": "user", "content": prompt}]},
config={"configurable": {"thread_id": "da-structured"}, "recursion_limit": 100},
)
print("\ndeep_agent tool trace:")
for msg in da_res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  -> {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        preview = str(msg.content)[:140].replace("\n", " ")
        print(f"  <- {msg.name}: {preview}...")
print("\ncreate_deep_agent structured_response:")
print(extract_structured_json(da_res, allow_counts=True))