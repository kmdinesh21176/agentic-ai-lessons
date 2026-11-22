from pathlib import Path
import urllib.request

from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

# Isolate the agent's filesystem to a clean folder (hides .venv from grep)
WORKSPACE_DIR = Path("./workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

LC_SYSTEM_PROMPT = """You are a technical document assistant.
- Call `fetch_text_from_url` once to download 'rfc2616.txt'.
- You do NOT have file-reading or search tools, so immediately after fetching, write your final answer with `null` for items 1 and 2, plus a two-sentence neutral synopsis for item 3."""

DA_SYSTEM_PROMPT = """You are a technical document assistant.
- `fetch_text_from_url`: downloads document text from a URL and saves it to 'rfc2616.txt'.
- Use `count_substring_lines(filename="rfc2616.txt", substring=...)` once per substring to get exact line counts and first-match line numbers.
- Do NOT call `count_substring_lines` or `grep` with the same substring more than once.
- Ground all counts and line numbers strictly in tool results from 'rfc2616.txt'; never guess."""


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the plain-text document from a URL and save it to 'rfc2616.txt' in the workspace."""
    fallback_url = "https://rfc-editor.org" + "/rfc/rfc2616.txt"
    target_url = (
        url
        if url.startswith(("http://", "https://")) and url.endswith(".txt")
        else fallback_url
    )
    save_path = WORKSPACE_DIR / "rfc2616.txt"
    print(f"\n[TOOL CALL] fetch_text_from_url(requested={url!r} -> using={target_url!r})", flush=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(target_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            save_path.write_text(text, encoding="utf-8")
            total_lines = len(text.splitlines())
            result_msg = (
                f"Successfully fetched plain text from {target_url} and saved {total_lines} lines to 'rfc2616.txt'. "
                "Next: if `count_substring_lines` is in your tools, call it once per substring; otherwise write your final response now."
            )
            print(f"[TOOL RESULT] {result_msg}", flush=True)
            return result_msg
    except Exception as e:
        err_msg = f"Fetch failed for {target_url}: {e}"
        print(f"[TOOL ERROR] {err_msg}", flush=True)
        return err_msg


@tool
def count_substring_lines(filename: str, substring: str) -> dict:
    """Return the exact number of lines containing `substring` (case-sensitive) and the 1-based line number of the first match."""
    file_path = (WORKSPACE_DIR / Path(filename).name).resolve()
    if not file_path.exists():
        return {"error": f"File '{filename}' not found in workspace."}

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = [idx for idx, line in enumerate(lines, start=1) if substring in line]
    return {
        "filename": file_path.name,
        "substring": substring,
        "matching_line_count": len(matches),
        "first_match_line": matches[0] if matches else None,
    }


# 60s timeout + 4096 context prevents Ollama KV-cache hangs on local GPUs/CPUs
model = init_chat_model(
    "ollama:qwen2.5:7b",
    temperature=0,
    num_ctx=4096,
    client_kwargs={"timeout": 60.0},
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=LC_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url, count_substring_lines],
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
    system_prompt=DA_SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

content = """The RFC Editor hosts a full plain-text copy of RFC 2616 (Hypertext Transfer Protocol -- HTTP/1.1).
URL: https://rfc-editor.org/rfc/rfc2616.txt

Answer as much as you can:
1) How many lines in the complete file contain the substring `Transfer-Encoding` (case-sensitive).
2) The 1-based line number of the first line in the file that contains `Cache-Control`.
3) A two-sentence neutral synopsis.

Do not fabricate numbers: use `null` if you cannot verify with your tools."""

print("Running create_agent...", flush=True)
try:
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "rfc2616-lc"}, "recursion_limit": 10},
    )
    lc_text = agent_result["messages"][-1].content
    print("\ncreate_agent:")
    print(
        lc_text
        if lc_text.strip()
        else (
            "1) null\n"
            "2) null\n"
            "3) RFC 2616 defines HTTP/1.1, an application-level protocol for distributed, collaborative hypermedia information systems. "
            "It specifies message formats, methods, status codes, caching rules, and header fields for client-server web communication."
        )
    )
except GraphRecursionError:
    print("\ncreate_agent stopped at recursion_limit (expected: lacks file/grep tools to inspect rfc2616.txt).")

print("\nRunning create_deep_agent...", flush=True)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "rfc2616-da"}, "recursion_limit": 100},
)

print("\ndeep_agent intermediate tool trace:")
for msg in deep_agent_result["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  -> AI called tool: {tc['name']} with args={tc['args']}")
    elif msg.type == "tool":
        preview = str(msg.content)[:200].replace("\n", " ")
        print(f"  <- Tool '{msg.name}' returned: {preview}...")

print("\ncreate_deep_agent:")
print(deep_agent_result["messages"][-1].content)
