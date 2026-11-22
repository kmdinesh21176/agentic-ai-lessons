"""
================================================================================
CONCEPT: HTML Web Page Ingestion -> Clean Markdown Scratchpad -> Deep Agent
================================================================================
1. Live HTTP Fetch: Downloads a real HTML webpage (with tags, scripts, styles).
2. BeautifulSoup Cleaning: Strips non-content tags (`script`, `style`, `nav`,
   `footer`, `header`) and converts structural HTML (`h1`-`h3`, `p`, `li`, `a`,
   `pre`) into clean, readable Markdown saved to `./workspace/page.md`.
3. Deep Agent Inspection: Uses built-in `read_file` and `grep` on `/page.md`
   to answer questions accurately without stuffing raw HTML into the prompt.
================================================================================
"""

from pathlib import Path
import re
from typing import Optional
import urllib.request

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# 1. Isolated workspace directory
WORKSPACE_DIR = Path("./workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# 2. Structured output schema
class WebPageSummary(BaseModel):
    page_title: str = Field(description="Main title or H1 heading of the webpage.")
    domain_mention_count: Optional[int] = Field(
        default=None,
        description="Exact number of lines in page.md containing 'Example Domain' (case-sensitive).",
    )
    summary: str = Field(description="A two-sentence neutral summary of the webpage content.")


# 3. HTML -> Clean Markdown converter using BeautifulSoup
def html_to_markdown(html: str) -> str:
    """Strip boilerplate HTML tags and convert main content elements to clean Markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content / noisy tags
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    root = soup.body or soup
    md_lines: list[str] = []

    for el in root.find_all(["h1", "h2", "h3", "p", "li", "pre"], recursive=True):
        # Skip nested <p> inside <li> to avoid duplicate text
        if el.name == "p" and el.find_parent("li"):
            continue

        # Convert inline links <a href="...">text</a> -> [text](href)
        for a in el.find_all("a", href=True):
            link_text = a.get_text(" ", strip=True)
            href = a["href"].strip()
            if link_text and href:
                a.replace_with(f"[{link_text}]({href})")

        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue

        if el.name == "h1":
            md_lines.append(f"# {text}")
        elif el.name == "h2":
            md_lines.append(f"## {text}")
        elif el.name == "h3":
            md_lines.append(f"### {text}")
        elif el.name == "li":
            md_lines.append(f"- {text}")
        elif el.name == "pre":
            md_lines.append(f"```\n{el.get_text().strip()}\n```")
        else:
            md_lines.append(text)

    return "\n\n".join(md_lines) + "\n"


# 4. Web-fetching tool that saves clean Markdown to ./workspace/page.md
@tool
def fetch_webpage_as_markdown(url: str) -> str:
    """Fetch an HTML webpage from `url`, convert it to clean Markdown, and save it to '/page.md'."""
    target_url = url if url.startswith(("http://", "https://")) else "https://www.iana.org/domains"
    save_path = WORKSPACE_DIR / "page.md"
    print(f"\n[TOOL CALL] fetch_webpage_as_markdown(url={target_url!r})", flush=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    req = urllib.request.Request(target_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
            markdown_text = html_to_markdown(raw_html)
            save_path.write_text(markdown_text, encoding="utf-8")
            total_lines = len(markdown_text.splitlines())
            return (
                f"Fetched HTML from {target_url}, converted to Markdown, and saved {total_lines} lines to '/page.md'. "
                "Now use `read_file` and `grep` on '/page.md' to inspect the content."
            )
    except Exception as e:
        return f"Fetch failed: {e}"


# 5. Initialize local Ollama model & Deep Agent
model = init_chat_model(
    "ollama:qwen2.5:7b",
    temperature=0,
    num_ctx=4096,
    client_kwargs={"timeout": 60.0},
)
structured_extractor = model.with_structured_output(WebPageSummary)

# SYSTEM_PROMPT = """You are a web research assistant with filesystem tools (`read_file`, `grep`, `ls`).
# Follow these exact steps once:
# 1. Call `fetch_webpage_as_markdown` with the requested URL to save '/page.md'.
# 2. Call `read_file(file_path="/page.md", offset=1, limit=50)` to read the converted Markdown content.
# 3. Call `grep(pattern="Example Domain", glob="page.md", output_mode="count")` to count matching lines.
# 4. Never repeat the same tool call. Ground your answer strictly in the tool outputs."""


SYSTEM_PROMPT = """You are a web research assistant with filesystem tools (`read_file`, `grep`, `ls`).
Follow these exact steps sequentially:
1. First, call ONLY `fetch_webpage_as_markdown` with the requested URL and wait for the result.
2. Next, call `read_file(file_path="/page.md", offset=0, limit=50)` (note: offset is 0-based so line 1 is included) and `grep(pattern="root zone", glob="page.md", output_mode="count")`.
3. Never repeat the same tool call. Ground your answer strictly in the tool outputs."""


deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_webpage_as_markdown],
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

# 6. Run against a real HTML webpage (https://www.iana.org/domains)
prompt = """Fetch and analyze the HTML webpage at https://www.iana.org/domains:
1) Identify the main page title.
2) Count how many lines in the saved Markdown file contain `Example Domain` (case-sensitive).
3) Write a two-sentence neutral summary of what the page says."""

print("Running create_deep_agent on live HTML webpage...", flush=True)
res = deep_agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": "html-md-demo"}, "recursion_limit": 50},
)

print("\nTool trace:")
verified_count = None
read_content = ""
for msg in res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  -> {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        preview = str(msg.content)[:160].replace("\n", " ")
        print(f"  <- {msg.name}: {preview}...")
        if msg.name == "grep":
            m = re.search(r":\s*(\d+)\s*$", str(msg.content).strip())
            if m:
                verified_count = int(m.group(1))
        elif msg.name == "read_file":
            read_content = str(msg.content)

# 7. Extract final structured JSON
final_summary: WebPageSummary = structured_extractor.invoke(
    f"Populate WebPageSummary strictly from this Markdown content:\n\n{read_content}"
)
if verified_count is not None:
    final_summary.domain_mention_count = verified_count

print("\nSaved Markdown file preview (./workspace/page.md):")
print((WORKSPACE_DIR / "page.md").read_text(encoding="utf-8"))

print("Structured JSON output:")
print(final_summary.model_dump_json(indent=2))
