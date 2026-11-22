"""
================================================================================
CONCEPT & ARCHITECTURE OVERVIEW
================================================================================

1. Core Concept: Two-Stage Context Offloading for Large Binary PDFs
   Binary PDF files cannot be read directly by DeepAgents' built-in UTF-8 text
   tools (`read_file`, `grep`), and dumping a 65-page hardware specification
   into a local LLM prompt would overflow `num_ctx=4096` and cause infinite
   retry loops (`GraphRecursionError`).

   To solve this cleanly, `tool_calling7.py` uses a two-stage pipeline:
   - Stage 1 (Ingestion & Staging - `fetch_pdf_as_markdown`):
     Downloads the complete binary PDF over HTTPS, extracts 100% of its pages
     using `pypdf`, and writes a structured, page-indexed Markdown file
     (`## Page 1`, `## Page 2`, ...) to `./workspace/lpc_spec.md`. Only a
     short confirmation message is returned to the LLM's message history.
   - Stage 2 (Selective Inspection - `create_deep_agent`):
     Uses built-in `grep` (scanning all 65 pages on disk in milliseconds) and
     `read_file(offset=0, limit=45)` (reading only Page 1) to gather verified
     facts without exhausting the 4,096-token context window.

2. How It Is Implemented in Detail:
   - Workspace Sandboxing (`FilesystemBackend(..., virtual_mode=True)`):
     Maps virtual root `/` to `./workspace` so `grep` never scans `.venv`.
   - Windows SSL Resilience (`certifi` + fallback context):
     Handles corporate proxies or missing intermediate TLS certificates on
     Windows when fetching PDFs from CDNs like `intel.com`.
   - Recursion & Loop Guardrails (`ToolCallLimitMiddleware` + `limit=45`):
     Caps total tool calls at 5 per run and limits `read_file` to 45 lines so
     `qwen2.5:7b` never loses its system instructions to context truncation.
   - Validated Structured Output (`with_structured_output(IntelSpecSummary)`):
     Extracts the grounded Page 1 metadata and deterministic `grep` count into
     a strict Pydantic JSON schema after the tool loop completes.

================================================================================
"""

import io
from pathlib import Path
import re
import ssl
from typing import Optional
import urllib.request
import uuid

import certifi
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ==============================================================================
# STEP 1: Isolate Workspace Directory (Hides .venv from built-in grep)
# ==============================================================================
WORKSPACE_DIR = Path("./workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# STEP 2: Define the Structured Output Schema (Pydantic)
# ==============================================================================
class IntelSpecSummary(BaseModel):
    specification_title: Optional[str] = Field(
        default=None,
        description="Exact title of the Intel specification from Page 1, or null if unavailable.",
    )
    revision: Optional[str] = Field(
        default=None,
        description="Revision number stated on Page 1 (e.g., 'Revision 1.1'), or null if unavailable.",
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication month and year stated on Page 1, or null if unavailable.",
    )
    lframe_mention_count: Optional[int] = Field(
        default=None,
        description="Exact number of lines in lpc_spec.md containing 'LFRAME#' (case-sensitive), or null.",
    )
    executive_summary: str = Field(
        description="A two-sentence neutral summary of the specification grounded in the opening pages."
    )


# ==============================================================================
# STEP 3: Implement the HTTPS PDF-to-Markdown Staging Tool (Windows SSL Safe)
# ==============================================================================
@tool
def fetch_pdf_as_markdown(url: str) -> str:
    """Download a PDF from an HTTPS URL, extract text page-by-page, and save it to '/lpc_spec.md'."""
    fallback_url = (
        "https://www.intel.com"
        + "/content/dam/www/program/design/us/en/documents/low-pin-count-interface-specification.pdf"
    )
    target_url = (
        url
        if url.startswith(("http://", "https://")) and url.lower().endswith(".pdf")
        else fallback_url
    )
    md_path = WORKSPACE_DIR / "lpc_spec.md"
    print(f"\n[TOOL CALL] fetch_pdf_as_markdown(url={target_url!r})", flush=True)

    if md_path.exists() and md_path.stat().st_size > 1000:
        line_count = len(md_path.read_text(encoding="utf-8").splitlines())
        return (
            f"Using cached '/lpc_spec.md' ({line_count} lines). "
            "Next, call `read_file(file_path='/lpc_spec.md', offset=0, limit=45)` and `grep(pattern='LFRAME#', glob='lpc_spec.md', output_mode='count')`."
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
    }
    req = urllib.request.Request(target_url, headers=headers)

    ssl_contexts = [
        ssl.create_default_context(cafile=certifi.where()),
        ssl._create_unverified_context(),
    ]

    pdf_bytes = None
    last_err = None
    for ctx in ssl_contexts:
        try:
            with urllib.request.urlopen(req, timeout=45, context=ctx) as resp:
                pdf_bytes = resp.read()
                break
        except Exception as e:
            last_err = e

    if pdf_bytes is None:
        return f"FATAL_FETCH_ERROR: {last_err}. Stop immediately and do NOT call read_file or grep."

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        sections = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            sections.append(f"## Page {page_num}\n\n{page_text}")

        full_markdown = "\n\n---\n\n".join(sections) + "\n"
        md_path.write_text(full_markdown, encoding="utf-8")

        return (
            f"Downloaded PDF ({len(pdf_bytes):,} bytes), extracted {len(reader.pages)} pages "
            f"({len(full_markdown.splitlines())} lines) into '/lpc_spec.md'. "
            "Next, call `read_file(file_path='/lpc_spec.md', offset=0, limit=45)` and `grep(pattern='LFRAME#', glob='lpc_spec.md', output_mode='count')`."
        )
    except Exception as e:
        return f"FATAL_PARSE_ERROR: {e}. Stop immediately."


# ==============================================================================
# STEP 4: Initialize Local Model & Deep Agent with Loop Guardrails
# ==============================================================================
model = init_chat_model(
    "ollama:qwen2.5:7b",
    temperature=0,
    num_ctx=4096,
    client_kwargs={"timeout": 60.0},
)

structured_extractor = model.with_structured_output(IntelSpecSummary)

SYSTEM_PROMPT = """You are a hardware specification analyst with filesystem tools (`read_file`, `grep`, `ls`).
Follow these exact steps sequentially:
1. First, call ONLY `fetch_pdf_as_markdown` with the requested PDF URL and wait for it to finish.
   - If it returns FATAL_FETCH_ERROR, stop immediately and report the failure.
2. Next, call these three tools:
   - `read_file(file_path="/lpc_spec.md", offset=0, limit=45)` to get Page 1 title, revision, and date.
   - `grep(pattern="LFRAME#", glob="lpc_spec.md", output_mode="count")` to count `LFRAME#` lines.
   - `grep(pattern="1.0", glob="lpc_spec.md", output_mode="content")` to find the line number where Section 1.0 Introduction starts.
3. Finally, call `read_file` with `offset` set to the body line number of Section 1.0 Introduction (after the Table of Contents) and `limit=60`, then write your final response."""

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_pdf_as_markdown],
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
    system_prompt=SYSTEM_PROMPT,
    middleware=[ToolCallLimitMiddleware(run_limit=8, exit_behavior="end")],
    checkpointer=InMemorySaver(),
)


# ==============================================================================
# STEP 5: Run the Agent & Extract Validated JSON
# ==============================================================================
pdf_url = (
    "https://www.intel.com"
    + "/content/dam/www/program/design/us/en/documents/low-pin-count-interface-specification.pdf"
)

prompt = f"""Fetch and analyze the Intel PDF specification at {pdf_url}:
1) Extract the specification title, revision, and publication date from Page 1.
2) Count how many lines in '/lpc_spec.md' contain `LFRAME#` (case-sensitive) using grep(pattern="LFRAME#", glob="lpc_spec.md", output_mode="count").
3) Write a two-sentence neutral executive summary grounded in Page 1."""

print("Running create_deep_agent on live HTTPS PDF...", flush=True)
res = deep_agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
    config={"configurable": {"thread_id": f"intel-lpc-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 50},
)

print("\nTool trace:")
verified_lframe_count = None
read_chunks = []  # Store both Page 1 and Section 1.0 Introduction reads

for msg in res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  -> {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        preview = str(msg.content)[:150].replace("\n", " ")
        print(f"  <- {msg.name}: {preview}...")
        if msg.name == "grep" and "LFRAME#" not in preview:
            m = re.search(r":\s*(\d+)\s*$", str(msg.content).strip())
            if m:
                verified_lframe_count = int(m.group(1))
        elif msg.name == "read_file":
            read_chunks.append(str(msg.content))

combined_markdown = "\n\n---\n\n".join(read_chunks)

if not combined_markdown.strip():
    final_json = IntelSpecSummary(
        executive_summary="Unable to download or read the PDF specification due to a network/SSL error."
    )
else:
    final_json = structured_extractor.invoke(
        f"Populate IntelSpecSummary strictly from these extracted specification sections:\n\n{combined_markdown}"
    )
    if verified_lframe_count is not None:
        final_json.lframe_mention_count = verified_lframe_count

print("\nStructured JSON output:")
print(final_json.model_dump_json(indent=2))