# CONCEPT: Stage 4 - Human-in-the-Loop (HITL) Approvals with `interrupt_on` & `Command(resume=...)`
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()
WORKSPACE_DIR = Path("./workspace_hitl").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

(WORKSPACE_DIR / "release_notes.md").write_text(
    "# Firmware v2.4.0 Release Candidate\n- Binary: `/build/imu_fw_v2_4.elf`\n- Target MCU: `STM32F407VG`\n- Unity Tests: 12/12 PASSED\n",
    encoding="utf-8",
)

@tool
def flash_firmware_to_mcu(binary_path: str, mcu_target: str, erase_flash: bool = True) -> str:
    """Flash a compiled ELF firmware binary onto a physical microcontroller over SWD/JTAG."""
    return f"FLASH_SUCCESS: Programmed {binary_path} onto {mcu_target} (erase_flash={erase_flash})."
import uuid
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command  # Reserved for resuming in Part 3

# Initialize local Ollama model and build the DeepAgent with an interrupt on flash_firmware_to_mcu
model = init_chat_model("ollama:qwen2.5:7b", temperature=0, num_ctx=4096, client_kwargs={"timeout": 60.0})

release_agent = create_deep_agent(
    model=model,
    tools=[flash_firmware_to_mcu],
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
    system_prompt="You are an Embedded Release Engineer. First read '/release_notes.md', then call `flash_firmware_to_mcu` once.",
    interrupt_on={"flash_firmware_to_mcu": True},
    middleware=[ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")],
    checkpointer=InMemorySaver(),
)

# Step 1: Run the agent; it reads the release notes and pauses prior to executing the flash tool
thread_config = {"configurable": {"thread_id": f"hitl-fw-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 50}
paused_state = release_agent.invoke(
    {"messages": [{"role": "user", "content": "Verify /release_notes.md and flash the firmware to the target MCU."}]},
    config=thread_config,
)
# Inspect the __interrupt__ payload generated before flash_firmware_to_mcu ran
if "__interrupt__" in paused_state:
    interrupt_info = paused_state["__interrupt__"][0].value  # Extract interrupt dictionary
    action_req = interrupt_info["action_requests"][0]  # First pending tool call
    print("\n[HITL INTERRUPT] Agent paused before sensitive tool execution:")
    print(f"  Tool Name : {action_req['name']}")
    print(f"  Arguments : {action_req['args']}")

    # Ask the human engineer to approve, edit arguments, or reject the tool call
    choice = input("\nDecision ([a]pprove / [e]dit erase_flash=False / [r]eject): ").strip().lower()
    if choice.startswith("e"):  # Modify arguments before execution
        edited_args = {**action_req["args"], "erase_flash": False}
        decision = {"type": "edit", "edited_action": {"name": action_req["name"], "args": edited_args}}
    elif choice.startswith("r"):  # Block the tool call with feedback
        decision = {"type": "reject", "message": "Rejected by firmware lead: hold release until hardware review."}
    else:  # Approve original tool arguments as-is
        decision = {"type": "approve"}

    # Step 2: Resume the paused LangGraph thread using the exact same thread_config
    final_state = release_agent.invoke(Command(resume={"decisions": [decision]}), config=thread_config)
else:
    final_state = paused_state  # Fallback if no interrupt was triggered

print("\nFinal Agent Response:\n" + str(final_state["messages"][-1].content))
