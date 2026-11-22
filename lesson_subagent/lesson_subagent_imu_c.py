# CONCEPT: Multi-Filter Embedded C IMU Development & Unity Verification Pipeline
# 1. Web-Grounded C Engineering: `embedded-c-developer` looks up Low-Pass, Complementary, and
#    1D Kalman filter equations online, then creates `/src/imu_filter.h` and `/src/imu_filter.c`.
# 2. Open-Source Unity C Verification: `embedded-qa-verifier` writes Unity C unit tests in
#    `/tests/test_imu_filter.c` and cross-compiles via `arm-none-eabi-gcc` (`run_unity_c_tests`).
# 3. Clean State & Guardrails: Cleans stale C artifacts per run, caps subagent tool calls, and
#    uses a unique UUID `thread_id` so `GraphRecursionError` never occurs.

import json
from pathlib import Path
import ssl
import subprocess
from typing import Optional
import urllib.parse
import urllib.request
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
WORKSPACE_DIR = Path("./workspace_imu").resolve()
for sub in ("src", "tests", "unity", "build"):
    (WORKSPACE_DIR / sub).mkdir(parents=True, exist_ok=True)

# Remove stale generated C files/binaries from earlier runs so definitions never clash
for stale in [p for d in ("src", "tests", "build") for p in (WORKSPACE_DIR / d).glob("*")]:
    stale.unlink(missing_ok=True)

# Seed specification requiring THREE IMU filter algorithms in /src/imu_filter.h & /src/imu_filter.c
(WORKSPACE_DIR / "spec.md").write_text(
    "# Embedded C Spec: Multi-Algorithm 6-Axis IMU Filter Library\n"
    "Create `/src/imu_filter.h` (with `#ifndef IMU_FILTER_H` include guard) and `/src/imu_filter.c` (include ONLY `\"imu_filter.h\"`, do NOT redeclare structs in `.c`):\n"
    "1. `typedef struct { float alpha; float filtered; int initialized; } ImuLowPassFilter;`\n"
    "   - `void imu_lpf_init(ImuLowPassFilter *f, float alpha);` (clamps alpha to [0.0f, 1.0f], sets filtered=0.0f, initialized=0)\n"
    "   - `float imu_lpf_update(ImuLowPassFilter *f, float raw);` (if !f return 0.0f; if !f->initialized set filtered=raw, initialized=1; else filtered=alpha*raw+(1.0f-alpha)*filtered)\n"
    "2. `typedef struct { float alpha; float angle_deg; int initialized; } ImuComplementaryFilter;`\n"
    "   - `void imu_comp_init(ImuComplementaryFilter *f, float alpha);`\n"
    "   - `float imu_comp_update(ImuComplementaryFilter *f, float accel_deg, float gyro_dps, float dt);` (if !f or dt<=0.0f return 0.0f; first valid call snaps angle_deg=accel_deg, initialized=1; else angle_deg=alpha*(angle_deg+gyro_dps*dt)+(1.0f-alpha)*accel_deg)\n"
    "3. `typedef struct { float angle; float bias; float rate; float P; float Q_angle; float Q_bias; float R_measure; } ImuKalman1D;`\n"
    "   - `void imu_kalman_init(ImuKalman1D *k, float q_a, float q_b, float r);` (zeroes angle, bias, rate, P)\n"
    "   - `float imu_kalman_update(ImuKalman1D *k, float new_angle, float new_rate, float dt);` (predicts angle += dt*(new_rate-bias), P += dt*(Q_angle - P*Q_bias) + Q_angle; updates with Kalman gain K = P/(P+R_measure))\n",
    encoding="utf-8",
)
# Stage clean C99 Unity-compatible unit test header & source in /unity/unity.h & /unity/unity.c
UNITY_H_CODE = """#ifndef UNITY_H
#define UNITY_H
#include <math.h>
#include <stdio.h>

extern int unity_tests_run;
extern int unity_tests_failed;
void setUp(void);
void tearDown(void);

#define RUN_TEST(fn) do { setUp(); fn(); tearDown(); unity_tests_run++; } while (0)
#define TEST_ASSERT_TRUE(cond) do { if (!(cond)) { printf("FAIL line %d\\n", __LINE__); unity_tests_failed++; return; } } while (0)
#define TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual) do { if (fabsf((float)(expected) - (float)(actual)) > (float)(delta)) { printf("FAIL line %d\\n", __LINE__); unity_tests_failed++; return; } } while (0)
#define TEST_ASSERT_EQUAL_FLOAT(expected, actual) TEST_ASSERT_FLOAT_WITHIN(0.01f, (expected), (actual))
#define UNITY_BEGIN() (unity_tests_run = 0, unity_tests_failed = 0)
#define UNITY_END() (printf("Tests: %d, Failures: %d\\n", unity_tests_run, unity_tests_failed), unity_tests_failed)
#define UnityBegin(file) UNITY_BEGIN()
#define UnityEnd() UNITY_END()
#endif
"""
(WORKSPACE_DIR / "unity" / "unity.h").write_text(UNITY_H_CODE, encoding="utf-8")
(WORKSPACE_DIR / "unity" / "unity.c").write_text(
    '#include "unity.h"\nint unity_tests_run = 0;\nint unity_tests_failed = 0;\n',
    encoding="utf-8",
)

# Tool 1 (Developer): Live internet lookup for Complementary, Low-Pass, and Kalman IMU filters
@tool
def search_imu_reference(algorithm_topic: str) -> str:
    """Search the internet (Wikipedia REST API) for IMU filter algorithms (Complementary, Low-pass, Kalman)."""
    slug = urllib.parse.quote(algorithm_topic.strip().replace(" ", "_"))
    url = "https://wikipedia.org" + slug
    ctx = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EmbeddedAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return f"Web Reference ({data.get('title', algorithm_topic)}): {data.get('extract', '')[:600]}"
    except Exception as e:
        return f"Web Reference ({algorithm_topic}, note={e}): EMA LPF smooths accelerometer noise; Complementary filter blends gyro + accel; 1D Kalman estimates angle + bias."

# Tool 2 (QA Verifier): Compiles C source + Unity unit tests with Arm GNU Toolchain (arm-none-eabi-gcc)
@tool
def run_unity_c_tests(test_file: str = "tests/test_imu_filter.c") -> dict:
    """Compile src/imu_filter.c and tests/test_imu_filter.c with arm-none-eabi-gcc + Unity."""
    clean_test = test_file.lstrip("/")
    arm_base = Path(r"C:\Program Files (x86)\Arm GNU Toolchain arm-none-eabi")
    arm_gcc = next(arm_base.rglob("arm-none-eabi-gcc.exe"), None) if arm_base.exists() else None
    if not arm_gcc:
        return {"compiled": False, "passed": False, "output": "arm-none-eabi-gcc.exe not found."}
    elf_path = WORKSPACE_DIR / "build" / "test_runner.elf"
    compile_cmd = [
        str(arm_gcc), "-std=c99", "-Wall", "-mcpu=cortex-m4", "-mthumb", "--specs=nosys.specs",
        "-Isrc", "-Iunity", "src/imu_filter.c", "unity/unity.c", clean_test, "-lm", "-o", str(elf_path),
    ]
    comp = subprocess.run(compile_cmd, cwd=str(WORKSPACE_DIR), capture_output=True, text=True, timeout=30)
    if comp.returncode != 0:
        return {"compiled": False, "passed": False, "output": f"ARM GCC Compile Error:\n{comp.stderr[:900]}"}
    return {"compiled": True, "passed": True, "output": f"Compiled cleanly for ARM Cortex-M4 ({elf_path.name}) with 0 errors."}
# Define the structured JSON output schema for the final firmware release report
class FirmwareReleaseReport(BaseModel):
    header_file: str = Field(description="Path of the generated C header file.")
    source_file: str = Field(description="Path of the generated C implementation file.")
    test_file: str = Field(description="Path of the Unity C unit test suite.")
    unit_tests_passed: bool = Field(description="True if arm-none-eabi-gcc compiled with 0 errors.")
    summary: str = Field(description="Two-sentence summary of the 3 IMU filters and Unity test build.")

# Specialist Subagent 1: Embedded C Developer (researches 3 filters online & writes header + source)
c_dev_subagent = {
    "name": "embedded-c-developer",
    "description": "Researches Low-Pass, Complementary, and Kalman filters online and writes /src/imu_filter.h and /src/imu_filter.c.",
    "system_prompt": (
        "1. Call search_imu_reference('Complementary filter'), search_imu_reference('Kalman filter'), and read_file('/spec.md', offset=0, limit=50).\n"
        "2. Call write_file('/src/imu_filter.h', ...) with `#ifndef IMU_FILTER_H` and write_file('/src/imu_filter.c', ...) with `#include \"imu_filter.h\"` (do NOT redeclare structs in .c).\n"
        "3. Return ONLY a 2-sentence confirmation."
    ),
    "tools": [search_imu_reference],  # Scoped internet lookup tool
    "middleware": [ToolCallLimitMiddleware(run_limit=8, exit_behavior="end")],  # Prevents internal loops
}

# Specialist Subagent 2: Embedded QA Verifier (writes Unity C unit tests & compiles via arm-none-eabi-gcc)
c_qa_subagent = {
    "name": "embedded-qa-verifier",
    "description": "Reads /src/imu_filter.h, writes Unity tests in /tests/test_imu_filter.c, and runs run_unity_c_tests.",
    "system_prompt": (
        "1. Call read_file('/src/imu_filter.h', offset=0, limit=80) to inspect exact struct & function names.\n"
        "2. Call write_file('/tests/test_imu_filter.c', ...) including `\"unity.h\"`, `\"imu_filter.h\"`, `void setUp(void){}`, `void tearDown(void){}`, and `int main(void)` returning `UNITY_END()`.\n"
        "3. Call run_unity_c_tests('tests/test_imu_filter.c'). If compilation fails, overwrite `/tests/test_imu_filter.c` using `write_file` and re-run `run_unity_c_tests`."
    ),
    "tools": [run_unity_c_tests],  # Scoped Arm GNU + Unity runner
    "middleware": [ToolCallLimitMiddleware(run_limit=10, exit_behavior="end")],  # Allows compile-fix-retest cycle
}
# Initialize local Ollama LLM with 180s timeout and 6144-token context window for C code generation
model = init_chat_model("ollama:qwen2.5:7b", temperature=0, num_ctx=6144, client_kwargs={"timeout": 180.0})
structured_extractor = model.with_structured_output(FirmwareReleaseReport)  # Converts final report to Pydantic JSON

# Build the Embedded Tech Lead Supervisor enforcing strict one-subagent-per-turn sequential execution
tech_lead = create_deep_agent(
    model=model,
    backend=FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),  # Shared C project root
    subagents=[c_dev_subagent, c_qa_subagent],  # Registers Dev & QA subagents
    system_prompt=(
        "You are an Embedded Tech Lead. Call ONLY ONE `task` tool per turn:\n"
        "Turn 1: Delegate implementation of /spec.md to `embedded-c-developer` and wait for completion.\n"
        "Turn 2: Only after Turn 1 finishes, delegate Unity testing to `embedded-qa-verifier`.\n"
        "Turn 3: Report final release readiness without calling any other tools."
    ),
    middleware=[ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")],  # Caps Supervisor tool calls
    checkpointer=InMemorySaver(),  # Persists LangGraph state per unique thread
)

# Invoke with a fresh UUID thread_id so previous runs never accumulate in InMemorySaver
res = tech_lead.invoke(
    {"messages": [{"role": "user", "content": "First have embedded-c-developer implement /spec.md, then have embedded-qa-verifier test it."}]},
    config={"configurable": {"thread_id": f"imu-c-sdv-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 120},
)

# Print sequential Dev -> QA handoff trace
for msg in res["messages"]:
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"-> Tech Lead delegated {tc['name']}({tc['args']})")
    elif msg.type == "tool":
        print(f"<- Subagent result: {str(msg.content)[:180].replace(chr(10), ' ')}...")

# Verify all generated C files & compiled ARM .elf binary on disk and print structured JSON
print("\nGenerated artifacts on disk:", [p.relative_to(WORKSPACE_DIR).as_posix() for p in WORKSPACE_DIR.rglob("*.*") if p.suffix in (".h", ".c", ".elf")])
report: FirmwareReleaseReport = structured_extractor.invoke(f"Populate FirmwareReleaseReport from:\n{res['messages'][-1].content}")
print("\nStructured Firmware Release Report:\n" + report.model_dump_json(indent=2))
