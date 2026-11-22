from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# 1. Initialize your local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 2. Define a custom tool using LangChain's @tool decorator.
# NOTE: The docstring and type hints are crucial—the LLM reads them 
# to understand what the tool does and what arguments it expects!
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b

# 3. Bind the tool to the LLM
# This tells the model: "Hey, you now have access to these tools."
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)

# 4. Invoke the model with a prompt that requires the tool
response = llm_with_tools.invoke("Can you calculate 1425 multiplied by 389? Use your tool.")

# 5. Inspect what the LLM decided to do
print("--- LLM Response Object ---")
print(response)

print("\n--- Tool Calls Requested by LLM ---")
print(response.tool_calls)