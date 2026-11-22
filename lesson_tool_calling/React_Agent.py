from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent  # <--- Updated import

# 1. Initialize your local Llama 3.2 model via Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Define a couple of tools for our agent to use
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b

tools = [multiply, add]

# 3. Create the agent using LangChain's modern create_agent function
# (Note: we use `system_prompt` instead of passing raw message structures in the invoke step)
agent_executor = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt="You are a helpful mathematical assistant."
)

# 4. Invoke the agent with a multi-step math problem using the standard input format
print("--- Running Modern Agent ---")
response = agent_executor.invoke({
    "messages": [("user", "First calculate 50 plus 50, and then multiply that result by 10.")]
})

# 5. Print out the full conversation history to see the agent's thought process
print("\n--- Full Agent Execution Steps ---")
for message in response["messages"]:
    msg_type = message.__class__.__name__
    print(f"\n[{msg_type}]:")
    print(message.content)
    if hasattr(message, "tool_calls") and message.tool_calls:
        print(f"Tool Calls Requested: {message.tool_calls}")