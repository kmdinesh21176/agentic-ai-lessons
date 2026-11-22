from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Initialize your local Llama 3.2 model via Ollama with zero creativity for precise tool usage
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Turn a standard Python function into a tool using the @tool decorator.
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together and return the result."""
    return a * b

# 3. Group our tools into a list and bind them to the LLM
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)

# 4. Create a dictionary to map string tool names to actual executable Python functions
tool_map = {
    "multiply": multiply
}

# 5. Define the user prompt requiring math and store it as a HumanMessage object
user_prompt = "Can you calculate 1425 multiplied by 389? Use your tool."
messages = [HumanMessage(content=user_prompt)]

# 6. Send the message history to the model
print("--- Step 1: Sending prompt to LLM ---")
response = llm_with_tools.invoke(messages)

# 7. Append the model's response (which contains the tool call request) to our message history
messages.append(response)

print("\n--- Step 2: LLM Requested Tool ---")
print(response.tool_calls)

# 8. Check if the LLM actually requested any tools
if response.tool_calls:
    # Loop through each tool call requested by the model
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]  # Important: We need this ID to match the response back to the call!
        
        print(f"\n--- Step 3: Executing Python Function '{tool_name}' ---")
        print(f"Arguments passed: {tool_args}")
        
        # Look up the function and execute it
        selected_tool = tool_map[tool_name]
        tool_result = selected_tool.invoke(tool_args)
        
        print(f"Tool Output: {tool_result}")
        
        # 9. Create a ToolMessage containing the result and the matching tool_call_id
        # This tells the LLM: "Here is the exact answer to the tool call you requested."
        tool_msg = ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call_id
        )
        
        # Append the tool message to our running message history
        messages.append(tool_msg)

    # 10. Send the entire message history BACK to the LLM
    print("\n--- Step 4: Sending Tool Result Back to LLM for Final Answer ---")
    final_response = llm_with_tools.invoke(messages)
    
    # 11. Print the LLM's final conversational response to the user
    print("\n--- Final LLM Response ---")
    print(final_response.content)