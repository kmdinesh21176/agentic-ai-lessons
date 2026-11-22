
from langchain_ollama import ChatOllama

# Initialize local LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Send a prompt
response = llm.invoke("What is LangChain?")

# Print the response
print(response.content)
