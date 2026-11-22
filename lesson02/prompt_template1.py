
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Initialize the local model
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive=0
)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms with an example."
)

# Create a chain
chain = prompt | llm

# Execute the chain
response = chain.invoke({
    "topic": "LangChain"
})

print(response.content)
