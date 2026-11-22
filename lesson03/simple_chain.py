
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Step 1: Initialize Ollama
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive=0
)

# Step 2: Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms with an example."
)

# Step 3: Create LCEL chain
chain = prompt | llm

# Step 4: Execute chain
response = chain.invoke({
    "topic": "LangChain"
})

# Step 5: Print response
print(response.content)
