
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced Python teacher. Explain concepts simply."),
    ("human", "Explain {topic} with a practical example.")
])

chain = prompt | llm

response = chain.invoke({
    "topic": "Python functions"
})

print(response.content)
