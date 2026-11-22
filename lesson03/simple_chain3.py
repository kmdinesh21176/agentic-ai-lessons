from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize Ollama
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    num_ctx=2048,
    num_predict=256,
    keep_alive=0
)

# Chain 1: Generate explanation
explanation_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail using simple language."
)
explanation_chain = explanation_prompt | llm | StrOutputParser()

# Chain 2: Generate Python example
example_code_prompt = ChatPromptTemplate.from_template(
    """
    Based on this explanation, generate a simple,
    working Python coding example for {topic}.

    Explanation:
    {explanation}

    Include the Python code and explain how it works.
    """
)
example_code_chain = example_code_prompt | llm | StrOutputParser()

# Chain 3: Summarize explanation
summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the following explanation in 3 bullet points:

    {explanation}
    """
)
summary_chain = summary_prompt | llm | StrOutputParser()

# --- STREAMLINED OVERALL CHAIN ---
overall_chain = (
    # 1. Start with {"topic": "..."} and generate "explanation"
    {"explanation": explanation_chain, "topic": lambda x: x["topic"]}
    
    # 2. Assign the "example" using the existing "topic" and "explanation"
    | RunnablePassthrough.assign(example=example_code_chain)
    
    # 3. Assign the "summary" using the existing "explanation"
    | RunnablePassthrough.assign(summary=summary_chain)
)

# Execute
result = overall_chain.invoke({
    "topic": "Python programming"
})

print("\nPYTHON EXAMPLE:")
print(result["example"])

print("\nFINAL SUMMARY:")
print(result["summary"])