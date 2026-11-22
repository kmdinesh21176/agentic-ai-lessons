
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

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

# Chain 2: Summarize explanation
summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the following explanation in 3 bullet points:

    {explanation}
    """
)

summary_chain = summary_prompt | llm | StrOutputParser()

# Connect both chains
overall_chain = (
    {
        "explanation": explanation_chain,
    }
    | summary_chain
)

# Execute
result = overall_chain.invoke({
    "topic": "Python programming"
})

print("\nFINAL SUMMARY:")
print(result)
