
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

# Connect all three chains
overall_chain = (
    RunnableParallel(
        explanation=explanation_chain,
        topic=lambda x: x["topic"]
    )
    | RunnableParallel(
        explanation=lambda x: x["explanation"],
        topic=lambda x: x["topic"]
    )
    | RunnableParallel(
        explanation=lambda x: x["explanation"],
        example=example_code_chain
    )
    | RunnableParallel(
        example=lambda x: x["example"],
        summary=summary_chain
    )
)

# Execute
result = overall_chain.invoke({
    "topic": "Python programming"
})

print("\nPYTHON EXAMPLE:")
print(result["example"])

print("\nFINAL SUMMARY:")
print(result["summary"])
