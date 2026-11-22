
# Import ChatOllama to connect LangChain to our local Ollama model
from langchain_ollama import ChatOllama

# Import ChatPromptTemplate to create prompts with variables
from langchain_core.prompts import ChatPromptTemplate

# Import StrOutputParser to convert the LLM response into a string
from langchain_core.output_parsers import StrOutputParser


# Initialize the local Llama 3.2 model
llm = ChatOllama(
    model="llama3.2",  # Name of the model downloaded in Ollama
    temperature=0,     # Use 0 for more consistent responses
    num_ctx=2048,      # Maximum context window for this request
    num_predict=256,   # Limit the generated response length
    keep_alive=0       # Unload the model after the request
)


# Create a prompt template with a variable called topic
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple language."
)


# Connect the prompt, LLM, and output parser using LCEL
# | means pass the output of one component to the next
chain = prompt | llm | StrOutputParser()


# Execute the chain and provide a value for the topic variable
result = chain.invoke({
    "topic": "Python functions"
})


# Print the generated response
print(result)


# Display the Python data type of the response
# StrOutputParser should return a normal Python string
print("\nResult type:")
print(type(result))
