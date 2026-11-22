
# Import the local Ollama chat model
from langchain_ollama import ChatOllama

# Import the prompt template component
from langchain_core.prompts import ChatPromptTemplate

# Import JsonOutputParser to parse JSON responses
from langchain_core.output_parsers import JsonOutputParser


# Initialize the Ollama model
llm = ChatOllama(
    model="llama3.2",  # Use the locally downloaded model
    temperature=0,     # Keep responses as consistent as possible
    num_ctx=2048,      # Set the context window
    num_predict=256,   # Limit the response length
    keep_alive=0       # Unload the model after the request
)


# Create a prompt template
# {language} will be replaced with the language we provide
prompt = ChatPromptTemplate.from_template(
    """
    Return information about {language}.

    Return ONLY valid JSON with these fields:
    name, creator, year, use_cases.

    use_cases must be a list of strings.
    Do not include Markdown code fences.
    """
)


# Connect the prompt, LLM, and JSON parser
# The parser converts the generated JSON text into Python data
chain = prompt | llm | JsonOutputParser()


# Execute the chain and provide the language name
result = chain.invoke({
    "language": "Python"
})


# Print the complete parsed Python dictionary
print(result)


# Access a single value using its dictionary key
print("\nLanguage name:")
print(result["name"])


# Access the list stored under the use_cases key
print("\nUse cases:")

# Loop through each item in the use_cases list
for use_case in result["use_cases"]:
    print("-", use_case)
