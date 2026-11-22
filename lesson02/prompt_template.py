
from langchain_core.prompts import ChatPromptTemplate

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

# Provide a value for the variable
formatted_prompt = prompt.invoke({
    "topic": "Python"
})

# Display the formatted prompt
print(formatted_prompt)
