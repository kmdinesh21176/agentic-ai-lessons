
# Import BaseModel to define a structured data model
# Import Field to describe each field
from pydantic import BaseModel, Field

# Import ChatOllama to connect to the local model
from langchain_ollama import ChatOllama


# Define the expected structure of a student
class Student(BaseModel):

    # Require a non-empty student name
    name: str = Field(min_length=2)

    # Restrict age to a reasonable range
    age: int = Field(ge=18, le=100)

    # Require at least one skill
    skills: list[str] = Field(min_length=1)

    # Require a meaningful experience description
    experience: str = Field(min_length=20)


# Initialize the local Llama model
llm = ChatOllama(
    model="llama3.2",  # Select the Ollama model
    temperature=0,     # Use deterministic-style generation
    num_ctx=2048,      # Set context window
    num_predict=256,   # Limit generated tokens
    keep_alive=0       # Unload model after request
)


# Ask LangChain to produce output matching the Student schema
# This integration must support structured output for the selected model
structured_llm = llm.with_structured_output(Student)


# Send a request to generate a sample student profile
# Ask the model to generate a realistic sample student profile
result = structured_llm.invoke(
    """
    Create a realistic sample profile for a Python developer.

    Requirements:
    - Name must be a realistic sample name.
    - Age must be between 20 and 35.
    - Include at least 4 programming skills.
    - Experience must be a meaningful sentence
      describing the developer's experience.
    - Do not leave any field empty.
    """
)


# Print the complete Pydantic object
print(result)


# Access individual fields using dot notation
print("\nStudent name:", result.name)
print("Age:", result.age)
print("Skills:", result.skills)
print("Experience:", result.experience)


# Check the Python type of the returned object
print("\nResult type:")
print(type(result))
