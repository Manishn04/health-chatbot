
from langchain.prompts import PromptTemplate

# Simple prompt template for health chatbot
chatbot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful health assistant chatbot.
You have access to medical and wellness knowledge but you're not a substitute for a doctor.

Context:
{context}

User question:
{question}

Answer in a clear and concise way, and include a note if users should seek professional help.
"""
)
from langchain.prompts import PromptTemplate

# Simple prompt template for health chatbot
chatbot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful health assistant chatbot.
You have access to medical and wellness knowledge but you're not a substitute for a doctor.

Context:
{context}

User question:
{question}

Answer in a clear and concise way, and include a note if users should seek professional help.
"""
)
