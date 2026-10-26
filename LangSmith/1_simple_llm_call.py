from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)

'''
The runs in LangSmith for this are:
1. PromptTemplate
2. ChatGoogleGenerativeAI
3. StrOutputParser
'''