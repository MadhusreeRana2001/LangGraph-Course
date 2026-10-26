import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# This overwrites the vairable set inside .env only for this file without having to change this variable's value globally
os.environ["LANGSMITH_PROJECT"] = "two-layer-sequence"
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic} within 1000 words. Refrain from creating any tables or flowcharts.',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1, model2 = "gemini-3.8-flash", "gemini-3.5-flash-lite"
llm1 = ChatGoogleGenerativeAI(model=model1)
llm2 = ChatGoogleGenerativeAI(model=model2)
parser = StrOutputParser()
chain = prompt1 | llm1 | parser | prompt2 | llm2 | parser

# Configuration for additional tags and metadata
config = {
    # Give the trace a custom name instead of the default 'RunnableSequence'
    'run_name': 'sequential chain',
    'tags': [
        'llm app',
        'report generation',
        'summarization'
    ],
    'metadata': {
        'model1': model1,
        'model2': model2
    }
}

result = chain.invoke(
    {'topic': 'Unemployment in India'},
    config=config
)
print(result)