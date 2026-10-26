'''
This file implements a simple single agent.

References:

1. For Weather Stack API: https://docs.apilayer.com/weatherstack/docs/quickstart-guide

2. For Agent implementation:
i. https://reference.langchain.com/python/langchain/agents/factory/create_agent
ii. https://www.krishnaik.in/ai-tutorials/langchain/create-agent
'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import requests
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun


os.environ["LANGSMITH_PROJECT"] = "Agent Demo"
load_dotenv()
weather_stack_api_key = os.getenv("WEATHERSTACK_API_KEY")


# Tool 1
search_tool = DuckDuckGoSearchRun()

# Tool 2
def get_weather_data(city: str) -> str:
    """
    Retrives the real-time weather data for a given city
    """

    print("Using the tool: 'get_weather_data'")
    url = f'https://api.weatherstack.com/current?access_key={weather_stack_api_key}&query={city}'
    response = requests.get(url).json()

    return response


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt="You are a helpful assistant"
)

# Tool 1: What is the release date of Dhadak 2?
# input = {
#     "messages": [{
#         "role": "user",
#         "content": "What is the release date of Dhadak 2?"
#     }]
# }

# Tool 2: What is the current temp of Gurgaon?
# input = {
#     "messages": [{
#         "role": "user",
#         "content": "What is the current temp of Gurgaon?"
#     }]
# }

# Tools 1 and 2: Identify the birthplace of Gandhi and give its current temperature.
input = {
    "messages": [{
        "role": "user",
        "content": "Identify the birthplace of Gandhi and give its current temperature."
    }]
}

# This is a single graph execution
response = agent.invoke(input)
print(f"Full response: {response}")

# Print the final assistant output from the messages list
print(f"\nRequired response: {response['messages'][-1].text}")