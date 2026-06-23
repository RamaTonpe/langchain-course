from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
#from tavily import TavilyClient
import os


#tavily = TavilyClient()


# class Source(BaseModel):
#     """Schema for a source used by the agent"""

#     url: str = Field(description="The URL of the source")


# class AgentResponse(BaseModel):
#     """Schema for agent response with answer and sources"""

#     answer: str = Field(description="Thr agent's answer to the query")
#     sources: List[Source] = Field(
#         default_factory=list, description="List of sources used to generate the answer"
#     )

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args: 
#         query: The query to search for
#     Returns:
#         The search result
#     """

#     print(f"searching for {query}")
#     return tavily.search(query=query)



#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(model="llama-3.3-70b-versatile")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="what is the weather in Tokyo?")]})
    print(result)



if __name__ == "__main__":
    main()
