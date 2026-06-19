import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    info = """
    Mechanism design (sometimes implementation theory or institution design)[1] is a branch of economics and game theory. It studies how to construct rules—called mechanisms or institutions—that produce good outcomes according to some predefined metric, even when the designer does not know the players' true preferences or what information they have. Mechanism design thus focuses on the study of solution concepts for a class of private-information games.

    Mechanism design has broad applications, including traditional domains of economics such as market design, but also political science (through voting theory). It is a foundational component in the operation of the internet, being used in networked systems (such as inter-domain routing),[2] e-commerce, and advertisement auctions by Facebook and Google.

    Because it starts with the end of the game (a particular result), then works backwards to find a game that implements it, it is sometimes described as reverse game theory.[2] Leonid Hurwicz explains that "in a design problem, the goal function is the main given, while the mechanism is the unknown. Therefore, the design problem is the inverse of traditional economic theory, which is typically devoted to the analysis of the performance of a given mechanism."[3]

    The 2007 Nobel Memorial Prize in Economic Sciences was awarded to Leonid Hurwicz, Eric Maskin, and Roger Myerson "for having laid the foundations of mechanism design theory."[4] The related works of William Vickrey that established the field earned him the 1996 Nobel prize
    """

    summary_template = """
    given the information {information} about a topic I want you to create:
    1. a short summary
    2. two interesting facts about it
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    #llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information" : info})

    print(response)

if __name__ == "__main__":
    main()
