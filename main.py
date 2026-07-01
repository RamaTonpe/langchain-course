from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of the given product."""
    print({product})
    prices = {"laptop":10, "headphones":5}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Calculate discount on product price based on discount tier.
    Available discount: Gold, Silver, Bronze."""
    print({price}, {discount_tier})
    discounts = {"Gold":0.1, "Silver":0.05, "Bronze":0.02}
    return price - price*discounts.get(discount_tier, 0)

@traceable (name="LangChain Agent Loop")
def run_agent(question: str): 
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(              #query
        content="You are a helpful shopping assistant." 
        "You have access to a product catalog and discounts."
        "Apply the discount to the product price and return the discounted price"
        ),
        HumanMessage(content=question)
        ]
    
    for iteration in range(1, MAX_ITERATIONS+1):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls


        if not tool_calls:  #final answer
            print("final answer")
            print(ai_message.content)
            print(messages)
            return ai_message.content
    
        tool_call = tool_calls[0] #first tool call
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError({tool_name} ,"not found")
        
        observation = tool_to_use.invoke(tool_args) #action-call/use tool

        print(observation)

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id)) #give observation back to llm

    print("maxed out")



    

if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying the Gold discount?")


