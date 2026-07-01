from dotenv import load_dotenv

load_dotenv()



import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type = "tool")
def get_product_price(product: str) -> float:
    """Look up the price of the given product."""
    print({product})
    prices = {"laptop":10, "headphones":5}
    return prices.get(product, 0)

@traceable(run_type = "tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Calculate discount on product price based on discount tier.
    Available discount: Gold, Silver, Bronze."""
    print({price}, {discount_tier})
    discounts = {"Gold":0.1, "Silver":0.05, "Bronze":0.02}
    return price - price*discounts.get(discount_tier, 0)

tools_for_llm = [
    
    {
            "type": "function",
            "function": {
                "name": "get_product_price",
                "description": "Look up the price of the given product. Products include laptop and headphones",
                "parameters": {
                    "type": "object",
                    "required" : ["product"],
                    "properties": {
                        "product": {"type": "string", "description": "The name of the product"}
                    },
                },
            },
    },

     {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "Calculate discount on product price based on discount tier. Available discount: Gold, Silver, Bronze. Discount rates are set by the function",
                "parameters": {
                    "type": "object",
                    "required" : ["price", "discount_tier"],
                    "properties": {
                        "price": {"type": "float", "description": "The price of the product"},
                        "discount_tier": {"type": "string", "description": "The discount tier"}
                    },
                },
            },
    },


]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

@traceable (name="LangChain Agent Loop")
def run_agent(question: str): 
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    messages = [{"role": "system",               #query
        "content":"""You are a helpful shopping assistant. 
        You have access to a product catalog and discounts.
        Apply the discount to the product price and return the discounted price
        Make sure you use the tools available and do not do any calculations yourself
        Do not make assumpions about the discounts
        Only use the numbers provided by the user or by the tools. DO NOT hallucinate
        DO NOT do any calculations yourself only use the tools provided"""

        },
        {"role": "user", "content": question}
        ]
    
    for iteration in range(1, MAX_ITERATIONS+1):
        response = ollama_chat_trace(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls


        if not tool_calls:  #final answer
            print("final answer")
            print(ai_message.content)
            print(messages)
            return ai_message.content
    
        tool_call = tool_calls[0] #first tool call
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError({tool_name} ,"not found")
        
        observation = tool_to_use(**tool_args) #action-call/use tool

        print(observation)

        messages.append(ai_message)
        messages.append({
            "role": "tool", 
            "content" : str(observation)
            }
        ) #give observation back to llm

    print("maxed out")



    

if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying the Gold discount?")


