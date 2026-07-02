import re
import inspect
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
    float(price)
    return price - price*discounts.get(discount_tier, 0)

tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools_dict)
tool_names = ",".join(tools_dict.keys())
pass

react_prompt = f""" 
    STRICT RULES - DO NOT guess or assume product price.
    DO not guess of assume discount rates.
    Only call apply_discount after you have called get_product_price
    Answer the following questions as best you can. You have access to the following tools:

    {tool_descriptions}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought:""
"""

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable (name="LangChain Agent Loop")
def run_agent(question: str): 
    prompt = react_prompt.format(question=question)
    scratchpad = ""
    for iteration in range(1, MAX_ITERATIONS+1):
        full_prompt = prompt + scratchpad
        response = ollama_chat_trace(model=MODEL, messages=[{"role": "user", "content": full_prompt}], options = {"stop": ["\nObservation"], "temperature": 0},)
        output = response.message.content

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("final answer\n")
            print(final_answer)


        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not (action_match or action_input_match):
            print("Error")
            break
    

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools_dict:
            observation = f"{tool_name} not found. These are the tools available: {list[str](tools_dict.keys())}"
        else:
            observation = str(tools_dict[tool_name](*args))

        scratchpad = f"{output}\nObservation: {observation}\nThought:"

        print(observation)

    print("maxed out")



    

if __name__ == "__main__":
    print("Hello LangChain agent (.bind_tools)!")
    print()
    result = run_agent("what is the price of a laptop after applying the Gold discount?")



