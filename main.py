import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
 
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
 
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
model_client = OpenAIChatCompletionClient(
    model="gemini-3.1-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model_info={
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    },
)

# ---------------- TOOLS ----------------
def track_order(order_id: str) -> str:
    """Find an order by its ID, for example 1005"""
    orders = pd.read_csv("orders.csv", dtype=str).fillna("")
    order_id = order_id.replace("#", "").strip()
 
    result = orders[orders["order_id"] == order_id]
    if result.empty:
        return "Order not found."
    return str(result.to_dict("records"))


def search_product(keyword: str) -> str:
    """Search products by ONE keyword, for example: kurta, shirt, black, hoodie"""
    products = pd.read_csv("products.csv")
 
    result = products[
        products["name"].str.contains(keyword, case=False)
        | products["category"].str.contains(keyword, case=False)
    ]
    if result.empty:
        return "No product found."
    return str(result.to_dict("records"))

receptionist = AssistantAgent(
    name="receptionist",
    model_client=model_client,
    handoffs=["order_tracker", "product_advisor"],
    system_message="""You are the receptionist of an online clothing store.
If the customer asks about an order, hand off to order_tracker.
If the customer asks about products, price, size or stock, hand off to product_advisor.""",
)

order_tracker = AssistantAgent(
    name="order_tracker",
    model_client=model_client,
    tools=[track_order],
    reflect_on_tool_use=True,
    system_message="""You help customers track orders.
Use the track_order tool. Tell the status, courier and delivery date.
Reply in the customer's language. At the end write TERMINATE.""",
)

product_advisor = AssistantAgent(
    name="product_advisor",
    model_client=model_client,
    tools=[search_product],
    reflect_on_tool_use=True,
    system_message="""You help customers find products.
Use the search_product tool. Tell the price and which sizes are in stock
(stock_S, stock_M, stock_L, stock_XL; 0 means out of stock).
Reply in the customer's language. At the end write TERMINATE.""",
)


team = Swarm(
    participants=[receptionist, order_tracker, product_advisor],
    termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(10),
)

async def main():
    while True:
        question = input("\nCustomer (type exit to quit): ")
        if question.lower() == "exit":
            break
        await Console(team.run_stream(task=question))
        await team.reset()   # start fresh for the next question
 
    await model_client.close()
 
 
asyncio.run(main())