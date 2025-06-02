import os
import yfinance as yf
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
####################################################
#                     Tools                        #
####################################################
def get_last_month_prices(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1mo", interval="1d")
        if data.empty:
            return f"Ticker '{ticker}' not found or has no data for the last month."
        return data
    except Exception as e:
        return f"Error fetching data for '{ticker}': {str(e)}"
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_last_month_prices",
            "description": "Gets daily prices for the last month for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL' for Apple)."
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

function_map = {
    "get_last_month_prices": lambda args: get_last_month_prices(args["ticker"]),
}

def chat_with_gpt(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_messages = []
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func = function_map.get(func_name)
            arguments = json.loads(tool_call.function.arguments)
            result = func(arguments)

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        messages.append(message.model_dump())  # Add GPT's tool_call message
        messages.extend(tool_messages)         # Add tool response messages

        return chat_with_gpt(messages)         # Let GPT continue with tool results

    return message.content


def main():
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    print("Start chatting with the bot (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        reply = chat_with_gpt(messages)
        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)

if __name__ == "__main__":
    main()
