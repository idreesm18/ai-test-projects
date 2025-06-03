import os
import yfinance as yf
import pandas as pd
import json
from prompts import ANALYST_ROLES, YFINANCE_DOCS_SUMMARY
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

####################################################
#                     Tools                        #
####################################################

def fetch_yfinance_data(ticker, method="history", params=None):
    try:
        t = yf.Ticker(ticker)
        if not hasattr(t, method):
            return f"Method '{method}' is not valid for yfinance.Ticker."

        attr = getattr(t, method)
        result = attr(**params) if callable(attr) else attr

        if isinstance(result, (pd.DataFrame, pd.Series)):
            if result.empty:
                return f"No data returned by {method} for {ticker}."
            return result.head(25).to_string()

        return str(result)

    except Exception as e:
        return f"Error fetching {method} for {ticker}: {str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_yfinance_data",
            "description": "Fetches various data from yfinance using the Ticker object. Specify the method and optional parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker (e.g., 'AAPL')."
                    },
                    "method": {
                        "type": "string",
                        "description": "yfinance method/attribute to access (e.g., 'history', 'financials', 'info')."
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional parameters to pass to the method (e.g., {'period': '1mo'})."
                    }
                },
                "required": ["ticker", "method"]
            }
        }
    }
]

function_map = {
    "fetch_yfinance_data": lambda args: fetch_yfinance_data(
        args["ticker"], args["method"], args.get("params", {})
    ),
}

####################################################
#                Chat Interaction                  #
####################################################

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

        messages.append(message.model_dump())
        messages.extend(tool_messages)
        return chat_with_gpt(messages)

    return message.content

####################################################
#                      Log                         #
####################################################

def log_conversation(user_msg, assistant_msg, filename="chat_log.md"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"**User:** {user_msg}\n\n")
        f.write(f"**Assistant:** {assistant_msg}\n\n---\n\n")

####################################################
#                     Main                         #
####################################################

def main():
    print("Choose analyst role (macro / technical / quant / general):")
    role_key = input(" > ").strip().lower()
    role_description = ANALYST_ROLES.get(role_key, ANALYST_ROLES["general"])

    system_prompt = f"{role_description}\n\n{YFINANCE_DOCS_SUMMARY}"
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\nSet role: {role_key.capitalize()} Analyst\n")
    print("Start chatting with the bot (type 'exit' to stop):")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})
        reply = chat_with_gpt(messages)
        messages.append({"role": "assistant", "content": reply})
        print("Bot:", reply)

        log_conversation(user_input, reply)

if __name__ == "__main__":
    main()
