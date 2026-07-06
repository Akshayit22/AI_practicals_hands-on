from dotenv import load_dotenv
import os
import requests

load_dotenv()

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print


# weather tools
@tool
def get_weather(city: str) -> str:
    """get_weather tool"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()
    # print("Debug",data)
    
    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"
    
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    
    return f"Weather in {city}: {desc}, {temp}°C"


# response = get_weather.invoke("Nashik")
# print(response)

# News tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    "get latest news for city"
    
    response = tavily_client.search(
        query=f"fetch latest news in city: {city} in english",
        search_depth="basic",
        max_results=3
    )
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    return results
        
# response = get_news.invoke("Nashik")
# print(response)

# llm

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather,get_news])

# Agent Loop - very important
messages = []

print("City intelligence system")
print("type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)
        
        # check if tool is required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                # human in the loop
                confirm = input(f"Agent want to user {tool_name} Agent, Approce (yes/no)")
                if confirm.lower() == "no":
                    print()
                    break
                
                # excute tool
                tool_result = tools[tool_name].invoke(tool_call)
                
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id = tool_call['id']
                ))
                
            continue
        else:
            print(result.content)
            break