from langchain_core.messages import HumanMessage, ToolMessage;
from langchain_mistralai import ChatMistralAI;
from langchain.tools import tool;
from tavily import TavilyClient;
from dotenv import load_dotenv;
from rich import print;
import requests;
import os;

load_dotenv();

@tool
def get_weather(city: str) -> str:

    """Get Current Weather Of A City"""

    API_KEY = os.getenv("OPENWEATHER_API_KEY");

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric";

    response = requests.get(url);

    data = response.json();

    if str(data.get("cod")) != "200":

        return f"Error: {data.get("message", "Could Not Fetch Weather API")}";

    temp = data["main"]["temp"];

    desc = data["weather"][0]["description"];

    return f"Weather In {city}: {desc}, {temp}°C";

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"));

@tool
def get_news(city: str) -> str:

    """Get Latest News About A City"""

    response = tavily_client.search(
        query=f"Latest News In {city}",
        search_depth="basic",
        max_results=3
    );

    results = response.get("results", []);

    if not results:

        return f"No News Found For {city}";

    news_list = [];

    for r in results:

        title = r.get("title", "No title");

        url = r.get("url", "");

        snippet = r.get("content", "");

        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...");

    return f"Latest News In {city}:\n\n" + "\n\n".join(news_list);

llm = ChatMistralAI(model="mistral-small-2506");

tools = { "get_weather": get_weather, "get_news": get_news };

llm_with_tool = llm.bind_tools([get_weather, get_news]);

message = [];

print("Type Your City | Type Exit To Quit");

while True:

    user_input = input("You : ");

    if user_input.lower() == "exit":

        break;

    message.append(HumanMessage(content=user_input));

    while True:

        result = llm_with_tool.invoke(message);

        message.append(result);

        if result.tool_calls:

            all_denied = True;

            for tool_call in result.tool_calls:

                tool_name = tool_call["name"];

                confirm = input(f"Agent Wants To Call {tool_name} Approve (yes/no): "); # Human In The Loop

                if confirm.lower() == "no":

                    print("Tool Call Denied");

                    message.append(ToolMessage(
                        content="User Denied This Tool Call. Answer Without Using This Tool.",
                        tool_call_id=tool_call["id"]
                    ));

                else:

                    all_denied = False;

                    tool_result = tools[tool_name].invoke(tool_call);

                    message.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    ));

            continue;

        else:

            print(result.content);

            break;