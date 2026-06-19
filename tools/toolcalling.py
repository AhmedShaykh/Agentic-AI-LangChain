from langchain_core.messages import HumanMessage;
from langchain_mistralai import ChatMistralAI;
from langchain.tools import tool;
from dotenv import load_dotenv;
from rich import print;

load_dotenv();

@tool # Create Custom Tool
def get_text_length(text: str) -> int: # Return Type

    """Returns The Number Of Character In A Given Text"""; # Docstring

    return len(text);

tools = { "get_text_length" : get_text_length };

llm = ChatMistralAI(model = "mistral-small-2506");

llm_with_tool = llm.bind_tools([get_text_length]); # Bind Tool

llm_with_tool_forced = llm.bind_tools([get_text_length], tool_choice = "any"); # Force Tool Call

messages = [];

prompt = input("Enter Your Text: ");

query = HumanMessage(f"What Is The Character Length Of The Following Text: `{prompt}`");

messages.append(query);

print(query);

result = llm_with_tool_forced.invoke(messages);

print(result);

messages.append(result);

if result.tool_calls: # Execute Tool

    tool_name = result.tool_calls[0]["name"];

    tool_message = tools[tool_name].invoke(result.tool_calls[0]);

    print(tool_name, tool_message);

    messages.append(tool_message);

result = llm_with_tool.invoke(messages);

print("Final Result: " , result.content);