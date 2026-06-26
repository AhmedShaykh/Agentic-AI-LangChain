from langchain_core.messages import HumanMessage;
from langchain_mistralai import ChatMistralAI;
from langchain.tools import tool;
from dotenv import load_dotenv;
from rich import print;

load_dotenv();

@tool # Create Custom Tool
def get_text_length(text: str) -> int: # Return Type

    """Returns The Number Of Character In A Given Text"""; # Docstring Need To Read LLM Help

    return len(text);

llm = ChatMistralAI(model = "mistral-small-2506");

llm_forced = llm.bind_tools([get_text_length], tool_choice="any"); # Always Calls Tool

llm_with_tool   = llm.bind_tools([get_text_length]); # Bind Tool

messages = [];

prompt = input("Enter Your Text: ");

query = HumanMessage(f"What Is The Character Length Of The Following Text: `{prompt}`");

messages.append(query);

result = llm_forced.invoke(messages);

print(result);

messages.append(result);

if result.tool_calls: # Execute Tool

    tool_message = get_text_length.invoke(result.tool_calls[0]);

    print(tool_message);

    messages.append(tool_message);

result = llm_with_tool.invoke(messages);

print("Final Result: " , result.content);