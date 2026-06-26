from langchain_core.messages import AIMessage, SystemMessage, HumanMessage;
from langchain_mistralai import ChatMistralAI;
from dotenv import load_dotenv;
from rich import print;

load_dotenv();

model = ChatMistralAI(
    model="mistral-small-2603",
    temperature=0.6
);

print("Press 1 For Angry Mode");

print("Press 2 For Funny Mode");

print("Press 3 For Sad Mode");

choice = int(input("Enter Your AI Mode: "))

if choice == 1:

    mode = "You Are An Angry AI Agent. You Respond Aggressively & Impatiently.";

elif choice == 2:

    mode = "You Are An Funny AI Agent. You Respond With Humor & Jokes.";

elif choice == 3:

    mode = "You Are An Sad AI Agent. You Respond In A Depressed & Amotional Tone.";

messages = [ # Memory
    SystemMessage(content=mode)
];

print("Welcome To Application & Enter 0 To Exit");

while True:

    prompt = input("You : ");

    if prompt == "0":

        break;

    messages.append(HumanMessage(content=prompt));

    response = model.invoke(messages);

    messages.append(AIMessage(content=response.content));

    print(f"Bot : {response.content}");

print(messages);