from langchain_core.output_parsers import PydanticOutputParser;
from langchain_core.prompts import ChatPromptTemplate;
from langchain_mistralai import ChatMistralAI;
from typing import List, Optional;
from pydantic import BaseModel;
from dotenv import load_dotenv;
import json;

load_dotenv();

model = ChatMistralAI(model="mistral-small-2603");

class Movie(BaseModel):
    title: str;
    release_year : Optional[int];
    genre: List[str];
    director: Optional[str];
    cast: List[str];
    rating: Optional[float];
    summary: str;

parser = PydanticOutputParser(pydantic_object=Movie);

prompt = ChatPromptTemplate.from_messages([
    ("system", """
                Extract Movie Information From The paragraph.
                If the title is not explicitly mentioned, infer it from context.
                If truly unknown, use "Unknown".
                {format_instructions}
    """),
    ("human", "{paragraph}")
]);

para = input("Enter Movie Paragraph: ");

final_prompt = prompt.invoke({ 
    "paragraph" : para,
    "format_instructions": parser.get_format_instructions()
});

response = model.invoke(final_prompt)

movie_data = parser.parse(response.content)

print(json.dumps(movie_data.model_dump(), indent=2));