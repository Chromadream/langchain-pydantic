from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.stdout import StdOutCallbackHandler
from pydanticchain import PydanticChain
from langchain.prompts.prompt import PromptTemplate

from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

class DocumentMetadata(BaseModel):
    title      : Optional[str] = Field(description='The title of the content')
    author     : Optional[str] = Field(description='The author of the content')
    created_at : Optional[str] = Field(description='The date in the format YYYY-MM-DD that the content was created if it appears in the content')
    language   : str           = Field(description='The 2 character ISO 639-1 language code of the primary language of the content')
    summary    : str           = Field(description='Summary of the content')

def main():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = "Extract metadata from the following content: {content}"
    content = """
Updated 24th March 2023
Generative AI - Chapter 1: Establishing the Investment Framework
BlackLake Equity Research
AI holds the potential to give rise to new enterprises and furnish existing players with fresh growth opportunities by greatly enhancing end-user productivity."""
    chain = PydanticChain(llm=llm, prompt=PromptTemplate.from_template(prompt), model_class=DocumentMetadata)
    response = chain.run({'content': content}, callbacks=[StdOutCallbackHandler()])
    print(response.json())
    

if __name__ == "__main__":
    main()